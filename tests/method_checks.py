"""CPU checks of the training methods, one per process (each loads its own reference trainer; see refs.trainer_module).

    python tests/method_checks.py <check> <scratch dir>

Two test-only patches make them run on a CPU: dense attention with the same causal window instead of flex attention
(which has no CPU backward, and on CPU sees float32 values under autocast), and sequences cut to SEQ tokens. The Cut
chain's checks also shrink the vocabulary (CUT_VOCAB): its head matmuls are bfloat16, which this CPU runs slowly.
"""
import copy
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import torch

from loud import data, refs, train

SEQ = 129


def patch_attention():
    um = sys.modules['utils.model']

    def dense_attention(q, k, v, window):  # written out, so it also has a double backward
        i = torch.arange(q.shape[1])
        mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] <= window)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.to(q.dtype).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).masked_fill(~mask, float('-inf'))
        return (scores.softmax(-1) @ v).transpose(1, 2)
    um.attention = dense_attention


def patch_sequences(length=SEQ, vocab=None):
    real = data.benchmark_split

    def short(*args, **kwargs):
        split = real(*args, **kwargs)
        split.train, split.heldout = split.train[:, :length].contiguous(), split.heldout[:, :length].contiguous()
        if vocab is not None:
            split.train, split.heldout = split.train % vocab, split.heldout % vocab
        return split
    train.benchmark_split, train.SEQ = short, length


def proxy_gradient(scratch):
    """micro_batch 1 is the float32 mean of per-sequence gradients; micro_batch = batch is one direct backward."""
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    o = proxy_options(dict(preset='2m', batch=4, micro_batch=1))
    trainer = Backprop(o, reference, torch.device('cpu'), 0, 10)
    model, params = trainer.model, list(trainer.model.parameters())
    batch = torch.randint(0, model.config.vocab_size, (4, SEQ), generator=torch.Generator().manual_seed(1))

    def direct(rows):
        model.zero_grad(set_to_none=True)  # the trainer leaves .grad aliasing its accumulators
        model.requires_grad_(True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = model(rows[:, :-1].contiguous(), rows[:, 1:].contiguous())
        loss.backward()
        grads = [torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.float().clone() for p in params]
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        return loss.detach().float(), grads

    per_sequence = [direct(batch[i:i + 1]) for i in range(4)]
    expected = [sum(g[j] for _, g in per_sequence) / 4 for j in range(len(params))]
    loss = trainer.gradient(batch)
    assert torch.allclose(loss, torch.stack([l for l, _ in per_sequence]).mean())
    for p, e in zip(params, expected):
        assert torch.equal(p.grad, e.to(p.dtype)), 'micro_batch 1 is not the float32 mean of per-sequence gradients'

    whole_loss, whole = direct(batch)
    trainer.o = dict(o, micro_batch=4)
    loss = trainer.gradient(batch)
    assert torch.equal(loss, whole_loss)
    for p, e in zip(params, whole):
        assert torch.equal(p.grad, e.to(p.dtype)), 'micro_batch = batch is not one direct backward'

    for steps in (606, 610, 1212):  # backprop_preset.py: lr = initial * min(1, left / warmdown), one-based steps
        group = dict(lr=1.0)
        optimizer = type('O', (), dict(param_groups=[group]))
        for step in (1, steps // 2, steps - 1, steps):
            train.set_rate(optimizer, step, steps, 0.3)
            assert group['lr'] == min(1.0, ((steps - step + 1) / steps) / 0.3)


def proxy_run(scratch):
    """The whole loop on a tiny model: every H1-H4 output exists and is consistent."""
    from loud.proxy import run
    refs.trainer_module('boostcut')
    patch_attention()
    patch_sequences()
    spec = dict(experiment='cpu', variant='proxy', seed=0, task='proxy', args=dict(
        preset='2m', sequences=48, heldout=4, batch=4, checkpoints=True, checkpoint_optimizer=True, heldout_every=2,
        means=True, save_means=True, per_sequence_grads=True, grad_noise_every=1, official_eval=False))
    out = Path(scratch) / 'proxy'
    result = run(spec, out)
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and result['updates'] == 11 and len(metrics) == 11
    assert all({'trace_cov', 'true_sq', 'b_simple'} <= set(row) for m in metrics for row in m['grad_noise'].values())
    assert {f'ckpt_{f:02d}.pt' for f in (0, 10, 20, 50, 100)} <= {p.name for p in out.iterdir()}
    optim = torch.load(out / 'ckpt_50.optim.pt', weights_only=True)
    assert optim['state'] and all({'fast', 'slow'} <= set(v) for v in optim['state'].values())
    # means from 60% of the epoch: updates 10 and 11 (every 5, and the last); each then differs from the final weights
    assert result['means_updates'] == 2 and len(result['means_heldout']) == 9 and (out / 'means.pt').exists()
    assert all(v != result['heldout_loss'] for v in result['means_heldout'].values())
    heldout = [json.loads(line) for line in (out / 'heldout.jsonl').read_text().splitlines()]
    assert [h['step'] for h in heldout if 'mean_decay' not in h].count(11) == 1  # the last update is scored once


def identity_window(scratch):
    """rows = width runs PR #26's probe() itself; the K row windows, each scaled by K, average to the full cycle."""
    from loud.identity import estimator_class
    from loud.options import PRESETS
    m = refs.trainer_module('pr26')
    n_layer, width = PRESETS['tiny']
    config = m.GPTConfig(sequence_len=SEQ, n_layer=n_layer, n_head=2, n_kv_head=2, n_embd=width)
    model = m.make_model(torch.device('cpu'), 0, config)
    settings = m.Settings(fc_k=width // 2, route_k=width, no_chain=True, skip_mid=True)
    x = torch.randn(16, width, generator=torch.Generator().manual_seed(2))
    y = torch.randint(0, config.vocab_size, (16,), generator=torch.Generator().manual_seed(3))
    ours, theirs = estimator_class(m)(model, settings, 0, 0, 1), m.FastRelay(model, settings, 0, 0, 1)
    state = ours.generator.get_state()
    full = ours.probe(x, y)
    assert torch.equal(full, theirs.probe(x, y)), 'the full cycle is not PR #26 probe()'
    parts = []
    for k in range(4):
        ours.generator.set_state(state)
        ours.rows, ours.start = width // 4, k * width // 4
        parts.append(ours.probe(x, y))
    assert torch.allclose(sum(parts) / 4, full, rtol=1e-4, atol=1e-6), 'row windows do not average to the cycle'
    # without rewinding: the windows of one cycle read one basis (and still draw signs as PR #26 does)
    ours.generator.set_state(state)
    ours.cycle, together = 7, []
    for k in range(4):
        ours.start = k * width // 4
        together.append(ours.probe(x, y))
    assert torch.allclose(sum(together) / 4, full, rtol=1e-4, atol=1e-6), 'a cycle does not hold its basis'
    ours.cycle = 8
    assert not torch.allclose(ours.probe(x, y), together[-1]), 'a new cycle does not draw a new basis'


def identity_step(scratch):
    """An IdentityPath update over two virtual GPUs is PR #26's: one FastRelay per rank (its own seeds) on one
    sequence each, then Boost.step's sum of the per-sequence gradients over token counts, divided by the ranks."""
    from loud.identity import IdentityPath
    from loud.options import identity_options
    m = refs.trainer_module('pr26')
    patch_attention()
    o = identity_options(dict(preset='tiny', batch=2))
    ours = IdentityPath(o, torch.device('cpu'), 0, 10, torch.zeros(1, SEQ, dtype=torch.long))
    model = m.make_model(torch.device('cpu'), 0, ours.config)
    optimizer = m.make_optimizer(model, ours.recipe, 10)
    ranks = [m.FastRelay(model, ours.settings, 0, rank, 2) for rank in range(2)]
    batch = torch.randint(0, ours.config.vocab_size, (2, SEQ), generator=torch.Generator().manual_seed(4))
    loss, _ = ours.update(1, batch)
    flats, losses = [], []
    for rank, estimator in enumerate(ranks):  # Boost.step on each rank, its all-reduce as a sum
        x, y = batch[rank:rank + 1, :-1].contiguous(), batch[rank:rank + 1, 1:].contiguous()
        gradients = estimator.estimate(x, y)
        flats.append(torch.cat([(gradients[n] / (y != -1).sum()).float().reshape(-1)
                                for n, _ in model.named_parameters()]))
        losses.append(estimator.capture.clean_loss[y != -1].mean())
    flat, offset = sum(flats) / 2, 0
    for p in model.parameters():
        p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
        offset += p.numel()
    optimizer.step()
    assert abs(loss.item() - torch.stack(losses).mean().item()) < 1e-6
    for (name, a), b in zip(ours.model.named_parameters(), model.parameters()):
        assert torch.equal(a, b), f'{name} differs after one update'


def identity_run(scratch):
    """The whole loop with E6's rotating row windows and the drift check."""
    from loud.identity import run
    refs.trainer_module('pr26')
    patch_attention()
    patch_sequences()
    spec = dict(experiment='cpu', variant='identity', seed=0, task='identity', args=dict(
        preset='tiny', sequences=24, heldout=4, batch=4, rows=32, row_offset_per_step=True, drift_every=1,
        checkpoints=True, means=True, official_eval=False))
    out = Path(scratch) / 'identity'
    result = run(spec, out)
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and len(metrics) == 5
    assert [m['probe_rows'][0] for m in metrics] == [(s % 4) * 32 for s in range(1, 6)]
    assert all(-1 <= m['drift'][k] <= 1 for m in metrics for k in ('blend', 'fast', 'slow'))


def diagnostics(scratch):
    """H6 on a tiny model whose writers (attention and MLP c_proj) are zero: the identity path is then exact, so its
    credit guess must match autograd at every site, and each token's map to a site is a scaled RMS norm (equal
    singular values but the radial one)."""
    from loud.diagnostics import Toolkit, site_names
    from loud.options import PRESETS
    m = refs.trainer_module('pr26')
    patch_attention()
    n_layer, width = PRESETS['tiny']
    config = m.GPTConfig(sequence_len=SEQ, n_layer=n_layer, n_head=2, n_kv_head=2, n_embd=width)
    model = m.make_model(torch.device('cpu'), 0, config)
    with torch.no_grad():
        for block in model.transformer.h:
            block.attn.c_proj.weight.zero_()
            block.mlp.c_proj.weight.zero_()
        model.resid_lambdas.copy_(torch.linspace(0.8, 1.2, n_layer))
    settings = m.Settings(fc_k=width // 2, route_k=width, no_chain=True, skip_mid=True)
    row = torch.randint(0, config.vocab_size, (2, SEQ), generator=torch.Generator().manual_seed(5))
    tools = Toolkit(m, model, settings, row[:1, :-1].contiguous(), row[:1, 1:].contiguous())
    cos = tools.cos_to_autograd()[0]
    assert abs(cos['all']) <= 1 + 1e-3 and 'block_0' in cos  # float32 rounding over long vectors
    assert cos['attn.c_proj'] > 0.999 and cos['mlp.c_proj'] > 0.999  # the writers' credit is exact here
    summary, tensors = tools.credit_dump()
    for site in site_names(n_layer)[1:]:
        assert tensors[site]['true'].shape == (SEQ - 1, width)
        assert summary[site]['p10'] > 0.99, (site, summary[site])  # exact up to the probe's finite differences
    svd, _ = tools.token_map_svd(chunk=64)
    for site in site_names(n_layer)[1:]:
        assert svd[site]['condition_without_radial']['p90'] < 1.01 and svd[site]['condition']['median'] > 1e3
    linear, _ = tools.linearity_check(sigmas=(1e-3, 1e-1), draws=1)
    assert set(linear) == set(site_names(n_layer))
    assert all(r[f'{p} 0.001']['nonlinearity']['median'] >= 0 for r in linear.values() for p in ('float32', 'bf16'))

    # direction_stats on a model with live writers: its finite-difference curvature against an exact
    # Hessian-vector product (double backward works with the dense attention used here), and the pending update
    model = m.make_model(torch.device('cpu'), 0, config)
    tools = Toolkit(m, model, settings, row[:1, :-1].contiguous(), row[:1, 1:].contiguous())
    name = 'transformer.h.1.mlp.c_fc.weight'
    stats, _ = tools.direction_stats(row, directions=2)
    assert stats['directions_from'] == 'gradient' and stats['sequences'] == 2
    exact_model = copy.deepcopy(model).float().requires_grad_(True)
    weight = dict(exact_model.named_parameters())[name]
    with torch.enable_grad():
        grads = []
        for i in range(2):
            out = exact_model(row[i:i + 1, :-1].contiguous(), row[i:i + 1, 1:].contiguous(), loss_reduction='sum')
            grads.append(torch.autograd.grad(out, weight)[0])
        u, _, vh = torch.linalg.svd(sum(grads) / 2, full_matrices=False)
        d = torch.outer(u[:, 0], vh[0])
        total = exact_model(row[:, :-1].contiguous(), row[:, 1:].contiguous(), loss_reduction='sum') / 2
        g = torch.autograd.grad(total, weight, create_graph=True)[0]
        hvp = torch.autograd.grad((g * d).sum(), weight)[0]
    assert abs(stats[name][0]['curvature'] - (hvp * d).sum().item()) < 2e-2 * abs((hvp * d).sum().item()) + 1e-6
    names = [n for n in stats if n.startswith('transformer.')]
    fake = dict(state={n: dict(fast=torch.randn_like(dict(model.named_parameters())[n]),
                               slow=torch.randn_like(dict(model.named_parameters())[n])) for n in names},
                beta=0.9, fast_weight=0.5)
    pending, _ = tools.direction_stats(row, fake, directions=1)
    assert pending['directions_from'] == 'pending update'
    assert all(len(pending[n]) == 1 and pending[n][0]['noise'] >= 0 for n in names)


CUT_SEQ, CUT_VOCAB = 65, 512  # the Cut checks: 64 predictions per sequence, a 512-token vocabulary
CUT_ROWS = 72                   # their data: 40 sequences feed 10 updates of 4 (2 Boost), 32 are held out


def cut_module():
    """PR #40 run eagerly (both sides of a comparison) with dense attention, and loud.cut.record_settings shrunk to
    CUT_SEQ tokens and CUT_VOCAB."""
    import torch._dynamo
    torch._dynamo.config.disable = True
    m = refs.trainer_module('pr40')
    patch_attention()
    m.attention = sys.modules['utils.model'].attention  # the script's own reference to the flex attention
    import loud.cut
    real = loud.cut.record_settings

    def small(*args):
        s = real(*args)
        s.model.sequence_len, s.model.vocab_size = CUT_SEQ - 1, CUT_VOCAB
        return s
    loud.cut.record_settings = small
    return m


def cut_rows_data():
    return torch.randint(0, CUT_VOCAB, (CUT_ROWS, CUT_SEQ), generator=torch.Generator().manual_seed(6))


def cut_rank(scratch, rank):
    """One of 8 gloo processes running PR #40's own train_stages (its loop, groups and all-reduces) for 4 updates:
    2 Boost, then the switch and 2 Cut. Each all-reduce sums in rank order, as the port does, so the two can agree to
    the bit. Its CUDA calls become no-ops; load_rows returns cut_rows_data()."""
    import argparse
    import loud.cut
    m = cut_module()
    dist = torch.distributed
    dist.init_process_group('gloo', rank=int(rank), world_size=8)

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        assert op == dist.ReduceOp.SUM and not async_op
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        total = parts[0].clone()
        for part in parts[1:]:
            total += part
        tensor.copy_(total)
    dist.all_reduce = all_reduce
    for name in ('synchronize', 'reset_peak_memory_stats'):
        setattr(torch.cuda, name, lambda *a, **k: None)
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    m.load_rows = lambda path, rows: cut_rows_data()
    args = argparse.Namespace(data_dir=Path(scratch), output=Path(scratch) / 'record', steps=None, stop_after_step=4,
                              resume=None, eval_every=0, checkpoint_every=0)
    m.train_stages(args, loud.cut.record_settings(m, '2m', 4), torch.device('cpu'), int(rank), 8)
    torch.distributed.destroy_process_group()


def cut_ranks(scratch):
    """The single-process port (8 virtual GPUs, two per sequence taking turns) against PR #40's train_stages on 8 real
    gloo processes, both eager in float32: bit-identical per-update losses and weights after 2 Boost and 2 Cut
    updates (the processes' all-reduces sum in rank order; on GPUs NCCL's order changes the rounding)."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), TORCHDYNAMO_DISABLE='1')
    procs = [subprocess.Popen([sys.executable, __file__, 'cut_rank', scratch, str(r)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in range(8)]
    m = cut_module()
    from loud.cut import CutChain
    from loud.options import cut_options
    rows = cut_rows_data()
    order = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(1))]  # the record's order
    training, heldout = order[:-32], order[-32:]
    o = cut_options(dict(preset='2m', batch=4, sequences=CUT_ROWS, heldout=32))
    ours = CutChain(o, torch.device('cpu'), 0, len(training) // 4, heldout)
    losses = [ours.update(step, training[(step - 1) * 4:step * 4])[0].item() for step in range(1, 5)]
    for proc in procs:
        out, _ = proc.communicate(timeout=1800)
        assert proc.returncode == 0, out[-4000:]
    record = [json.loads(line) for line in (Path(scratch) / 'record' / 'metrics.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in record] == ['boost', 'boost', 'cut', 'cut']
    theirs = torch.load(Path(scratch) / 'record' / 'checkpoint.pt', weights_only=True)['model']
    for step, (a, b) in enumerate(zip(losses, [r['train_loss'] for r in record]), 1):
        assert a == b, f'update {step}: loss {a} here, {b} on 8 processes'
    for name, p in ours.model.state_dict().items():
        assert torch.equal(p, theirs[name]), f'{name} differs by {(p.float() - theirs[name].float()).abs().max()}'


def cut_windows(scratch):
    """rows = full is PR #40's Cut estimate; at fewer rows, the K windows of a held cycle (each scaled by K) average
    to the full cycle's chain credits, the tiled windows (E7) give each token its window's credit, and the GPUs of a
    sequence split each token's window into halves."""
    m = cut_module()
    from loud.cut import estimator_classes, record_settings
    s = record_settings(m, '2m', 4)
    width = s.model.n_embd
    model = m.make_model(torch.device('cpu'), 0, s.model, s.projection_std, s.head_std)
    _, Cut = estimator_classes(m)
    row = cut_rows_data()[:1]
    x, y = row[:, :-1].contiguous(), row[:, 1:].contiguous()
    ours, theirs = Cut(model, s, 0, 0, 1, 1, None), m.Cut(model, s, 0, 0, 1, 1, None)
    a, b = ours.estimate(x, y), theirs.estimate(x, y)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), 'the full cycle is not the record'

    saved = {}

    def keep(self, *args):
        saved['args'] = args
        return Cut.chain_credits(self, *args)
    ours.chain_credits = types.MethodType(keep, ours)
    ours.estimate(x, y)
    del ours.chain_credits
    state = ours.generator.get_state()
    full = ours.chain_credits(*saved['args'])
    K, rows = 4, width // 4
    ours.generator.set_state(state)  # the cycle's held bases are then the ones the full cycle drew
    ours.rows, ours.cycle, windows = rows, 7, []
    for k in range(K):
        ours.start = k * rows
        windows.append(ours.chain_credits(*saved['args']))
    ours.start = None
    tiled = ours.chain_credits(*saved['args'])
    tokens = torch.arange(x.shape[1])
    for key, value in full.items():
        mean = sum(w[key] for w in windows) / K
        assert torch.allclose(mean, value, rtol=1e-4, atol=1e-6), f'{key}: the windows do not average to the cycle'
        if not isinstance(key, tuple):  # ('lam', site): the mixing scalars' probes do not use the rows
            each = torch.stack([w[key] for w in windows])[tokens % K, tokens]
            assert torch.allclose(tiled[key], each, rtol=1e-4, atol=1e-6), f'{key}: a tiled token misses its window'
    halves = [Cut(model, s, 0, part, 2, 2, None) for part in range(2)]
    for start in (0, None):
        for half in halves:
            half.rows, half.start = rows, start
        index = torch.cat([h.row_index(8) for h in halves], 1)
        first = torch.zeros(8, dtype=torch.long) if start == 0 else torch.arange(8) * rows
        assert torch.equal(index, (first[:, None] + torch.arange(rows)) % width), 'the halves do not tile a window'


def cut_run(scratch):
    """The whole loop on the 2-layer model (width 128), two virtual GPUs on one sequence per update: Boost, the switch,
    E7's ramp (64 rows until it reaches the full cycle at 60%), the drift check, H4, the means (updates 10 and 12) and
    the record's end merge, timed."""
    cut_module()
    from loud.cut import run
    patch_sequences(CUT_SEQ, CUT_VOCAB)
    spec = dict(experiment='cpu', variant='cut', seed=0, task='cut', args=dict(
        preset='2m', sequences=20, heldout=8, batch=1, virtual_gpus=2, rows_start=64, rows_full_at=0.6,
        drift_every=3, checkpoints=True, heldout_every=4, official_eval=False))
    out = Path(scratch) / 'cut'
    result = run(spec, out)
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and len(metrics) == 12
    assert [r['phase'] for r in metrics] == ['boost'] * 2 + ['cut'] * 10
    assert [r.get('chain_rows') for r in metrics] == [None] * 2 + [64] * 5 + [128] * 5
    assert [r['step'] for r in metrics if 'drift' in r] == [3, 6, 9, 12]
    assert all(-1 <= v <= 1 for r in metrics if 'drift' in r for v in r['drift'].values())
    assert {'holdout_plain', 'holdout_merged', 'merge_choices', 'merge_seconds'} <= set(result)
    assert result['holdout_merged'] <= result['holdout_plain'] and result['means_updates'] == 2
    assert result['train_seconds'] >= result['merge_seconds']
    assert {'final.pt', 'final_plain.pt', 'ckpt_100.pt'} <= {p.name for p in out.iterdir()}


if __name__ == '__main__':
    torch.set_num_threads(2 if sys.argv[1] == 'cut_rank' else 8)  # every core would oversubscribe a shared machine
    torch.manual_seed(0)
    globals()[sys.argv[1]](*sys.argv[2:])
    print('ok')

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


def patch_sequences(*loops, length=SEQ, vocab=None):
    real = data.benchmark_split

    def short(*args, **kwargs):
        split = real(*args, **kwargs)
        split.train, split.heldout = split.train[:, :length].contiguous(), split.heldout[:, :length].contiguous()
        if vocab is not None:
            split.train, split.heldout = split.train % vocab, split.heldout % vocab
        return split
    for loop in (train, *loops):
        loop.benchmark_split, loop.SEQ = short, length


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


def proxy_merge(scratch):
    """The proxy with PR #40's recipe and its end merge: timed, chosen on the held-out rows, the plain weights kept
    and scored apart."""
    from loud.proxy import run
    refs.trainer_module('pr40')
    patch_attention()
    patch_sequences()
    spec = dict(experiment='cpu', variant='merge', seed=0, task='proxy', args=dict(
        recipe='pr40', preset='2m', sequences=48, heldout=4, batch=4, means=True, end_merge=True, official_eval=False))
    out = Path(scratch) / 'merge'
    result = run(spec, out)
    assert result['complete'] and result['merge_seconds'] > 0 and (out / 'final_plain.pt').exists()
    assert result['holdout_merged'] <= result['holdout_plain'] and result['merge_choices']
    merged, plain = (torch.load(out / f, weights_only=True) for f in ('final.pt', 'final_plain.pt'))
    assert (result['holdout_merged'] < result['holdout_plain']) == any(
        not torch.equal(merged[k], plain[k]) for k in merged)




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
    """The whole loop: H3, H4 and the result."""
    from loud.identity import run
    refs.trainer_module('pr26')
    patch_attention()
    patch_sequences()
    spec = dict(experiment='cpu', variant='identity', seed=0, task='identity', args=dict(
        preset='tiny', sequences=24, heldout=4, batch=4, heldout_every=2, checkpoints=True, official_eval=False))
    out = Path(scratch) / 'identity'
    result = run(spec, out)
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and len(metrics) == 5 and 'heldout_loss' in result
    assert {f'ckpt_{f:02d}.pt' for f in (0, 10, 20, 50, 100)} <= {p.name for p in out.iterdir()}


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
    # reverse_sum (verify.cut_prefix_reversed's calibration): the same run up to the rounding of the 8-way sum
    o = cut_options(dict(preset='2m', batch=4, sequences=CUT_ROWS, heldout=32, reverse_sum=True))
    reversed_sum = CutChain(o, torch.device('cpu'), 0, len(training) // 4, heldout)
    for step in (1, 2):
        loss = reversed_sum.update(step, training[(step - 1) * 4:step * 4])[0].item()
        assert abs(loss - losses[step - 1]) < 1e-5, f'update {step}: reversed sum {loss}, rank order {losses[step - 1]}'


def cut_windows(scratch):
    """rows = full is PR #40's Cut estimate. At rows = width / K, with one signed basis for every site: the K 'time'
    windows average to the full cycle's chain credits, with the identity prior too (whatever its guess, and exactly
    the credit, window by window, when the guess is exact); 'token' windows give each token its window's credit; a
    row budget of `rows` for every token is the token windows, a budget of half reads each token's first half; the
    GPUs of a sequence split each token's window into halves, and through the trainer, two virtual GPUs per sequence
    meeting in group_sum give what one GPU measures."""
    m = cut_module()
    from loud.cut import estimator_classes, record_settings
    s = record_settings(m, '2m', 4)
    width, L = s.model.n_embd, s.model.n_layer
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
    fixed = ours.basis()
    ours.basis = lambda: fixed  # one signed basis, so windows compare exactly with the full cycle
    credits = lambda: ours.chain_credits(*saved['args'])
    full = credits()
    K, rows, n = 4, width // 4, x.shape[1]
    tokens = torch.arange(n)
    ours.rows = rows

    def windows():
        out = []
        for k in range(K):
            ours.start = k * rows
            out.append(credits())
        ours.start = None
        return out
    time = windows()
    tiled = credits()
    for key, value in full.items():
        mean = sum(w[key] for w in time) / K
        assert torch.allclose(mean, value, rtol=1e-4, atol=1e-6), f'{key}: the windows do not average to the cycle'
        if not isinstance(key, tuple):  # ('lam', site): the mixing scalars' probes do not use the rows
            each = torch.stack([w[key] for w in time])[tokens % K, tokens]
            assert torch.allclose(tiled[key], each, rtol=1e-4, atol=1e-6), f'{key}: a token misses its window'
    ours.prior, ours.top = True, torch.randn(n, width, generator=torch.Generator().manual_seed(3))  # any guess
    primed = windows()
    for i in range(L):
        mean = sum(w[i] for w in primed) / K
        assert torch.allclose(mean, full[i], rtol=1e-4, atol=1e-5), f'mid {i}: the prior biases the windows'
    assert torch.allclose(primed[0]['emb'], time[0]['emb']), 'S0 keeps the plain estimate'
    ours.guesses = lambda: {i: full[i] for i in range(L)}  # the exact credit as the guess: no residual left
    for w in windows():
        assert all(torch.allclose(w[i], full[i], rtol=1e-4, atol=1e-6) for i in range(L)), 'an exact guess moves'
    del ours.guesses
    ours.prior = False

    class Budget:
        def __init__(self, count):
            self.count = count

        def rows(self, capture):
            return torch.full((capture.x.numel(),), self.count)
    ours.budget = Budget(rows)
    same = credits()
    ours.budget = Budget(rows // 2)
    halved = credits()
    ours.budget, ours.rows = None, rows // 2
    plain_half = credits()
    ours.rows = rows
    for key in tiled:
        if not isinstance(key, tuple):
            assert torch.allclose(same[key], tiled[key]), f'{key}: a full budget is not the token windows'
            assert torch.allclose(halved[key], plain_half[key], rtol=1e-4, atol=1e-6), f'{key}: a half budget'

    halves = [Cut(model, s, 0, part, 2, 2, None) for part in range(2)]
    for start in (0, None):
        for half in halves:
            half.rows, half.start = rows, start
        index = torch.cat([h.row_index(8) for h in halves], 1)
        first = torch.zeros(8, dtype=torch.long) if start == 0 else torch.arange(8) * rows
        assert torch.equal(index, (first[:, None] + torch.arange(rows)) % width), 'the halves do not tile a window'

    # the same through the trainer with two virtual GPUs per sequence, meeting in group_sum: the K windows of one
    # basis average to the record's chain credits, and rows = 2 (one row per GPU) sum to what one GPU measures
    from loud.cut import CutChain, merge_loss
    from loud.options import cut_options

    def chain_credits(gpus, settings):
        """Part 0's chain credits (complete after group_sum) of one measure per setting, every one from the same
        generator states, so every setting reads the same bases."""
        o = cut_options(dict(preset='2m', batch=1, virtual_gpus=gpus, sequences=CUT_ROWS, heldout=32))
        trainer = CutChain(o, torch.device('cpu'), 0, 40, cut_rows_data()[-32:])
        trainer.reset('cut')
        group, out = trainer.groups[0], []
        states = [[g.get_state() for g in (e.generator, e.capture.generator, e.capture.shared_generator)]
                  for e in group]
        real = type(group[0]).chain_credits

        def keep(self, *args):
            credits = real(self, *args)
            if self.part == 0:
                out.append({k: v.clone() for k, v in credits.items()})
            return credits
        type(group[0]).chain_credits = keep
        for rows_, start in settings:
            for e, state in zip(group, states):
                for g, saved_state in zip((e.generator, e.capture.generator, e.capture.shared_generator), state):
                    g.set_state(saved_state)
                e.rows, e.start = rows_, start
            trainer.measure(group, x, y)
        type(group[0]).chain_credits = real
        return out
    split = chain_credits(2, [(None, None)] + [(rows, k * rows) for k in range(K)] + [(2, 0)])
    single = chain_credits(1, [(2, 0)])[0]
    for key, value in split[0].items():
        mean = sum(w[key] for w in split[1:1 + K]) / K
        assert torch.allclose(mean, value, rtol=1e-4, atol=1e-6), f'{key}: split windows do not average to the cycle'
        assert torch.allclose(split[-1][key], single[key], rtol=1e-4, atol=1e-6), f'{key}: rows = split differs'

    # merge_batch: the end merge's candidates scored in one batch are the rank-by-rank scores up to rounding
    rows8 = cut_rows_data()[-32:]
    assert abs(merge_loss(8, True)(model, rows8) - merge_loss(8, False)(model, rows8)) < 1e-5


def cut_run(scratch):
    """The whole loop on the 2-layer model (width 128), two virtual GPUs on one sequence per update: Boost, the switch,
    64 'time' rows rotating across updates with the identity prior, H4, the means (updates 10 and 12) and the
    record's end merge, timed; and chain_chunk runs exactly the planned shapes (the ones the warmup compiles)."""
    m = cut_module()
    import loud.cut
    from loud.options import cut_options
    patch_sequences(length=CUT_SEQ, vocab=CUT_VOCAB)
    calls, kernel = set(), m.chain_chunk

    def recorded(site, move, *args, **kwargs):  # what Dynamo specializes chain_chunk on
        calls.add((site, move.shape[1]))
        return kernel(site, move, *args, **kwargs)
    m.chain_chunk = recorded
    args = dict(preset='2m', sequences=20, heldout=8, batch=1, virtual_gpus=2, rows=64, row_assign='time',
                prior='identity', checkpoints=True, heldout_every=4, official_eval=False)
    out = Path(scratch) / 'cut'
    result = loud.cut.run(dict(experiment='cpu', variant='cut', seed=0, task='cut', args=args), out)
    plan = loud.cut.CutChain(cut_options(args), torch.device('cpu'), 0, 12, torch.zeros(8, CUT_SEQ, dtype=torch.long))
    assert calls == plan.graphs, f'chain_chunk ran {sorted(calls - plan.graphs)} unplanned, missed ' \
                                 f'{sorted(plan.graphs - calls)}'
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and len(metrics) == 12
    assert [r['phase'] for r in metrics] == ['boost'] * 2 + ['cut'] * 10
    assert [r.get('chain_window') for r in metrics] == [None] * 2 + [[s % 2 * 64, s % 2 * 64 + 64]
                                                                      for s in range(3, 13)]
    assert {'holdout_plain', 'holdout_merged', 'merge_choices', 'merge_seconds'} <= set(result)
    assert result['holdout_merged'] <= result['holdout_plain'] and result['means_updates'] == 2
    assert result['train_seconds'] >= result['merge_seconds']
    assert {'final.pt', 'final_plain.pt', 'ckpt_100.pt'} <= {p.name for p in out.iterdir()}
    calls_ = [json.loads(line) for line in (out / 'heldout.jsonl').read_text().splitlines()]
    assert [c['call'] for c in calls_] == list(range(1, len(calls_) + 1))
    assert sum(c['reason'] == 'merge' for c in calls_) > 80  # the plain weights and every candidate the merge scored


def cut_key_blocks(scratch):
    """E17's key_blocks: every block listed is the record itself; a subset leaves the other blocks' c_k gradients as
    the record's (exact key errors, which its probes also measure) and changes the rest."""
    cut_module()
    import loud.cut
    from loud.options import cut_options
    patch_sequences(length=CUT_SEQ, vocab=CUT_VOCAB)
    batch = cut_rows_data()[:1]

    def grads(**kw):
        args = dict(preset='2m', sequences=20, heldout=8, batch=1, virtual_gpus=2, official_eval=False, **kw)
        chain = loud.cut.CutChain(cut_options(args), torch.device('cpu'), 0, 12, torch.zeros(8, CUT_SEQ,
                                                                                             dtype=torch.long))
        step = next(i for i, b in enumerate(chain.schedule, 1) if b[0] == 'cut')
        chain.reset('cut')
        chain.update(step, batch)
        return {n: p.grad.clone() for n, p in chain.named}

    record, every, one = grads(), grads(key_blocks=[0, 1]), grads(key_blocks=[1])
    assert all(torch.equal(record[n], every[n]) for n in record)
    keys = [n for n in record if n.endswith('attn.c_k.weight')]
    assert all(torch.allclose(record[n], one[n], atol=1e-6) for n in keys)
    assert not all(torch.allclose(record[n], one[n], atol=1e-6) for n in record)
    none = grads(key_probes=False)  # no probes: c_k as the record's, the K-branch input credit gone
    assert all(torch.allclose(record[n], none[n], atol=1e-6) for n in keys)
    assert not all(torch.allclose(record[n], none[n], atol=1e-6) for n in record)
    try:
        cut_options(dict(preset='2m', key_probes=False, key_blocks=[1]))
        raise AssertionError('accepted key_blocks without key probes')
    except ValueError:
        pass
    for bad in ([2], [0, 0], 'all'):
        try:
            cut_options(dict(preset='2m', key_blocks=bad))
        except ValueError:
            continue
        raise AssertionError(f'accepted key_blocks={bad!r}')


def cut_deep_readout(scratch):
    """E5's deep readout on the 2-layer model (two virtual GPUs on one sequence): S0's and S1's credits are the
    record's own (the same cycles), the mixing scalars' too (the same moves), S2's solved credit is the record's
    measured one up to the model's nonlinearity, and the update's gradient follows; the plan has S0's cycle and the
    later sites' scalar moves only."""
    cut_module()
    import loud.cut
    from loud.options import cut_options
    patch_sequences(length=CUT_SEQ, vocab=CUT_VOCAB)
    batch = cut_rows_data()[:1]
    seen = {}

    def run(tag, **kw):
        args = dict(preset='2m', sequences=20, heldout=8, batch=1, virtual_gpus=2, official_eval=False, **kw)
        chain = loud.cut.CutChain(cut_options(args), torch.device('cpu'), 0, 12, torch.zeros(8, CUT_SEQ,
                                                                                             dtype=torch.long))
        step = next(i for i, b in enumerate(chain.schedule, 1) if b[0] == 'cut')
        chain.reset('cut')
        with torch.no_grad():  # a few updates in: the init's zero writers would make every credit trivial
            for p in chain.model.parameters():
                p.add_(0.02 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())))
        original = chain.Cut.chain_credits

        def kept(self, *args):
            credits = original(self, *args)
            seen.setdefault(tag, credits)  # part 0's (group-summed: the sequence's)
            return credits
        chain.Cut.chain_credits = kept
        try:
            chain.update(step, batch)
        finally:
            chain.Cut.chain_credits = original
        return chain, {n: p.grad.clone() for n, p in chain.named}

    _, record = run('record')
    chain, deep = run('deep', deep_readout=True)
    a, b = seen['record'], seen['deep']
    assert torch.allclose(a['emb'], b['emb'], atol=1e-6) and torch.allclose(a[0], b[0], atol=1e-6)
    for k in range(2):
        assert torch.allclose(a['lam', k], b['lam', k], rtol=1e-4, atol=1e-6), (k, a['lam', k], b['lam', k])
    per_token = torch.nn.functional.cosine_similarity(a[1], b[1], dim=-1)
    assert per_token.median() > 0.99 and per_token.quantile(0.1) > 0.98, per_token.quantile(torch.tensor([.1, .5]))
    cos = lambda x, y: float(x.flatten() @ y.flatten() / (x.norm() * y.norm()))
    assert all(cos(record[n], deep[n]) > 0.99 for n in record), {n: cos(record[n], deep[n]) for n in record}
    assert chain.graphs == {(-1, 64), (1, 4)}, chain.graphs
    for bad in (dict(rows=64), dict(rows=64, row_budget='ngram', budget_theta=1.0)):
        try:
            cut_options(dict(preset='2m', deep_readout=True, **bad))
        except ValueError:
            continue
        raise AssertionError(f'accepted deep_readout with {bad}')


def cut_budget(scratch):
    """E13's hook through the trainer: a ROW_BUDGETS strategy is built fresh by every reset (the run's never sees the
    warmup's random tokens), gives each Cut estimate's tokens their counts (once per GPU of the sequence) and sees
    every update's batch after the update; Boost never asks it. E13's own rule ('ngram'): the low tier is the tokens
    both the n-gram table and the network find easy, and the table learns the batches it observes."""
    cut_module()
    import loud.cut
    from loud.options import cut_options
    built = []

    class Recorder:
        def __init__(self, o, vocab):
            self.count, self.asked, self.observed = o['rows'] // 2, 0, []
            built.append(self)

        def rows(self, capture):
            self.asked += 1
            return torch.full((capture.x.numel(),), self.count)

        def observe(self, batch):
            self.observed.append(batch.clone())
    loud.cut.ROW_BUDGETS['recorder'] = Recorder
    o = cut_options(dict(preset='2m', batch=1, virtual_gpus=2, sequences=CUT_ROWS, heldout=8, rows=64,
                         row_budget='recorder', end_merge=False, means=False))
    data = cut_rows_data()
    chain = loud.cut.CutChain(o, torch.device('cpu'), 0, 10, data[-8:])
    first_cut = next(step for step, (phase, *_) in enumerate(chain.schedule, 1) if phase == 'cut')
    chain.warmup(data[-1:])
    warm = built[-1]
    assert warm.asked == 2 and len(warm.observed) == 1, 'the Cut warmup update did not use and feed its budget'
    chain.reset()
    fresh = built[-1]
    assert fresh is not warm and fresh.asked == 0 and not fresh.observed, 'the run inherits the warmup budget'
    for step in range(1, first_cut + 2):
        chain.update(step, data[step - 1:step])
    assert fresh.asked == 4 and len(fresh.observed) == first_cut + 1, (fresh.asked, len(fresh.observed))
    from types import SimpleNamespace
    budget = loud.cut.NGramBudget(cut_options(dict(preset='2m', rows=128, row_budget='ngram', budget_theta=3.0)), 64)
    row = torch.arange(65) % 8  # a repeating pattern: unknown to the empty table, easy once it has seen it
    capture = SimpleNamespace(x=row[:-1], y=row[1:], clean_loss=torch.zeros(64))
    assert budget.rows(capture).tolist() == [128] * 64  # the n-gram model cannot know an unseen stream
    budget.observe(row[None])
    assert budget.rows(capture).tolist() == [64] * 64  # seen: every token is easy for both, so the low tier
    capture.clean_loss = torch.full((64,), 9.0)  # the network finds every token hard: the full cycle everywhere
    assert budget.rows(capture).tolist() == [128] * 64
    other = (torch.arange(65) * 3) % 8  # a capture of two sequences: each scored with its own last target
    budget.observe(other[None])
    two = SimpleNamespace(x=torch.stack([row[:-1], other[:-1]]), y=torch.stack([row[1:], other[1:]]),
                          clean_loss=torch.zeros(128))
    assert budget.rows(two).tolist() == [64] * 128, budget.rows(two)
    assert all(torch.equal(seen, data[s:s + 1]) for s, seen in enumerate(fresh.observed))  # each batch, in order
    try:
        loud.cut.CutChain(dict(o, row_budget='missing'), torch.device('cpu'), 0, 10, data[-8:])
    except ValueError as exc:
        assert 'unknown row_budget' in str(exc)
    else:
        raise AssertionError('an unknown row_budget was accepted')


def tokens(n, seed=1, vocab=4096):
    return torch.randint(0, vocab, (n, SEQ), generator=torch.Generator().manual_seed(seed))


def variants_blur(scratch):
    """E2 in float32 (no autocast): r has the documented length at all nine sites, + and - add the same r, and the
    antithetic mean cancels the odd orders: far closer to the clean gradient than one displaced gradient, and closer
    by about 4x when r halves. Then the trainer: the clean loss, and the mean of all 2 x micro-batches gradients."""
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    from loud.variants import Blur
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    cpu = torch.device('cpu')
    model = reference.make_model(cpu, 0)
    width, params = model.config.n_embd, list(model.parameters())
    rows = tokens(2)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    def gradient():
        model.requires_grad_(True)
        grads = torch.autograd.grad(model(x, y), params)
        model.requires_grad_(False)
        return torch.cat([g.reshape(-1) for g in grads])

    sites, inputs = [], {}  # the clean activations at the sites, read with the test's own hooks
    hooks = [model.transformer.wte.register_forward_hook(lambda m, a, out: sites.append(out))]
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_pre_hook(lambda m, a, i=i: inputs.__setitem__(i, a[0])))
        hooks.append(block.attn.register_forward_hook(lambda m, a, out, i=i: sites.append(inputs[i] + out)))
    with torch.no_grad():
        clean = model(x, y)
    for h in hooks:
        h.remove()
    clean_grad = gradient()

    s = 0.1
    blur = Blur(model, s, 0, cpu)
    assert torch.equal(blur.measure(lambda: model(x, y)), clean)
    assert len(blur.shifts) == len(sites) == 1 + model.config.n_layer
    for a, r in zip(sites, blur.shifts):  # per-coordinate RMS of r = s x the token's RMS
        assert torch.allclose(r.square().mean(-1).sqrt(), s * a.square().mean(-1).sqrt(), rtol=1e-5)
        assert torch.allclose(r.norm(dim=-1), s * a.square().mean(-1).sqrt() * width ** 0.5, rtol=1e-5)
    for sign in (1, -1):  # each site's writer output moves by exactly sign * r
        raw, moved = [], []
        hooks = [model.transformer.wte.register_forward_hook(lambda m, a, out: raw.append(out), prepend=True)]
        hooks += [b.attn.register_forward_hook(lambda m, a, out: raw.append(out), prepend=True)
                  for b in model.transformer.h]
        with blur.shifted(sign), torch.no_grad():
            hooks += [model.transformer.wte.register_forward_hook(lambda m, a, out: moved.append(out))]
            hooks += [b.attn.register_forward_hook(lambda m, a, out: moved.append(out)) for b in model.transformer.h]
            model(x, y)
        for h in hooks:
            h.remove()
        for before, after, r in zip(raw, moved, blur.shifts):
            assert torch.allclose(after - before, sign * r, atol=1e-6), 'the two signs do not add the same r'

    def displaced(sign):
        with blur.shifted(sign):
            return gradient()

    plus, minus = displaced(1), displaced(-1)
    one_sided, antithetic = (plus - clean_grad).norm(), ((plus + minus) / 2 - clean_grad).norm()
    assert antithetic < 0.2 * one_sided, (antithetic, one_sided)
    blur.shifts = [r / 2 for r in blur.shifts]
    half = ((displaced(1) + displaced(-1)) / 2 - clean_grad).norm()
    assert half < 0.4 * antithetic, (half, antithetic)  # second order: a quarter, up to higher orders

    o = proxy_options(dict(preset='2m', batch=2, blur=0.03))
    trainer = Backprop(o, reference, cpu, 0, 10)
    loss = trainer.gradient(rows)
    got = [p.grad.clone() for p in trainer.model.parameters()]
    replay = Blur(trainer.model, 0.03, 0, cpu)  # the same generator stream, replayed by hand
    sums, clean_losses, blurred = [torch.zeros_like(p, dtype=torch.float32) for p in got], [], []
    for i in range(2):
        micro = rows[i:i + 1]
        clean_losses.append(replay.measure(lambda: trainer.loss_of(micro)))
        for sign in (1, -1):
            trainer.model.requires_grad_(True)
            with replay.shifted(sign):
                step_loss = trainer.loss_of(micro)
                for total, g in zip(sums, torch.autograd.grad(step_loss, list(trainer.model.parameters()))):
                    total += g.float()
            trainer.model.requires_grad_(False)
            blurred.append(step_loss.detach())
    assert torch.equal(loss, torch.stack(clean_losses).mean()), 'the logged loss is not the clean loss'
    assert torch.allclose(trainer.blurred, torch.stack(blurred).mean())
    for g, total, p in zip(got, sums, trainer.model.parameters()):
        assert torch.equal(g, (total / 4).to(p.dtype)), 'not the mean over sequences and both signs'
    assert not trainer.model.transformer.wte._forward_hooks, 'the blur left a hook behind'


def optimizer_state(trainer):
    """Copies of the weights, every optimizer state tensor (in parameter order) and the optimizers' clocks."""
    held = [p.detach().clone() for p in trainer.model.parameters()]
    for o in trainer.optimizer.optimizers:
        for p in trainer.model.parameters():
            held += [v.clone() for _, v in sorted(o.state.get(p, {}).items()) if torch.is_tensor(v)]
    return held, [getattr(o, 't', None) for o in trainer.optimizer.optimizers]


def same_tensors(a, b):
    return len(a[0]) == len(b[0]) and all(torch.equal(u, v) for u, v in zip(a[0], b[0]))


def variants_control(scratch, recipe='boostcut'):
    """E4: each update is judged on 4 held-out rows, rotating, by their loss under the old and the new weights; a
    rejected update leaves the weights and every optimizer tensor exactly as before (after the first update: as a
    fresh optimizer), while ANVIL II's schedule clock advances with the loop, so the next update equals a fresh
    trainer's at that clock; adapt's multiplier follows the outcomes; warmdown 0 holds the rates constant."""
    from loud.heldout import mean_loss
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    reference = Reference(recipe, 1.0, '2m')
    patch_attention()
    cpu, rows, held = torch.device('cpu'), tokens(2), tokens(8, seed=4)
    make = lambda **args: Backprop(proxy_options(dict(recipe=recipe, preset='2m', batch=2, heldout=8, **args)),
                                   reference, cpu, 0, 10, held)
    reject = iter([0.0, 1.0] * 100)  # the held-out loss before and after: rose
    trainer = make(step_control='reject')
    assert [trainer.control.rows(s) for s in (1, 2, 3)] == [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 3]]
    before = optimizer_state(trainer)
    trainer.control.loss = lambda step: next(reject)
    _, fields = trainer.update(1, rows)
    assert fields['accepted'] is False and fields['judge_rows'] == [0, 1, 2, 3]
    after = optimizer_state(trainer)
    weights = len(list(trainer.model.parameters()))
    assert all(torch.equal(u, v) for u, v in zip(before[0], after[0][:weights]))
    assert all(not t.any() for t in after[0][weights:]), 'not a fresh optimizer'
    assert after[1][0] == before[1][0] + 1, "ANVIL II's clock must follow the loop's updates"
    del trainer.control.loss  # the real paired judgment again
    old = mean_loss(trainer.model, held[4:8], cpu, 4)
    _, fields = trainer.update(2, rows)
    assert fields['heldout_before'] == old and fields['accepted'] == (fields['heldout_after'] <= old), fields
    assert fields['heldout_after'] == mean_loss(trainer.model, held[4:8], cpu, 4) or not fields['accepted']
    fresh = make()
    fresh.optimizer.optimizers[0].t = 1
    fresh.update(2, rows)
    if fields['accepted']:
        assert all(torch.equal(a, b) for a, b in zip(trainer.model.parameters(), fresh.model.parameters()))

    trainer.update(3, tokens(2, seed=2))
    held_state = optimizer_state(trainer)
    trainer.control.loss = lambda step: next(reject)
    trainer.update(4, tokens(2, seed=3))
    later = optimizer_state(trainer)
    assert same_tensors(held_state, later), 'a mid-run rejection is not undone exactly'
    assert later[1][0] == held_state[1][0] + 1
    if recipe != 'boostcut':
        return

    trainer = make(step_control='adapt', accept_up=1.25, accept_down=0.5, warmdown=0)
    outcomes = iter([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] * 3)  # accept, accept, reject, accept, ...
    trainer.control.loss = lambda step: next(outcomes)
    multipliers = []
    for step in range(1, 11):
        _, fields = trainer.update(step, rows)
        multipliers.append(fields['lr_multiplier'])
        assert fields['lr']['anvil'] == reference.settings.anvil_lr * fields['lr_multiplier']  # constant schedule
    assert multipliers[:5] == [1.0, 1.25, 1.5625, 0.78125, 0.9765625]


def variants_control_pr40(scratch):
    """The same exact undo with PR #40's batched ANVIL II, whose per-matrix state is views into stacked tensors."""
    variants_control(scratch, 'pr40')


def variants_run(scratch):
    """The whole loop with each variant: the experiment series land in metrics.jsonl; the default trainer builds none
    of them."""
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference, run
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    patch_sequences()
    default = Backprop(proxy_options(dict(preset='2m', batch=2)), reference, torch.device('cpu'), 0, 10)
    assert default.blur is None and default.control is None and default.shaper is None
    for variant, args, fields in (
            ('adapt', dict(step_control='adapt', warmdown=0), {'accepted', 'lr_multiplier', 'heldout_before'}),
            ('blur', dict(blur=0.1), {'blurred_loss'}),
            ('snr', dict(per_sequence_grads=True, snr_scaling='snr'), {'snr_ratios'}),
            ('fisher', dict(fisher=True), set())):
        spec = dict(experiment='cpu', variant=variant, seed=0, task='proxy', args=dict(
            preset='2m', sequences=24, heldout=4, batch=4, official_eval=False, **args))
        out = Path(scratch) / variant
        result = run(spec, out)
        metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        assert result['complete'] and len(metrics) == 5 and all(fields <= set(m) for m in metrics), variant
        assert all(torch.isfinite(torch.tensor(m['train_loss'])) for m in metrics), variant


def proxy_trainer(**args):
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    o = proxy_options(dict(dict(preset='2m', batch=4), **args))
    return Backprop(o, reference, torch.device('cpu'), 0, 10), reference.module


def batches(count, vocab, seed=6):
    return torch.randint(0, vocab, (count, 4, SEQ), generator=torch.Generator().manual_seed(seed))


def anvil_hook(scratch):
    """ShapedAnvil whose shaper returns polar(pending) steps exactly as boost-cut's Anvil, and leaves Anvil in place."""
    from loud.anvil import shaped_optimizer
    trainer, m = proxy_trainer()
    original = m.Anvil
    twin = copy.deepcopy(trainer.model)
    polar = type('Polar', (), dict(lane_gain=True, direction=lambda self, p, pending, before: m.polar(pending)))()
    shaped = shaped_optimizer(m, twin, trainer.reference.settings, 10, polar)
    assert m.Anvil is original and type(trainer.optimizer.optimizers[0]) is original
    for rows in batches(3, trainer.model.config.vocab_size):
        for model, optimizer in ((trainer.model, trainer.optimizer), (twin, shaped)):
            model.requires_grad_(True)
            model(rows[:, :-1].contiguous(), rows[:, 1:].contiguous()).backward()
            model.requires_grad_(False)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    for (name, a), b in zip(trainer.model.named_parameters(), twin.parameters()):
        assert torch.equal(a, b), f'{name}: ShapedAnvil does not reproduce Anvil'


def batched_anvil_hook(scratch):
    """PR #40's BatchedAnvil and ShapedBatchedAnvil (E8's hook in H8) with a shaper that returns polar_batch's update
    and keeps the lane gain: bit-identical weights and state over 3 steps through ANVIL II's engagement."""
    m = cut_module()
    from loud.anvil import shaped_optimizer
    from loud.cut import record_settings
    s = record_settings(m, '2m', 4)

    class Unchanged:
        lane_gain = True

        @staticmethod
        def direction(p, pending, before):
            return m.polar_batch(pending[None], s.polar_margin)[0]

    model = m.make_model(torch.device('cpu'), 0, s.model, s.projection_std, s.head_std)
    twin = copy.deepcopy(model)
    plain, shaped = m.make_optimizer(model, s, 6), shaped_optimizer(m, twin, s, 6, Unchanged(), 'BatchedAnvil')
    assert type(shaped.optimizers[0]).__name__ == 'ShapedBatchedAnvil' and m.BatchedAnvil.__name__ == 'BatchedAnvil'
    generator = torch.Generator().manual_seed(8)
    for step in range(1, 4):
        for p, q in zip(model.parameters(), twin.parameters()):
            p.grad = torch.randn(p.shape, generator=generator).to(p.dtype)
            q.grad = p.grad.clone()
        for optimizer in (plain, shaped):
            m.set_rate(optimizer, step, 6, s.warmdown, s.final)
            optimizer.step()
        for (name, p), q in zip(model.named_parameters(), twin.parameters()):
            assert torch.equal(p, q), f'{name} differs after step {step}'
    for p, q in zip(plain.optimizers[0].param_groups[0]['params'], shaped.optimizers[0].param_groups[0]['params']):
        assert all(torch.equal(plain.optimizers[0].state[p][k], shaped.optimizers[0].state[q][k])
                   for k in ('fast', 'slow', 'energy'))

def snr_scaling(scratch):
    """E8's intervention: along the momentum's directions from before the batch, r from the batch's per-sequence
    projections (computed here one direction at a time), the control arm's polar factor from the SVD, the rate arm's
    uniform Frobenius ratio, the update's own directions while the rails are zero, and ANVIL II's step without its
    per-lane gain or renormalization, its directions from the rails as they stood before the step."""
    trainer, m = proxy_trainer(per_sequence_grads=True, snr_scaling='snr')
    rows = batches(1, trainer.model.config.vocab_size)[0]
    trainer.gradient(rows)
    shaper = trainer.shaper
    shaper.before_step(1, trainer.pergrad)
    name = 'transformer.h.1.mlp.c_fc.weight'
    p = dict(trainer.model.named_parameters())[name]
    generator = torch.Generator().manual_seed(12)
    pending, before = torch.randn(p.shape, generator=generator), torch.randn(p.shape, generator=generator)
    u, _, vh = torch.linalg.svd(before, full_matrices=False)
    grads = trainer.pergrad.stack(name).float()
    r = []
    for i in range(len(vh)):
        projections = torch.stack([u[:, i] @ g @ vh[i] for g in grads])  # p_s = u_i^T G_s v_i, sequence by sequence
        m2, s2 = projections.mean().square(), projections.var()
        r.append(max(0.0, (m2 - s2 / len(grads)).item()) / m2.item())
    r = torch.tensor(r)
    assert (r < 1).all() and (r > 0).any() and (r == 0).any(), r
    assert torch.allclose(shaper.direction(p, pending, before), (u * r) @ vh, atol=1e-5)
    shaper.arm = 'rate'
    assert torch.allclose(shaper.direction(p, pending, before), r.square().mean().sqrt() * (u @ vh), atol=1e-5)
    shaper.arm = 'control'
    control = shaper.direction(p, pending, before)
    assert torch.allclose(torch.linalg.svdvals(control), torch.ones(min(p.shape)), atol=1e-4)
    assert (control - m.polar(before)).norm() < 0.1 * control.norm()  # ANVIL's Newton-Schulz polar, roughly
    first = shaper.direction(p, pending, torch.zeros_like(pending))  # zero rails: the update's own directions
    assert (first - m.polar(pending)).norm() < 0.1 * first.norm()

    # ANVIL II's step with lane_gain off: the shaped direction, the decay, the rate and nothing else
    anvil = trainer.optimizer.optimizers[0]
    seen = {}
    direction = shaper.direction
    shaper.direction = lambda q, pend, bef: seen.setdefault(id(q), (direction(q, pend, bef), bef.clone()))[0]
    w = p.detach().clone()
    trainer.optimizer.step()
    lr, wd = anvil.param_groups[0]['lr'], anvil.param_groups[0]['weight_decay']
    slow = anvil.state[p]['slow']
    update, before = seen[id(p)]
    assert not before.any()  # the first step's rails are zero
    expected = w - lr * max(1, w.size(0) / w.size(1)) ** 0.5 * (update + (slow * w >= 0) * w * (wd * lr))
    assert torch.allclose(p, expected, atol=1e-6), 'the lane gain or its renormalization was applied'
    weight = anvil.fast_weight if anvil.t >= anvil.engage else 1.0  # the blend the next step reads
    expected = weight * anvil.state[p]['fast'] + (1 - weight) * anvil.state[p]['slow']
    seen.clear()
    trainer.optimizer.step()
    assert torch.equal(seen[id(p)][1], expected), 'the directions are not the rails from before the step'


def snr_diagnostic(scratch):
    """snr_spectrum, along the checkpoint momentum's singular directions: when the per-sequence gradients share the
    momentum's strong rank-1 direction, it has a high SNR and holds nearly all the energy (little below SNR 1); with
    pure noise along independent random rails, most energy is below SNR 1."""
    from loud.anvil import is_anvil_matrix
    from loud.snr import snr_spectrum
    trainer, _ = proxy_trainer()
    model = trainer.model
    names = [n for n, p in model.named_parameters() if is_anvil_matrix(n, p)]
    shapes = {n: p.shape for n, p in model.named_parameters()}
    generator = torch.Generator().manual_seed(13)
    signal = {n: torch.outer(torch.randn(shapes[n][0], generator=generator),
                             torch.randn(shapes[n][1], generator=generator)) for n in names}
    rails = lambda make: dict(state={n: dict(fast=make(n), slow=make(n)) for n in names}, fast_weight=0.5)

    def synthetic(strength):
        return lambda model, rows: {n: strength * signal[n] + 0.01 * torch.randn(
            (len(rows),) + tuple(shapes[n]), generator=generator) for n in names}

    rows = torch.zeros(16, SEQ, dtype=torch.long)
    strong, _ = snr_spectrum(model, rows, rails(lambda n: signal[n] + 0.01 * torch.randn(
        shapes[n], generator=generator)), batch=8, gradients=synthetic(1.0))
    noise_rails = rails(lambda n: torch.randn(shapes[n], generator=generator))
    weak, tensors = snr_spectrum(model, rows, noise_rails, batch=8, gradients=synthetic(0.0))
    for n in names:
        assert strong[n]['energy_below_1'] < 0.05 and strong[n]['snr_top16'][0] > 100, (n, strong[n])
        assert weak[n]['energy_below_1'] > 0.5, (n, weak[n])
        assert tensors[n]['snr'].shape == tensors[n]['singular'].shape
    assert set(strong['energy_below_1']) == {'attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'mlp.c_fc',
                                             'mlp.c_proj'}
    exact, _ = snr_spectrum(model, tokens(4), noise_rails, batch=8)  # exact backprop gradients run end to end
    assert all(0 <= v <= 1 for v in exact['energy_below_1'].values())


def fisher_factor(scratch):
    """E9: the sums of d d^T come from the sampled-label gradient at each block matrix's output (checked against an
    independent forward, token by token), leave the training gradient untouched, and ANVIL II orthogonalizes the
    damped inverse root of F times the pending update."""
    from loud.fisher import DAMPING, SAMPLE_SEED, inverse_root
    trainer, m = proxy_trainer(fisher=True)
    plain, _ = proxy_trainer()
    shaper, model = trainer.shaper, trainer.model
    rows = batches(1, model.config.vocab_size)[0]
    trainer.gradient(rows)
    plain.gradient(rows)
    for (name, p), q in zip(model.named_parameters(), plain.model.parameters()):
        assert torch.equal(p.grad, q.grad), f'{name}: the sampled-label backward leaked into the training gradient'

    reference = copy.deepcopy(plain.model).requires_grad_(True)
    names = [n for n, p in reference.named_parameters() if n.startswith('transformer.h.') and p.ndim == 2
             and 've_gate' not in n]
    modules = dict(reference.named_modules())
    generator = torch.Generator().manual_seed(0 + SAMPLE_SEED)  # the job seed (0) + the label generator's offset
    expected = [0] * len(names)
    for micro in rows.split(1):
        outputs = {}
        hooks = [modules[n.removesuffix('.weight')].register_forward_hook(
            lambda mod, args, out, n=n: outputs.__setitem__(n, out)) for n in names]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = reference(micro[:, :-1].contiguous())
        for h in hooks:
            h.remove()
        sampled = torch.multinomial(logits.detach().softmax(-1).flatten(0, 1), 1, generator=generator)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), sampled.flatten(), reduction='sum')
        deltas = torch.autograd.grad(loss, [outputs[n] for n in names])
        for j, d in enumerate(deltas):
            d = d[0].float()
            expected[j] = expected[j] + sum(torch.outer(d[t], d[t]) for t in range(d.shape[0]))  # token by token
    for j, name in enumerate(names):
        assert torch.allclose(shaper.batch_sum[j], expected[j], rtol=1e-3, atol=1e-6), name
    tokens_seen = 4 * (SEQ - 1)
    assert shaper.tokens == tokens_seen

    shaper.before_step(1)
    F, R = shaper.F[0], shaper.R[0]
    assert torch.allclose(F, expected[0] / tokens_seen, rtol=1e-3, atol=1e-9)
    damped = F + DAMPING * torch.linalg.eigvalsh(F).clamp_min(0).mean() * torch.eye(len(F))
    assert torch.allclose(R @ damped @ R, torch.eye(len(F)), atol=1e-3)
    assert torch.allclose(inverse_root(F), R)
    p = dict(model.named_parameters())[names[0]]
    pending = torch.randn(p.shape, generator=torch.Generator().manual_seed(11))
    assert torch.equal(shaper.direction(p, pending), m.polar(R @ pending))  # preconditioned, then ANVIL's own polar


def population_run(scratch):
    """E3 end to end on a tiny model, once per rule: the trunk to the branch, then every copy makes every update at
    local batch 2 with the single run's rates and ANVIL II's clock; the rule acts as described; the midpoints are
    logged; H4 saves the trunk, then each copy."""
    from loud import population
    m = refs.trainer_module('boostcut')
    patch_attention()
    patch_sequences(population)
    for rule in ('average', 'select', 'ensemble'):
        spec = dict(experiment='cpu', variant=rule, seed=0, task='population', args=dict(
            preset='2m', sequences=72, heldout=4, rule=rule, every=2, midpoints_every=2, official_eval=False,
            checkpoints=rule == 'average'))
        out = Path(scratch) / rule
        result = population.run(spec, out)
        config = json.loads((out / 'config.json').read_text())
        metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        events = [json.loads(line) for line in (out / 'population.jsonl').read_text().splitlines()]
        # 68 update sequences: 8 updates of 8; the branch after round(0.2 * 8) = 2, then 4 copies of 2 sequences
        assert config['branch_step'] == 2 and config['local_batch'] == 2 and result['updates'] == 8
        assert [r['step'] for r in metrics] == list(range(1, 9))
        assert [r['phase'] for r in metrics] == ['trunk'] * 2 + ['copies'] * 6
        assert all(len(r['copy_losses']) == 4 for r in metrics[2:])
        anvil = m.Anvil([torch.zeros(2, 2)], 1.0, 0.0, 8)
        for r in metrics:
            anvil.t = r['step'] - 1
            assert r['anvil_beta'] == anvil.rail_beta(), 'ANVIL II does not follow the single run'
        base = metrics[0]['lr']['anvil']
        assert all(abs(r['lr']['anvil'] - base * min(1.0, (8 - r['step'] + 1) / 8 / 0.3)) < 1e-12 for r in metrics)
        assert [e['step'] for e in events] == [4, 6, 8] and all(len(e['midpoints']) == 6 for e in events)
        assert all(abs(e['barriers'][p] - (e['midpoints'][p] - max(e['copies'][int(p[0])], e['copies'][int(p[2])])))
                   < 1e-12 for e in events for p in e['midpoints'])
        if rule == 'average':
            assert all(e['action'] == 'average' for e in events) and result['final'] == 'mean'
            # scored before the average at the same update, so the copies are still apart
            assert all(len(set(e['copies'])) > 1 for e in events), 'the midpoints were scored after the average'
            saved = {p.name for p in out.glob('ckpt_*.pt')}
            assert saved == {'ckpt_00.pt', 'ckpt_10.pt', 'ckpt_20.pt'} | {
                f'ckpt_{f}.c{c}.pt' for f in (50, 100) for c in range(4)}, saved
        if rule == 'select':
            assert all(set(e['action']) == {'select', 'replace', 'scores'} for e in events)
            assert result['final'] == f'copy {result["final_copy_scores"].index(min(result["final_copy_scores"]))}'
            assert result['heldout_loss'] == min(result['final_copy_scores'])
        if rule == 'ensemble':
            assert all('action' not in e for e in events)
            mean, best = result['final_mean_score'], min(result['final_copy_scores'])
            assert result['final'] == ('mean' if mean < best else f'copy {result["final_copy_scores"].index(best)}')
            assert result['heldout_loss'] == min(mean, best)


def population_rules(scratch):
    """The rules on live copies: each copy trains on its share of the batch; averaging leaves every copy at the mean
    of the weights and of every optimizer state tensor; selection gives the worst copy the best's weights and an
    unaliased copy of its optimizer state."""
    from loud.heldout import HeldoutScorer
    from loud.population import Clock, Population, state_tensors
    trainer, m = proxy_trainer(batch=8)
    vocab = trainer.model.config.vocab_size
    rows = torch.randint(0, vocab, (3, 8, SEQ), generator=torch.Generator().manual_seed(6))
    for step in (1, 2):
        trainer.update(step, rows[step - 1])
    heldout = HeldoutScorer(rows[2][:4], Path(scratch) / 'heldout.jsonl', torch.device('cpu'))
    o = dict(trainer.o, copies=4, rule='average')
    population = Population(o, trainer, trainer.reference, torch.device('cpu'), 0, heldout, Clock(torch.device('cpu')))
    seen = []
    for member in population.copies:
        update = member.update
        member.update = lambda step, batch, update=update: seen.append(batch.clone()) or update(step, batch)
    losses, _ = population.update(3, rows[0])
    assert len(losses) == 4 and all(torch.equal(b, rows[0][2 * k:2 * k + 2]) for k, b in enumerate(seen))
    tensors = [state_tensors(member) for member in population.copies]
    assert any(not torch.equal(a, b) for a, b in zip(tensors[0], tensors[1])), 'the copies did not diverge'
    expected = [torch.stack([t[i].float() for t in tensors]).mean(0) for i in range(len(tensors[0]))]
    population.average()
    for member in population.copies:
        for t, e in zip(state_tensors(member), expected):
            assert torch.allclose(t.float(), e, atol=1e-6), 'averaging left a weight or a state tensor apart'

    population.update(4, rows[1])
    action = population.select(4)
    best, worst = action['select'], action['replace']
    for p, q in zip(population.copies[worst].model.parameters(), population.copies[best].model.parameters()):
        assert torch.equal(p, q)
    best_anvil = population.copies[best].optimizer.optimizers[0]
    kept = {id(p): best_anvil.state[p]['slow'].clone() for p in best_anvil.param_groups[0]['params']}
    population.copies[worst].update(5, rows[0][:2])  # its state moves; the best copy's must not
    assert all(torch.equal(best_anvil.state[p]['slow'], kept[id(p)]) for p in best_anvil.param_groups[0]['params'])








def cut_compile_plan(scratch):
    """Every Cut config's plan: its chain's shapes, which the warmup compiles, fit the record's own Dynamo limit."""
    cut_module()
    from loud.cut import CutChain
    from loud.options import RANGES, cut_options
    from loud.run import specs_from_config
    configs = [p for p in sorted((Path(__file__).parents[1] / 'configs').rglob('*.yaml'))
               if 'task: cut' in p.read_text()]
    assert len(configs) >= 3, configs
    for path in configs:
        o = cut_options(specs_from_config(path)[0][0]['args'])
        n = RANGES.get(o['sequences'], o['sequences'])
        plan = CutChain(o, torch.device('cpu'), 0, (n - o['heldout']) // o['batch'],
                        torch.zeros(o['heldout'], CUT_SEQ, dtype=torch.long))
        assert 0 < len(plan.graphs) <= 17 and plan.recompile_limit == 64, (path, len(plan.graphs))


if __name__ == '__main__':
    torch.set_num_threads(2 if sys.argv[1] == 'cut_rank' else 8)  # every core would oversubscribe a shared machine
    torch.manual_seed(0)
    globals()[sys.argv[1]](*sys.argv[2:])
    print('ok')

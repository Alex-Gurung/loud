"""CPU checks of the training methods, one per process (each loads its own reference trainer; see refs.trainer_module).

    python tests/method_checks.py <check> <scratch dir>

Two test-only patches make them run on a CPU: dense attention with the same causal window instead of flex attention
(which has no CPU backward, and on CPU sees float32 values under autocast), and sequences cut to SEQ tokens.
"""
import copy
import json
import sys
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


def patch_sequences():
    real = data.benchmark_split

    def short(*args, **kwargs):
        split = real(*args, **kwargs)
        split.train, split.heldout = split.train[:, :SEQ].contiguous(), split.heldout[:, :SEQ].contiguous()
        return split
    train.benchmark_split, train.SEQ = short, SEQ


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


if __name__ == '__main__':
    torch.set_num_threads(8)  # the default (every core) oversubscribes a shared machine
    torch.manual_seed(0)
    globals()[sys.argv[1]](sys.argv[2])
    print('ok')

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


def patch_sequences(*loops):
    real = data.benchmark_split

    def short(*args, **kwargs):
        split = real(*args, **kwargs)
        split.train, split.heldout = split.train[:, :SEQ].contiguous(), split.heldout[:, :SEQ].contiguous()
        return split
    for loop in (train, *loops):
        loop.benchmark_split, loop.SEQ = short, SEQ


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
    polar = type('Polar', (), dict(direction=lambda self, p, pending: m.polar(pending)))()
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


def curvature_damping(scratch):
    """E8: the tracked directions are the pending update's top singular directions; h matches an exact Hessian-vector
    product; c is the across-sequence variance of the projections; the factors follow the formula; 'step' damps after
    polar() and 'pending' before it, where polar() undoes most of it."""
    from loud.curvature import CURVATURE_ROWS, damp
    trainer, m = proxy_trainer(per_sequence_grads=True, curvature_damping=3.0, damp_directions=2, damp_every=2)
    shaper, model = trainer.shaper, trainer.model
    rows = batches(3, model.config.vocab_size)
    trainer.update(1, rows[0])  # a refresh: at update 1, fast and slow are multiples of g, so pending is too
    params = dict(model.named_parameters())
    for i, name in enumerate(shaper.names):
        a, b = shaper.left[i], shaper.right[i]
        assert torch.allclose(a.T @ a, torch.eye(2), atol=1e-5) and torch.allclose(b.T @ b, torch.eye(2), atol=1e-5)
        u, _, vh = torch.linalg.svd(trainer.accumulators[list(params).index(name)], full_matrices=False)
        assert (u[:, :2] * a).sum(0).abs().min() > 0.999 and (vh[:2].T * b).sum(0).abs().min() > 0.999, name
        g = trainer.pergrad.stack(name).float()
        projections = torch.einsum('brc,rk,ck->bk', g, a, b)
        assert torch.allclose(shaper.noise[i], projections.var(0), rtol=1e-4)

    i = shaper.names.index('transformer.h.1.mlp.c_fc.weight')
    d = torch.outer(shaper.left[i][:, 0], shaper.right[i][:, 0])
    exact_model = copy.deepcopy(model).float().requires_grad_(True)
    weight = dict(exact_model.named_parameters())[shaper.names[i]]
    curvature_rows = rows[0][:CURVATURE_ROWS]
    with torch.enable_grad():
        loss = exact_model(curvature_rows[:, :-1].contiguous(), curvature_rows[:, 1:].contiguous())
        g = torch.autograd.grad(loss, weight, create_graph=True)[0]
        hvp = torch.autograd.grad((g * d).sum(), weight)[0]
    exact_h = (hvp * d).sum().item()
    assert abs(shaper.h[i][0].item() - exact_h) < 2e-2 * abs(exact_h) + 1e-6, (shaper.h[i][0].item(), exact_h)

    trainer.diagnose(1)
    trainer.update(2, rows[1])  # not a refresh: one more noise sample, factors from the formula
    lr = trainer.reference.settings.anvil_lr
    for i, name in enumerate(shaper.names):
        p = params[name]
        scale = 3.0 * lr ** 2 * max(1, p.size(0) / p.size(1))
        expected = 1 / (1 + (shaper.noise[i] / 2) / (4 * shaper.h[i].square() * scale))
        assert shaper.samples == 2 and torch.allclose(shaper.factors[i], expected, rtol=1e-5)
    trainer.diagnose(2)
    trainer.update(3, rows[2])  # a refresh again: new directions, and a summary with their overlap
    report = trainer.diagnose(3)['damping']
    assert {'h', 'c', 'h2_over_c', 'factor', 'overlap'} <= set(report['mlp.c_fc'])

    p = params[shaper.names[i]]
    pending = torch.randn(p.shape, generator=torch.Generator().manual_seed(7))
    a, b, f = shaper.left[i], shaper.right[i], shaper.factors[i]
    shaper.refreshing = False
    assert torch.equal(shaper.direction(p, pending), damp(m.polar(pending), a, b, f))
    shaper.where = 'pending'
    assert torch.equal(shaper.direction(p, pending), m.polar(damp(pending, a, b, f)))
    x = torch.randn(p.shape, generator=torch.Generator().manual_seed(8))
    coefficient = lambda y: (a.T @ y * b.T).sum(1)
    assert torch.allclose(coefficient(damp(x, a, b, f)), f * coefficient(x), atol=1e-5)
    rest = lambda y: y - a @ torch.diag(coefficient(y)) @ b.T
    assert torch.allclose(rest(damp(x, a, b, f)), rest(x), atol=1e-5)  # other directions are left alone

    # why 'step' is the default: halving a gradient-like matrix's top singular directions before polar() leaves them
    # at about 1 after it; halving them after polar() halves them
    U, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(9)))
    V, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(10)))
    G = U @ torch.diag(torch.logspace(0, -3, 128)) @ V.T
    a, b, half = U[:, :4], V[:, :4], torch.full((4,), 0.5)
    top = lambda y: (a.T @ y * b.T).sum(1)
    assert top(m.polar(damp(G, a, b, half))).min() > 0.9
    assert top(damp(m.polar(G), a, b, half)).max() < 0.6


def fisher_factor(scratch):
    """E9: the sums of d d^T come from the sampled-label gradient at each block matrix's output (checked against an
    independent forward, token by token), leave the training gradient untouched, and the update is the damped inverse
    root of S times the pending update at the polar factor's norm."""
    from loud.fisher import SAMPLE_SEED, inverse_root
    trainer, m = proxy_trainer(fisher_damping=0.1)
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
    tokens = 4 * (SEQ - 1)
    assert shaper.tokens == tokens

    shaper.before_step(1)
    S, R = shaper.S[0], shaper.R[0]
    assert torch.allclose(S, expected[0] / tokens, rtol=1e-3, atol=1e-9)
    damped = S + 0.1 * torch.linalg.eigvalsh(S).clamp_min(0).mean() * torch.eye(len(S))
    assert torch.allclose(R @ damped @ R, torch.eye(len(S)), atol=1e-3)
    assert torch.allclose(inverse_root(S, 0.1), R)
    p = dict(model.named_parameters())[names[0]]
    pending = torch.randn(p.shape, generator=torch.Generator().manual_seed(11))
    update = shaper.direction(p, pending)
    assert abs(update.norm().item() - min(p.shape) ** 0.5) < 1e-3
    assert torch.allclose(update / update.norm(), R @ pending / (R @ pending).norm(), atol=1e-6)


def population_run(scratch):
    """E3 end to end on a tiny model, once per rule: every remaining sequence feeds one copy's update at most once,
    the copies' rates and ANVIL II's clock follow the single run's update numbers, and the rule acts as described."""
    from loud import population
    m = refs.trainer_module('boostcut')
    patch_attention()
    patch_sequences(population)
    for rule in ('average', 'select', 'ensemble'):
        spec = dict(experiment='cpu', variant=rule, seed=0, task='population', args=dict(
            preset='2m', sequences=72, heldout=4, rule=rule, every=2, official_eval=False))
        out = Path(scratch) / rule
        result = population.run(spec, out)
        config = json.loads((out / 'config.json').read_text())
        metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        events = [json.loads(line) for line in (out / 'population.jsonl').read_text().splitlines()]
        # 68 update sequences: 17 updates of 4; 3 before the branch, then 56 left, 14 per copy: 3 rounds of 4 each
        assert config['branch_step'] == 3 and config['rounds'] == 3 and config['unused_sequences'] == 8
        assert [r['step'] for r in metrics] == list(range(1, 16)) and result['updates'] == 15
        turns = [(r['round'], r['copy']) for r in metrics if r['phase'] == 'population']
        assert turns == [(r, c) for r in (1, 2, 3) for c in range(4)]
        anvil = m.Anvil([torch.zeros(2, 2)], 1.0, 0.0, 17)
        for r in metrics:
            anvil.t = r['step'] - 1
            assert r['anvil_beta'] == anvil.rail_beta(), 'ANVIL II does not follow the single run'
        base = metrics[0]['lr']['anvil']
        assert all(abs(r['lr']['anvil'] - base * min(1.0, (17 - r['step'] + 1) / 17 / 0.3)) < 1e-12 for r in metrics)
        assert [e['round'] for e in events] == [2, 3] and all(len(e['midpoints']) == 6 for e in events)
        assert all(abs(e['barriers'][p] - (e['midpoints'][p] - max(e['copies'][int(p[0])], e['copies'][int(p[2])])))
                   < 1e-12 for e in events for p in e['midpoints'])
        if rule == 'average':
            assert all(e['action'] == 'average' for e in events) and result['final'] == 'mean'
        if rule == 'select':
            assert all(set(e['action']) == {'select', 'replace'} for e in events)
            last = events[-1]
            assert result['final'] == f'copy {last["action"]["select"]}'
            assert result['heldout_loss'] == min(result['final_copy_scores'])
        if rule == 'ensemble':
            mean, best = result['final_mean_score'], min(result['final_copy_scores'])
            assert result['final'] == ('mean' if mean < best else f'copy {result["final_copy_scores"].index(best)}')
            assert result['heldout_loss'] == min(mean, best)


def population_rules(scratch):
    """The rules on live copies: averaging leaves every copy at the mean with its own optimizer state; selection gives
    the worst copy the best's weights and an unaliased copy of its optimizer state."""
    from loud.heldout import HeldoutScorer
    from loud.common import JsonlLog
    from loud.population import Clock, Population
    trainer, m = proxy_trainer()
    rows = batches(3, trainer.model.config.vocab_size)
    for step in (1, 2):
        trainer.update(step, rows[step - 1])
    heldout = HeldoutScorer(rows[2], Path(scratch) / 'heldout.jsonl', torch.device('cpu'))
    events = JsonlLog(Path(scratch) / 'population.jsonl')
    o = dict(trainer.o, copies=3, rule='average')
    population = Population(o, trainer, trainer.reference, torch.device('cpu'), 0, heldout, events,
                            Clock(torch.device('cpu')))
    for c, member in enumerate(population.copies):  # make them differ
        member.optimizer.optimizers[0].t = 2 + c
        member.update(3 + c, rows[c])
    states = [copy.deepcopy(member.optimizer.optimizers[0].state_dict()) for member in population.copies]
    population.checkpoint(1, 5)
    weights = [list(member.model.parameters()) for member in population.copies]
    for ps in zip(*weights):
        assert all(torch.equal(ps[0], q) for q in ps[1:]), 'averaging left the copies apart'
    for member, state in zip(population.copies, states):
        assert all(torch.equal(member.optimizer.optimizers[0].state[p]['slow'], state['state'][i]['slow'])
                   for i, p in enumerate(member.optimizer.optimizers[0].param_groups[0]['params']))

    population.o = dict(o, rule='select')
    for c, member in enumerate(population.copies):
        member.update(6 + c, rows[(c + 1) % 3])
    scores = population.checkpoint(2, 8)
    best, worst = scores.index(min(scores)), scores.index(max(scores))
    for p, q in zip(population.copies[worst].model.parameters(), population.copies[best].model.parameters()):
        assert torch.equal(p, q)
    best_anvil = population.copies[best].optimizer.optimizers[0]
    kept = {id(p): best_anvil.state[p]['slow'].clone() for p in best_anvil.param_groups[0]['params']}
    population.copies[worst].update(9, rows[0])  # its state moves; the best copy's must not
    assert all(torch.equal(best_anvil.state[p]['slow'], kept[id(p)]) for p in best_anvil.param_groups[0]['params'])


if __name__ == '__main__':
    torch.set_num_threads(8)  # the default (every core) oversubscribes a shared machine
    torch.manual_seed(0)
    globals()[sys.argv[1]](sys.argv[2])
    print('ok')

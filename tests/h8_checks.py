"""CPU checks of H8 (loud/h8.py), one per process, as method_checks.py's:

    python tests/h8_checks.py <check> <scratch dir>

They run PR #40 eagerly with dense attention on a 2-layer model (method_checks.cut_module), so the record's own Boost
and Cut estimators are the references.
"""
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from method_checks import CUT_SEQ, cut_module, cut_rows_data, patch_sequences  # noqa: E402


def matrices(model):
    return [n for n, p in model.named_parameters() if p.ndim == 2 and n.startswith('transformer.h.')]


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item()


def record_setup(**changes):
    m = cut_module()
    from loud.cut import record_settings
    s = record_settings(m, '2m', 4)
    for k, v in changes.items():
        setattr(s, k, v)
    model = m.make_model(torch.device('cpu'), 0, s.model, s.projection_std, s.head_std)
    with torch.no_grad():  # weights a few updates in: the init's zero writers would make every credit trivial
        for p in model.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())))
    rows = cut_rows_data()[:2]
    return m, s, model, rows[:, :-1].contiguous(), rows[:, 1:].contiguous()


def h8_gradients(scratch):
    """The cut graph without detaches is GPT.forward; synthesis mode with the identity scheme matches PR #40's Cut
    estimate with its cross-token patches off, no future draws and every MLP unit routed (plan v2's validation: a
    per-matrix cosine of at least 0.99); cut mode equals synthesis everywhere but the MLP writers, which it also feeds
    the own-query path through the next block."""
    from loud import h8
    from loud.options import h8_options
    m, s, model, x, y = record_setup(value_path=False, path_scalars=False, key_path_draws=0, gate_path=False,
                                     future_draws=0)
    s.route_k = 4 * s.model.n_embd
    model.requires_grad_(True)
    losses = h8.cut_forward(model, x, y, held=False)[0]
    assert torch.equal(losses, model(x, y, loss_reduction='none').view_as(y)), 'the cut graph is not GPT.forward'

    params = dict(model.named_parameters())
    scheme = h8.Scheme(h8_options({}), s.model.n_embd, torch.device('cpu'), 0)
    synth, _, _ = h8.synthesis_gradients(model, x, y, scheme, 1)
    synth = {n: synth[id(p)] for n, p in params.items()}
    record = m.Cut(model.requires_grad_(False), s, 0, 0, 1, 1, None).estimate(x, y)
    model.requires_grad_(True)
    low = {n: round(cosine(synth[n], record[n]), 4) for n in matrices(model)}
    print('synthesis vs PR #40 Cut, block matrices:', low)
    assert all(c >= 0.99 for c in low.values()), low
    for n in ('transformer.wte.weight', 'lm_head.weight', 'resid_lambdas', 'x0_lambdas'):
        assert cosine(synth[n], record[n]) > 0.99, (n, cosine(synth[n], record[n]))

    loss = h8.cut_forward(model, x, y)[0].sum() / y.numel()
    cut = dict(zip(params, torch.autograd.grad(loss, list(params.values()), allow_unused=True)))
    for n in params:
        if '.mlp.' in n:
            assert cosine(cut[n], synth[n]) > 0.9, n  # plus the own-query path through the next block
        elif cut[n] is not None:
            assert torch.allclose(cut[n], synth[n], atol=1e-6, rtol=1e-4), n


def h8_boost(scratch):
    """Boost mode against PR #40's Boost estimate (its readouts, sublayer by sublayer)."""
    from loud import h8
    m, s, model, x, y = record_setup()
    model.requires_grad_(True)
    params = dict(model.named_parameters())
    loss, _ = h8.boost_loss(model, x, y)
    ours = dict(zip(params, torch.autograd.grad(loss, list(params.values()), allow_unused=True)))
    record = m.Boost(model.requires_grad_(False), s, 0, 0, 1, 1, None).estimate(x, y)
    found = {n: round(cosine(ours[n], record[n]), 4) for n in params if ours[n] is not None and record[n].any()}
    print('boost vs PR #40 Boost:', found)
    assert all(c > 0.98 for c in found.values()), found


def h8_schemes(scratch):
    """Partial rows are unbiased under the random column signs, for either assignment; token assignment gives each
    token its block; the identity prior at the full cycle is exact."""
    from loud.h8 import Scheme
    from loud.options import h8_options
    W, T = 128, 16
    g = torch.randn(1, T, W, generator=torch.Generator().manual_seed(1))
    for assign in ('time', 'token'):
        scheme = Scheme(h8_options(dict(preset='2m', rows=32, row_assign=assign)), W, torch.device('cpu'), 0)
        mean = sum(scheme.read(g, step) for step in range(4000)) / 4000
        assert (mean - g).norm() < 0.1 * g.norm(), (assign, ((mean - g).norm() / g.norm()).item())
        one = scheme.read(g, 0)
        assert not torch.allclose(one, g)
    from loud.options import H8
    full = Scheme(dict(H8, rows=128, prior='identity'), W, torch.device('cpu'), 0)  # every row: the prior is exact
    out = full([g, g], 0, [torch.randn_like(g), torch.randn_like(g)])  # S0 keeps the plain read, S1 the prior
    assert all(torch.allclose(o, g, atol=1e-5) for o in out)
    try:
        h8_options(dict(preset='2m', rows=128, prior='identity'))  # an arm that changes nothing is refused
    except ValueError as e:
        assert 'rows below the width' in str(e)
    else:
        raise AssertionError('a prior with every row was accepted')


def h8_reforward(scratch):
    """reforward: no move gives the clean losses; small moves at every site change each token's loss by the cut
    graph's credit there to first order; the later sites' changes it reports are the moved stream's."""
    from loud import h8
    m, s, model, x, y = record_setup()
    model.requires_grad_(True)
    losses, sites, top, _, _ = h8.cut_forward(model, x, y)
    credits = torch.autograd.grad(losses.sum(), sites)
    replay = h8.Replay(model.requires_grad_(False), x, y, chunk=16)
    B, T, W = sites[0].shape
    base = replay(0, torch.zeros(B, T, 1, W))[..., 0]
    assert torch.allclose(base, losses.detach(), atol=1e-5), (base - losses).abs().max()
    gen = torch.Generator().manual_seed(3)
    for k in range(len(sites)):
        rms = sites[k].detach().square().mean(-1, keepdim=True).sqrt()
        delta = (1e-3 * rms * torch.randn(B, T, W, generator=gen)).unsqueeze(2)
        moved, at = replay(k, delta, sites=True)
        measured = (moved[..., 0] - base).flatten()
        predicted = (credits[k] * delta[:, :, 0]).sum(-1).flatten()
        assert cosine(measured, predicted) > 0.999, (k, cosine(measured, predicted))
        assert set(at) == set(range(k + 1, len(sites)))
    zero, at = replay(1, torch.zeros(B, T, 1, W), sites=True)
    assert all(v.abs().max() < 1e-5 for v in at.values())


def h6_tools(scratch):
    """H6 on a tiny model. With every writer zeroed the identity path is exact, so credit_dump's guess matches the
    exact credit (cosine 1, variance ratio 0); on a live model, E5's deep readout from S1 recovers the later sites'
    credit up to the model's nonlinearity (a small probe keeps it small: the trained record's is far more linear);
    linearity_check reports its quantiles; h8_validation is H8 against PR #40's Cut, as h8_gradients."""
    from loud.diagnostics import Toolkit, site_names
    m, s, model, x, y = record_setup()
    x, y = x[:1], y[:1]
    zero = __import__('copy').deepcopy(model)
    with torch.no_grad():
        for block in zero.transformer.h:
            block.attn.c_proj.weight.zero_()
            block.mlp.c_proj.weight.zero_()
    summary, tensors = Toolkit(m, zero, x, y).credit_dump()
    names = site_names(s.model.n_layer)
    for name in names:
        assert summary[name]['cosine']['p10'] > 0.999 and summary[name]['variance_ratio'] < 1e-4, (name, summary[name])
    assert tensors['S1']['exact'].shape == (x.shape[1], s.model.n_embd)
    tools = Toolkit(m, model, x, y)
    deep, _ = tools.deep_readout_check(sigma=1e-3, chunk=8)
    assert tools.deep_readout_check(sigma=1e-3, chunk=8, precision='float32')[0]['settings']['precision'] == 'float32'
    delta, _ = tools.deep_readout_check(sigma=1e-3, chunk=8, precision='bf16_delta')
    for name in names[2:]:  # the chain's way of differencing keeps the readout (bf16 products of the changes)
        assert delta[name]['cosine ridge 0.0']['median'] > 0.9, (name, delta[name])
    from loud import h8
    replay, exact32 = h8.Replay(model, x, y), h8.DeltaReplay(model, x, y, dtype=torch.float32)
    move = 1e-2 * torch.randn(1, x.shape[1], 2, s.model.n_embd, generator=torch.Generator().manual_seed(5))
    a = replay(1, move) - replay(1, torch.zeros_like(move))
    b = exact32(1, move) - exact32(1, torch.zeros_like(move))
    assert (a - b).norm() < 1e-3 * a.norm(), 'DeltaReplay in float32 is not Replay'
    for name in names[2:]:
        assert deep[name]['cosine ridge 0.0']['median'] > 0.95, (name, deep[name])
        assert deep[name]['condition']['median'] >= 1
    linear, _ = tools.linearity_check(sigmas=(1e-3, 1e-1), draws=1)
    assert set(linear) == set(names)
    assert all(linear[n]['float32 0.001']['nonlinearity']['median'] < linear[n]['float32 0.1']['nonlinearity']['median']
               for n in names)
    valid, _ = tools.h8_validation()
    assert not valid['below_099'], valid
    exact, _ = tools.h8_validation('float32')  # both sides in float32: the record's fp32 chain, its capture
    assert not exact['below_099'] and exact['precision'] == 'float32', exact
    synth, _ = tools.cos_to_autograd('h8')  # H8's estimate is the cut graph's: close to autograd, short of 1
    assert set(synth['sites']) == set(names) and 0.5 < synth['all'] < 1, synth
    medians = [synth['sites'][n]['median'] for n in names]  # the top site, with the least above it, misses the least
    assert medians[-1] == max(medians) and medians[0] < 1, medians
    record, _ = tools.cos_to_autograd('cut')
    assert 0.5 < record['all'] < 1 and abs(record['block_matrices'] - synth['block_matrices']) < 0.1, (record, synth)


def h8_run(scratch):
    """Complete H8 runs through loud/train.py: synthesis with a partial-row scheme, prior and both logs; cut mode
    without Boost."""
    from loud import h8
    import loud.train
    cut_module()
    patch_sequences(length=CUT_SEQ, vocab=512)
    for variant, args in (('synth', dict(rows=32, prior='identity', compare_every=1, rank_every=2)),
                          ('cut', dict(mode='cut', boost=False))):
        spec = dict(experiment='cpu', variant=variant, seed=0, task='h8', args=dict(
            preset='2m', sequences=72, heldout=32, batch=4, official_eval=False, **args))
        out = Path(scratch) / variant
        result = h8.run(spec, out)
        records = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        assert result['complete'] and len(records) == 10, (variant, len(records))
        phases = [r['phase'] for r in records]
        if variant == 'synth':
            assert phases == ['boost'] * 2 + ['cut'] * 8, phases
            assert all(-1 <= r['cosine_to_identity'] <= 1 for r in records if r['phase'] == 'cut')
            assert any('stable_rank' in r for r in records)
        else:
            assert set(phases) == {'cut'}


def h8_latent(scratch):
    """E14's dynamics model and loss, the document mask, E15's predictors, the EMA teacher, and where E14's gradient
    reaches, each against a direct computation."""
    import torch.nn.functional as F
    from tokenizers import Tokenizer
    from loud import h8, latent, paths
    from loud.ema import EMAWeights
    tokenizer = Tokenizer.from_file(str(paths.REPO / 'external' / 'noise' / 'tokenizer' / 'tokenizer.json'))
    assert latent.EOT == tokenizer.token_to_id('<|endoftext|>')

    gen = torch.Generator().manual_seed(4)
    B, T, W = 2, 9, 16
    h, e, target = (torch.randn(B, T, W, generator=gen) for _ in range(3))
    idx = torch.randint(1, 50, (B, T), generator=gen)
    idx[0, 3], idx[1, 7] = latent.EOT, latent.EOT  # two documents open: targets at those positions are masked
    dynamics = latent.Dynamics(W)
    fc, proj = dynamics.mlp[0].weight, dynamics.mlp[2].weight
    pred = h[:, :-1] + F.gelu(F.layer_norm(torch.cat([e[:, 1:], h[:, :-1]], -1), (2 * W,)) @ fc.T) @ proj.T
    diff = (pred - target[:, 1:]).abs()
    elem = torch.where(diff < 1, 0.5 * diff ** 2, diff - 0.5)
    keep = (idx[:, 1:] != latent.EOT).float().unsqueeze(-1)
    direct = (elem * keep).sum() / (keep.sum() * W)
    assert torch.allclose(latent.nextlat_loss(dynamics, h, e, idx, target), direct, atol=1e-6)
    moved = target.clone()
    moved[0, 3] += 10
    assert torch.equal(latent.nextlat_loss(dynamics, h, e, idx, moved),
                       latent.nextlat_loss(dynamics, h, e, idx, target))
    assert fc.shape == (latent.HIDDEN, 2 * W) and proj.shape == (W, latent.HIDDEN)

    targets = latent.LatentTargets(2, W, lens=True, weight=0.5)
    assert len(targets.predictors) == 3  # every sublayer of 2 blocks but the last
    targets.target = target
    stream = torch.randn(B, T, W, generator=gen, requires_grad=True)
    miss = 1 - F.cosine_similarity(targets.predictors[1](stream), target, dim=-1)
    assert torch.allclose(targets(1, stream), 0.5 * miss.sum())
    own = targets(0, stream, own=True)
    grads = torch.autograd.grad(own, [stream] + list(targets.predictors[0].parameters()), allow_unused=True)
    assert grads[0] is not None and all(g is None for g in grads[1:]), "the embedding's path trains no predictor"

    student = torch.nn.Linear(3, 3).bfloat16()
    teacher = EMAWeights(student, steps=100)
    assert teacher.model.weight.dtype == torch.float32
    assert abs(teacher.decay(0) - 0.996) < 1e-12 and abs(teacher.decay(100) - 0.9995) < 1e-12
    assert abs(teacher.decay(50) - (0.996 + 0.9995) / 2) < 1e-12
    before = teacher.model.weight.clone()
    with torch.no_grad():
        student.weight.add_(1.0)
    teacher.update(50)
    d = teacher.decay(50)
    assert torch.allclose(teacher.model.weight, d * before + (1 - d) * student.weight.float(), atol=1e-7)

    # E14's gradient in the cut graph: the trunk and wte (through x0 and the next token's embedding), not the head
    m, s, model, x, y = record_setup()
    model.requires_grad_(True)
    _, sites, top, _, _ = h8.cut_forward(model, x, y)
    hidden = h8.um().norm(top)
    dynamics = latent.Dynamics(s.model.n_embd)
    aux = latent.nextlat_loss(dynamics, hidden, sites[0], x, hidden.detach())
    named = dict(model.named_parameters())
    reach = dict(zip(named, torch.autograd.grad(aux, list(named.values()), retain_graph=True, allow_unused=True)))
    assert reach['lm_head.weight'] is None or not reach['lm_head.weight'].any()
    for n in ('transformer.wte.weight', 'transformer.h.0.mlp.c_fc.weight', 'transformer.h.1.attn.c_q.weight',
              'resid_lambdas', 'x0_lambdas'):
        assert reach[n] is not None and reach[n].any(), n
    assert all(g.any() for g in torch.autograd.grad(aux, list(dynamics.parameters())))


def h8_latent_run(scratch):
    """Complete tiny runs of E14 (EMA target from the first update; the live target from the handoff) and E15 (the
    latent target beside the readouts, and alone), with their logs."""
    from loud import h8
    cut_module()
    patch_sequences(length=CUT_SEQ, vocab=512)
    h8.AUX_LOG_EVERY, h8.LATENT_LOG_EVERY = 2, 1
    arms = {'nextlat_ema_start': dict(nextlat=1.0, nextlat_onset='start'),
            'nextlat_live': dict(nextlat=0.3, nextlat_target='live'),
            'latent_plus': dict(latent_boost='plus'),
            'latent_only': dict(latent_boost='only')}
    for variant, args in arms.items():
        spec = dict(experiment='cpu', variant=variant, seed=0, task='h8', args=dict(
            preset='2m', sequences=72, heldout=32, batch=4, mode='cut', official_eval=False, **args))
        out = Path(scratch) / variant
        result = h8.run(spec, out)
        records = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        assert result['complete'] and len(records) == 10, variant
        by_step = {r['step']: r for r in records}
        if 'nextlat' in args:
            logged = [r['step'] for r in records if 'nextlat' in r]
            first = 2 if args.get('nextlat_onset') == 'start' else 4  # the handoff is at update 3
            assert logged and logged[0] == first and all(-1 <= r['nextlat']['top_cosine'] <= 1
                                                          for r in records if 'nextlat' in r), (variant, logged)
            assert {'top', 'S0', 'S2'} <= set(by_step[2]['effective_rank']) and 'effective_rank' in by_step[10]
        else:
            boost = [r for r in records if r['phase'] == 'boost']
            kinds = {'latent', 'readout'}
            assert all(set(r['latent_boost']) == {'0', '1', '2'} for r in boost), boost
            assert all(set(v) == kinds for r in boost for v in r['latent_boost'].values())
            assert -1 <= by_step[2]['handoff_cut_cosine'] <= 1


def block_local_reference(h8, model, x, y, transform):
    """Synthesis mode's block-local gradients written out from its definition, for the site credits transformed by
    `transform` (a list of [B, T, W] float credits to a list): block i's attention and value table from the credit at
    S(i+1), its MLP from resid_lambdas[i+1] times the credit at S(i+2) (the last block's from the top stream's), wte
    from S0."""
    L = model.config.n_layer
    losses, sites, top, attn_out, mlp_out = h8.cut_forward(model, x, y)
    *g, g_top = torch.autograd.grad(losses.sum() / y.numel(), sites + [top], retain_graph=True)
    g = transform([c.float() for c in g])
    lam = model.resid_lambdas.detach()
    out = {}
    for i, block in enumerate(model.transformer.h):
        attn = list(block.attn.parameters()) + ([model.value_embeds[str(i)].weight] if str(i) in model.value_embeds
                                                else [])
        out.update(zip(attn, torch.autograd.grad(attn_out[i], attn, g[i + 1], retain_graph=True)))
        mlp = list(block.mlp.parameters())
        out.update(zip(mlp, torch.autograd.grad(mlp_out[i], mlp, lam[i + 1] * g[i + 2] if i < L - 1 else g_top,
                                                retain_graph=True)))
    out[model.transformer.wte.weight] = torch.autograd.grad(sites[0], model.transformer.wte.weight, g[0])[0]
    return out


def synthesized(h8, model, x, y, scheme_args, **kwargs):
    from loud.options import h8_options
    scheme = h8.Scheme(h8_options(dict(preset='2m', **scheme_args)), model.config.n_embd, torch.device('cpu'), 0)
    grads, _, fields = h8.synthesis_gradients(model, x, y, scheme, 1, **kwargs)
    return {p: grads[id(p)] for p in model.parameters()}, fields


def h8_shaping(scratch):
    """E10. (a) clip: each site's credits capped at 3x the batch's median token norm, then the block-local gradients,
    as written out. (b) agree: against per-token contributions summed and sign-counted token by token (an entry kept
    where at least 70% of all tokens back its majority sign; zeros back neither); with the threshold at 0 every entry
    is kept and the gradients are synthesis mode's own."""
    from torch import nn
    from loud import h8
    _, _, model, x, y = record_setup()
    model.requires_grad_(True)

    def capped(g):
        norms = g.norm(dim=-1, keepdim=True)
        cap = 3 * torch.sort(norms.flatten()).values[(norms.numel() - 1) // 2]
        return torch.where(norms > cap, g * cap / norms, g)
    ours, _ = synthesized(h8, model, x, y, dict(clip_credit=True))
    expected = block_local_reference(h8, model, x, y, lambda gs: [capped(g) for g in gs])
    assert all(torch.allclose(ours[p], v, atol=1e-6, rtol=1e-4) for p, v in expected.items()), 'clip'
    plain, _ = synthesized(h8, model, x, y, {})
    g = torch.randn(2, 10, 8, generator=torch.Generator().manual_seed(4))
    g[0, 3] *= 20  # an outlier token: capped at 3x the median norm; the rest untouched
    capped_g, norms = h8.clip(g), g.norm(dim=-1)
    assert torch.allclose(capped_g[0, 3].norm(), 3 * norms.flatten().median())
    assert torch.equal(capped_g.flatten(0, 1)[norms.flatten() <= 3 * norms.flatten().median()],
                       g.flatten(0, 1)[norms.flatten() <= 3 * norms.flatten().median()])

    agreed, _ = synthesized(h8, model, x, y, {}, agree=True)
    tokens = y.numel()
    losses, sites, top, attn_out, mlp_out = h8.cut_forward(model, x, y)
    *credits, g_top = torch.autograd.grad(losses.sum() / tokens, sites + [top], retain_graph=True)
    io = {}
    with h8.capture_linears(model, io):
        losses, sites, top, attn_out, mlp_out = h8.cut_forward(model, x, y)
    lam, L, kept = model.resid_lambdas.detach(), model.config.n_layer, []
    for i, block in enumerate(model.transformer.h):
        for module, out, g in ((block.attn, attn_out[i], credits[i + 1]),
                               (block.mlp, mlp_out[i], lam[i + 1] * credits[i + 2] if i < L - 1 else g_top)):
            linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
            outputs = torch.autograd.grad(out, [io[m][1] for m in linears], g, retain_graph=True)
            for m, G in zip(linears, outputs):
                G2, X2 = G.flatten(0, -2), io[m][0].flatten(0, -2)
                total, up, down = (torch.zeros_like(m.weight) for _ in range(3))
                for t in range(tokens):  # token by token
                    c = torch.outer(G2[t], X2[t])
                    total, up, down = total + c, up + (c > 0), down + (c < 0)
                keep = torch.maximum(up, down) >= 0.7 * tokens  # 70% of all the tokens back the majority sign
                assert torch.allclose(agreed[m.weight], total * keep, atol=1e-6, rtol=1e-4), m
                kept.append(keep.float().mean().item())
    assert sum(kept) > 0 and max(kept) < 1, kept  # a real filter: 70% agreement over 128 tokens keeps few entries
    contributions = torch.zeros(10, 1)  # Codex's case: 4 positive, 6 zero: 40% back the sign, so it goes
    contributions[:4] = 1
    linear = nn.Linear(1, 1, bias=False)
    out = linear(contributions)
    dropped = h8.agreed_gradients(linear, [linear.weight], out, torch.ones_like(out), {linear: (contributions, out)},
                                  10, [])
    assert dropped[id(linear.weight)].abs().sum() == 0
    h8.AGREE = 0.0
    every, _ = synthesized(h8, model, x, y, {}, agree=True)
    h8.AGREE = 0.7
    assert all(torch.allclose(every[p], plain[p], atol=1e-6, rtol=1e-4) for p in plain), 'agree at 0 is not plain'


def h8_runway(scratch):
    """E11: each token's runway against a move-by-move Replay; alphas normalized to mean 1 per site; synthesis mode
    weights each site's credits by them."""
    from loud import h8
    _, _, model, x, y = record_setup()
    model.requires_grad_(True)
    losses, sites, top, _, _ = h8.cut_forward(model, x, y)
    credits = torch.autograd.grad(losses.sum() / y.numel(), sites)
    alphas, chosen = h8.runway_weights(model, x, y, sites, credits)
    replay = h8.Replay(model.requires_grad_(False), x, y)
    model.requires_grad_(True)
    grid = (0.0,) + h8.RUNWAY
    agree = []
    for k, (h, g) in enumerate(zip(sites, credits)):
        h, g = h.detach(), g.detach()
        reach = h.square().mean(-1, keepdim=True).sqrt() * h.shape[-1] ** 0.5
        moved = torch.stack([replay(k, (d * reach * -g / g.norm(dim=-1, keepdim=True)).unsqueeze(2))[..., 0]
                             for d in grid], -1)
        best = moved[..., 1:].argmin(-1) + 1
        expect = torch.where(moved[..., 1:].min(-1).values < moved[..., 0], best, torch.zeros_like(best))
        agree.append((expect == chosen[k]).float().mean().item())
        raw = torch.tensor(grid)[chosen[k]]
        assert torch.allclose(alphas[k] * raw.mean(), raw, atol=1e-6) and abs(alphas[k].mean() - 1) < 1e-5, k
    assert min(agree) > 0.99, agree
    ours, fields = synthesized(h8, model, x, y, {}, runway=True, log=True)
    expected = block_local_reference(h8, model, x, y, lambda gs: [g * a.unsqueeze(-1) for g, a in zip(gs, alphas)])
    assert all(torch.allclose(ours[p], v, atol=1e-6, rtol=1e-4) for p, v in expected.items()), 'runway weights'
    assert set(fields['runway']) == {f'S{k}' for k in range(len(sites))}
    assert all(abs(sum(v) - 1) < 1e-3 for v in fields['runway'].values())


def h8_ngram(scratch):
    """E13's statistics: the n-gram counts against a direct count; Witten-Bell probabilities sum to one over the
    vocabulary for seen and unseen histories; scoring is causal; copies on a crafted row; the low tier's partial rows
    are unbiased and the high tier is exact; the tier diagnostic on a tiny model."""
    import numpy as np
    from collections import Counter
    from loud import h8, ngram
    from loud.options import h8_options
    V = 16
    rows = np.random.default_rng(0).integers(0, V, (3, 41))
    table = ngram.NGram(V)
    before = table.losses(rows)
    assert np.allclose(before, np.log(V)), 'an empty table is uniform'
    table.insert(rows)
    for n in range(1, 5):
        # every window of n ids whose last id is a target (not a row's first id), packed oldest first
        direct = Counter(tuple(r[s:s + n]) for r in rows for s in range(len(r) - n + 1) if s + n - 1 >= 1)
        packed = {}
        for window, c in direct.items():
            key = 0
            for v in window:
                key = (key << ngram.BITS) | int(v)
            packed[key] = c
        assert table.pairs[n - 1] == packed, n
        history = rows[0][5:5 + max(n - 1, 1)].tolist()  # every order up to its length applies
        assert abs(table.distribution(history).sum() - 1) < 1e-9, n
    assert abs(table.distribution([V - 1, V - 1, V - 1]).sum() - 1) < 1e-9  # unseen histories
    assert not np.allclose(table.losses(rows), before) and np.all(table.losses(rows) < before + 1e-9)
    # a unigram value by hand: P(w) = (c(w) + T / V) / (N + T)
    counts = Counter(rows[:, 1:].flatten().tolist())
    N, T = sum(counts.values()), len(counts)
    w = rows[0, 1]
    p1 = (counts[w] + T / V) / (N + T)
    one = ngram.NGram(V, order=1)
    one.insert(rows)
    assert abs(one.probabilities(rows[:1, :2])[0, 0] - p1) < 1e-12

    crafted = np.array([[5, 6, 7, 8, 1, 2, 5, 6, 7, 8, 9]])
    flags = ngram.copies(crafted)[0]
    assert flags.tolist() == [False] * 8 + [True, False], flags  # the second 5, 6, 7 is followed by 8 again: a copy

    scheme = h8.Scheme(h8_options(dict(preset='2m', budget_theta=3.0)), 128, torch.device('cpu'), 0)
    g = torch.randn(2, 12, 128, generator=torch.Generator().manual_seed(2))
    low = torch.rand(2, 12, generator=torch.Generator().manual_seed(3)) < 0.5
    one = scheme([g], 1, low=low)[0]
    assert torch.equal(one[~low], g[~low]) and not torch.allclose(one[low], g[low])
    mean = sum(scheme([g], s, low=low)[0] for s in range(3000)) / 3000
    assert (mean - g).norm() < 0.1 * g.norm(), ((mean - g).norm() / g.norm()).item()

    _, _, model, x, y = record_setup()
    rows = cut_rows_data()
    summary, tensors = ngram.tiers(model, rows[:4], rows[4:20])
    shares = [summary[f'theta{t:g}']['tokens'] for t in ngram.THETAS]
    assert shares == sorted(shares) and all(0 <= summary[f'theta{t:g}']['energy'] <= 1 for t in ngram.THETAS)
    assert tensors['ngram_loss'].shape == (4, CUT_SEQ - 1) and len(summary['per_site']['theta2']) == 3
    # what loud/gate.py reads from it (configs/e13/gate_theta*.yaml): the one entry named by the prefix
    read = {site: v['energy'] for site, v in summary.items()
            if isinstance(v, dict) and 'energy' in v and site.startswith('theta2')}
    assert list(read) == ['theta2']


def h8_variants_run(scratch):
    """Complete H8 runs of each synthesis-mode experiment through loud/train.py, with their logs."""
    from loud import h8
    cut_module()
    patch_sequences(length=CUT_SEQ, vocab=512)
    for variant, args in (('clip', dict(clip_credit=True, compare_every=5)), ('agree', dict(agree=True)),
                          ('runway', dict(runway=True)), ('budget', dict(budget_theta=6.5))):
        spec = dict(experiment='cpu', variant=variant, seed=0, task='h8', args=dict(
            preset='2m', sequences=72, heldout=32, batch=4, official_eval=False, **args))
        out = Path(scratch) / variant
        result = h8.run(spec, out)
        records = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        assert result['complete'] and len(records) == 10, (variant, len(records))
        cut = [r for r in records if r['phase'] == 'cut']
        if variant == 'clip':
            assert all(-1 <= r['cosine_to_identity'] <= 1 for r in cut if r['step'] % 5 == 0)
        if variant == 'agree':
            assert 0 <= records[-1]['agree_kept'] <= 1
        if variant == 'runway':
            assert 'runway' in records[-1] and len(records[-1]['runway']) == 3
        if variant == 'budget':
            assert all(0 <= r['low_tier'] <= 1 for r in cut) and any(r['low_tier'] > 0 for r in cut)



def h8_arch(scratch):
    """E16's second architecture in H8: the cut graph without detaches is the track-3 GPT's own forward (its hashed
    n-gram value embeddings on every block included); a complete cut-mode Boost -> Cut run."""
    from loud import arch, h8
    m = cut_module()
    patch_sequences(length=CUT_SEQ, vocab=512)
    config = arch.config(2, 128, 512)
    config.vocab_size, config.sequence_len = 512, CUT_SEQ - 1
    model = arch.make_model(torch.device('cpu'), 0, config, 1.5, 0.02).requires_grad_(True)
    arch.module().attention = sys.modules['utils.model'].attention  # its forward's flex attention: dense on a CPU
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.02 * torch.randn(p.shape, generator=torch.Generator().manual_seed(p.numel())))
    rows = cut_rows_data()[:2]
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    ours = h8.cut_forward(model, x, y, held=False)[0]
    assert torch.allclose(ours, model(x, y, loss_reduction='none').view_as(y), atol=1e-6), 'not the track-3 forward'
    loss = h8.cut_forward(model, x, y)[0].sum()
    grads = torch.autograd.grad(loss, [model.ngram_embeds['2'].weight, model.ngram_proj['0'].weight])
    assert all(g.abs().sum() > 0 for g in grads), 'the n-gram path gets no credit'
    spec = dict(experiment='cpu', variant='arch', seed=0, task='h8', args=dict(
        preset='2m', arch='ngram', ngram_rows=512, mode='cut', sequences=72, heldout=32, batch=4, official_eval=False))
    result = h8.run(spec, Path(scratch) / 'arch')
    assert result['complete'] and result['updates'] == 10
    from loud.cut import record_settings  # the n-gram tables at wte's rate (record #5's rule), their projections not
    s = record_settings(m, '2m', 4)
    optimizer = arch.embedding_rates(m.make_optimizer(model, s, 10), model)
    rate = {id(p): g['lr'] for g in optimizer.optimizers[1].param_groups for p in g['params']}
    assert rate[id(model.ngram_embeds['2'].weight)] == rate[id(model.transformer.wte.weight)] == 300 * s.lr
    assert rate[id(model.ngram_proj['0'].weight)] == s.lr
    assert sum(len(g['params']) for g in optimizer.optimizers[1].param_groups) + len(
        optimizer.optimizers[0].param_groups[0]['params']) == len(list(model.parameters()))
    import loud.proxy  # E16's backprop arm: the same model under PR #40's recipe at batch 4
    args = dict(spec['args'], recipe='pr40')
    del args['mode']
    result = loud.proxy.run(dict(spec, variant='arch_bp', task='proxy', args=args), Path(scratch) / 'arch_bp')
    assert result['complete'] and result['updates'] == 10
    # the official scorer builds the record's GPTConfig(): only that shape can be scored
    from loud.options import PRESETS, TRACK3_ROWS, h8_options, proxy_options
    default = arch.module().GPTConfig()
    assert (default.n_layer, default.n_embd, default.ngram_rows) == PRESETS['track3'] + (TRACK3_ROWS,)
    for check, extra in ((h8_options, dict(mode='cut')), (proxy_options, dict(recipe='pr40'))):
        check(dict(preset='track3', arch='ngram', **extra))
        for bad in (dict(preset='2m'), dict(preset='track3', ngram_rows=512)):
            try:
                check(dict(dict(preset='track3', arch='ngram', **extra), **bad))
            except ValueError as e:
                assert 'official scorer' in str(e), e
            else:
                raise AssertionError(f'{check.__name__} accepted {bad} with the official score on')


def h6_synth_vs_real(scratch):
    """H6 item 6 on a tiny model: the real chain's partial-row credit (PR #40's Cut through cut.py's row flags) and
    H8's exact credit read through the same rows and signs agree token by token when every MLP unit is routed (only
    finite differences separate them); the partial rows themselves are far from the exact credit."""
    from loud.diagnostics import Toolkit, site_names
    m, s, model, x, y = record_setup()
    summary, _ = Toolkit(m, model, x[:1], y[:1]).synth_vs_real(rows=32)
    names = site_names(s.model.n_layer)
    for key in ('token rows, all routes', 'time rows, all routes', 'token rows, record routes'):
        assert set(summary[key]) == set(names), summary[key]
    for key in ('token rows, all routes', 'time rows, all routes'):
        for name in names:
            row = summary[key][name]
            assert row['real_vs_synth']['p10'] > 0.99, (key, name, row)
            assert row['synth_vs_exact']['median'] < 0.9, (key, name, row)


def h8_snr(scratch):
    """E8's H8 setting: each sequence's synthesized weight gradient (G_b^T X_b) sums to synthesis's block-local one,
    with every other gradient unchanged; a complete run with SNR scaling on a partial-row scheme steps with the shaped
    optimizer in the Cut phase only, and logs its ratios there."""
    from loud import h8
    from loud.options import H8
    m, s, model, x, y = record_setup()
    model.requires_grad_(True)
    scheme = lambda: h8.Scheme(dict(H8, rows=32), s.model.n_embd, torch.device('cpu'), 0)  # the same draws each time
    plain, _, _ = h8.synthesis_gradients(model, x, y, scheme(), 1)
    each = {}
    summed, _, _ = h8.synthesis_gradients(model, x, y, scheme(), 1, each=each)
    linears = {id(mod.weight) for mod in model.transformer.h.modules() if isinstance(mod, torch.nn.Linear)}
    assert set(each) == linears and all(g.shape[0] == len(x) for g in each.values())
    for k, g in plain.items():
        if k in each:
            scale = g.float().abs().max()
            assert torch.allclose(each[k].sum(0), g.float(), atol=1e-5 * scale, rtol=1e-4)
            assert torch.allclose(summed[k].float(), g.float(), atol=1e-5 * scale, rtol=1e-4)
        else:
            assert torch.equal(summed[k], g)
    patch_sequences(length=CUT_SEQ, vocab=512)
    out = Path(scratch) / 'snr'
    spec = dict(experiment='cpu', variant='snr', seed=0, task='h8', args=dict(
        preset='2m', sequences=72, heldout=32, batch=4, rows=32, snr_scaling='snr', official_eval=False))
    result = h8.run(spec, out)
    records = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and {r['phase'] for r in records} == {'boost', 'cut'}
    assert all(('snr_ratios' in r) == (r['phase'] == 'cut') for r in records), records
    ratios = [v for r in records if 'snr_ratios' in r for v in r['snr_ratios'].values()]
    assert all(0 <= v['r_mean'] <= 1 and 0 <= v['frobenius'] <= 1 for v in ratios), ratios


def h8_knobs(scratch):
    """The follow-up knobs: live cross-token paths (keys and values live is the full graph; held is not), and the rule
    constants a run sets (E10's clip factor and agreement share, E11's grid), with their checks."""
    from loud import h8
    from loud.options import h8_options
    m, s, model, x, y = record_setup()
    model.requires_grad_(True)
    w = model.transformer.h[0].attn.c_v.weight
    grad = lambda **kw: torch.autograd.grad(h8.cut_forward(model, x, y, **kw)[0].sum(), w)[0]
    full = grad(held=False)
    assert torch.allclose(grad(held=True, live=['keys', 'values']), full, atol=1e-6)
    assert not torch.allclose(grad(held=True), full, atol=1e-6)
    assert (grad(held=True, live=['values']) - full).norm() < (grad(held=True) - full).norm()
    o = h8_options(dict(preset='2m', clip_credit=True, clip_factor=1.0))
    h8.H8(o, torch.device('cpu'), 0, 10, torch.zeros(0, CUT_SEQ, dtype=torch.long))
    assert h8.CLIP == 1.0 and h8.RUNWAY == h8.RUNWAY_V2
    o = h8_options(dict(preset='2m', runway=True, runway_grid=[0.1, 0.4]))
    h8.H8(o, torch.device('cpu'), 0, 10, torch.zeros(0, CUT_SEQ, dtype=torch.long))
    assert h8.RUNWAY == (0.1, 0.4) and h8.CLIP == 3.0
    for bad in (dict(live_paths=['keys']), dict(clip_factor=2.0), dict(mode='cut', live_paths=['queries'])):
        try:
            h8_options(dict(preset='2m', **bad))
        except ValueError:
            continue
        raise AssertionError(f'accepted {bad}')


def h6_snr(scratch):
    """E8's snr_spectrum tool: H8's per-sequence synthesized gradients at K = 4 average, over draws, to the exact
    credits' (the scheme is unbiased), one draw being far noisier; the tool measures the training sequences after the
    checkpoint's update (wrapping past the epoch's end) for every estimator, from the checkpoint's saved momentum."""
    from types import SimpleNamespace
    from loud.anvil import is_anvil_matrix
    from loud.diagnostics import snr_tool, synthesized_gradients
    from loud.options import diagnose_options
    m, s, model, x, y = record_setup()
    row = torch.cat([x[:1], y[:1, -1:]], 1)
    exact = synthesized_gradients(model, row, 1, 'time', 1)
    draws = synthesized_gradients(model, row.expand(16, -1), 4, 'token', 1)  # 16 updates, fresh signs each
    flat = lambda g: torch.cat([v.flatten() for v in g.values()])
    cos = lambda a, b: float(a @ b / (a.norm() * b.norm()))
    target = flat({n: g[0] for n, g in exact.items()})
    single = cos(flat({n: g[0] for n, g in draws.items()}), target)
    mean = cos(flat({n: g.mean(0) for n, g in draws.items()}), target)
    assert mean > 0.9 and mean > single + 0.1, (single, mean)
    rows = torch.cat([x, y[:, -1:]], 1).repeat(3, 1)[:5]
    names = [n for n, p in model.named_parameters() if is_anvil_matrix(n, p)]
    shapes = dict((n, p.shape) for n, p in model.named_parameters())
    generator = torch.Generator().manual_seed(3)
    rails = lambda n: dict(fast=torch.randn(shapes[n], generator=generator),
                           slow=torch.randn(shapes[n], generator=generator))
    saved = dict(step=1, fast_weight=0.5, state={n: rails(n) for n in names})
    checkpoint = Path(scratch) / 'ckpt_20.pt'
    torch.save(saved, checkpoint.with_suffix('.optim.pt'))
    o = diagnose_options(dict(tool='snr_spectrum', checkpoint='x', sequences=4, batch=2))
    split = SimpleNamespace(train=rows, order=torch.arange(10, 15))
    summary, tensors = snr_tool(model, split, checkpoint, o)
    assert summary['first_sequence'] == 2 and summary['rows'] == [12, 13, 14, 10], summary['rows']
    assert set(summary['energy_below_1']) == set(o['estimators']) == set(tensors)
    for name in o['estimators']:
        assert all(0 <= v <= 1 for v in summary['energy_below_1'][name].values()), (name, summary[name])
        assert set(tensors[name]) == set(names)


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.manual_seed(0)
    globals()[sys.argv[1]](*sys.argv[2:])
    print('ok')

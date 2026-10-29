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
    full = Scheme(h8_options(dict(preset='2m', rows=128, prior='identity')), W, torch.device('cpu'), 0)
    guess = [torch.randn_like(g)]
    assert torch.allclose(full([g], 0, guess)[0], g, atol=1e-5)


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


if __name__ == '__main__':
    torch.set_num_threads(8)
    torch.manual_seed(0)
    globals()[sys.argv[1]](*sys.argv[2:])
    print('ok')

"""H6: the diagnostic toolkit (plan v2, section 3). Offline measurements on H4 checkpoints, on held-out sequences
(never trained on). Autograd is allowed here: nothing feeds training. Task 'diagnose' runs one tool on one checkpoint
and writes <tool>.json (and <tool>.pt tensors) to its run directory.

Sites are H8's: S0, the raw embedding (wte's output), and S1..S8, block i's mid stream as S(i+1). The exact credit at
a site is the cut graph's (loud/h8.py): what the chain measures there, every key and value held, each token's own
query live. Probes run through H8's reforward (Replay), as the chain runs them.

    cos_to_autograd     an estimator's weight gradient against autograd's (check_relay.py's comparison), cosine per
                        matrix type, block and parameter group: `estimator` identity (PR #26's), h8 (H8 synthesis,
                        identity scheme; also per site, the cut graph's credit against the full graph's) or cut
                        (PR #40's Cut at the record's settings)
    credit_dump         per site: the exact credit, the identity path's guess p (the top-stream credit carried down
                        the residual scalars), their per-token cosine and E7's variance ratio |g - p|^2 / |g|^2
    deep_readout_check  E5: per token, the 512 probes of one signed Hadamard cycle launched at S1 (bf16 forward
                        differences at chain_sigma), read at S2..S8: each site's 512 x 512 system (the probes' changes
                        there, the loss changes), its condition number, and the cosine of its solution to the exact
                        credit (ridge 0, and 1e-3 x mean diag toward the guess as the implementation would use)
    linearity_check     per site and probe size: how far each token's loss response to +v and -v is from linear,
                        float32 and bf16
    h8_validation       H8 synthesis (identity scheme) against PR #40's Cut estimate with its cross-token patches off,
                        no future draws, every MLP unit routed and every own-query key logit rerun: per-matrix cosine
                        (v2's bar: 0.99)
    ngram_tiers         E13: the credit energy and token share of the low tier (max(n-gram loss, model loss) <= theta)
                        on the held-out sequences, the n-gram table built from the training stream up to the checkpoint
                        (loud/ngram.py)
    synth_vs_real       H6 item 6: H8 synthesis against the real chain's (H5's) partial-row estimate with the same rows
                        and signs: per site, the per-token cosine of the two, and of each to the exact credit, at 128
                        rows by token and by time, with the record's MLP routes and with every unit
    snr_spectrum        E8 (H6 item 4; loud/snr.py): per block matrix, the SNR at the update batch along each singular
                        direction of the checkpoint's ANVIL II momentum, and the share of its energy below SNR 1, per
                        estimator of the per-sequence gradients (backprop; H8 synthesis with exact credits; partial
                        rows at K = 4, by time and by token). It measures the training sequences the run would read
                        next (64), not held-out ones: the SNR is over the training distribution

References are float32 (no autocast, 'highest' matmul precision), as in check_relay.py; probes run under bf16 where
the chain does.
"""
from contextlib import contextmanager, nullcontext
import copy
import json

import torch
import torch.nn.functional as F

from . import h8, ngram, paths, refs, snr
from .anvil import is_anvil_matrix
from .data import benchmark_split
from .options import ESTIMATORS, H8, diagnose_options
from .runlog import Run

TYPES = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'attn.ve_gate', 'mlp.c_fc', 'mlp.c_proj')
CHAIN_SIGMA = 0.01  # PR #40's Settings.chain_sigma: the probe step per coordinate, relative to the token's RMS


def site_names(n_layer):
    return [f'S{k}' for k in range(n_layer + 1)]


@contextmanager
def exact():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def groups(names, n_layer):
    block = [n for n in names if n.startswith('transformer.h.') and n.endswith('.weight')]
    out = {'block_matrices': block}
    out.update({kind: [n for n in block if n.endswith(f'.{kind}.weight')] for kind in TYPES})
    out.update({f'block_{i}': [n for n in block if n.startswith(f'transformer.h.{i}.')] for i in range(n_layer)})
    out.update(wte=['transformer.wte.weight'], value_embeds=[n for n in names if n.startswith('value_embeds.')],
               lm_head=['lm_head.weight'], resid_lambdas=['resid_lambdas'], x0_lambdas=['x0_lambdas'], all=list(names))
    return out


def pending_update(g, rails, saved):
    """The tensor ANVIL II's next step orthogonalizes, for gradient g: its rails stepped with g (fast at the saved
    fast_beta, slow at slow_beta), blended by the fast weight and mixed with g at the rail beta, as Anvil.step does.
    `saved` is a checkpoint's .optim.pt (loud/checkpoints.py), `rails` its state for this matrix."""
    fast = rails['fast'].to(g).lerp(g, 1 - saved['fast_beta'])
    slow = rails['slow'].to(g).lerp(g, 1 - saved['slow_beta'])
    weight = saved['fast_weight']
    return g.lerp(weight * fast + (1 - weight) * slow, saved['beta'])


def per_token_cosine(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def quantiles(t):
    t = t.float().flatten()
    return {q: round(t.quantile(v).item(), 5) for q, v in (('p10', 0.1), ('median', 0.5), ('p90', 0.9))}


def cosine(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return round((a @ b / (a.norm() * b.norm()).clamp_min(1e-30)).item(), 5)


class Toolkit:
    def __init__(self, m, model, x, y):
        self.m, self.model, self.x, self.y = m, model, x, y
        self.n_layer = model.config.n_layer

    def exact_credits(self):
        """A float32 copy of the model, the clean sites, and the cut graph's per-token credit of the summed loss at the
        nine sites and the top stream."""
        model = copy.deepcopy(self.model).float().requires_grad_(True)
        with exact():
            losses, sites, top, _, _ = h8.cut_forward(model, self.x, self.y)
            grads = torch.autograd.grad(losses.sum(), sites + [top])
        return model.requires_grad_(False), [s.detach() for s in sites], list(grads[:-1]), grads[-1]

    def cos_to_autograd(self, estimator='identity'):
        """An estimator's weight gradient on this sequence against autograd of the summed loss, both on the checkpoint
        (identity: PR #26's FastRelay at its main() defaults; h8: H8 synthesis with the identity scheme, plus per site
        the per-token cosine of the cut graph's credit to the full graph's; cut: PR #40's Cut, the record's
        settings)."""
        m, summary = self.m, {}
        ref = copy.deepcopy(self.model).float().requires_grad_(True)
        with exact():
            ref(self.x, self.y, loss_reduction='sum').backward()
        reference = {n: (torch.zeros_like(p) if p.grad is None else p.grad).float() for n, p in ref.named_parameters()}
        if estimator == 'identity':
            hidden = 4 * self.model.config.n_embd
            settings = m.Settings(fc_k=hidden // 8, route_k=hidden // 4, fast_mlp_mode='all', no_chain=True,
                                  value_path=False, skip_mid=True)  # PR #26's main() defaults
            relay = m.FastRelay(self.model, settings, 0, 0, 1)
            try:
                estimate = relay.estimate(self.x, self.y)
            finally:
                relay.close()
        elif estimator == 'h8':
            from .options import H8
            model = copy.deepcopy(self.model).float().requires_grad_(True)
            scheme = h8.Scheme(dict(H8), model.config.n_embd, self.x.device, 0)
            with exact():
                synth, _, _ = h8.synthesis_gradients(model, self.x, self.y, scheme, 1)
                estimate = {n: synth[id(p)] for n, p in model.named_parameters() if id(p) in synth}
                full = h8.cut_forward(model, self.x, self.y, held=False)
                cut = h8.cut_forward(model, self.x, self.y)
                full_credit = torch.autograd.grad(full[0].sum(), full[1])
                cut_credit = torch.autograd.grad(cut[0].sum(), cut[1])
            summary['sites'] = {name: quantiles(per_token_cosine(c[0], f[0]))
                                for name, c, f in zip(site_names(self.n_layer), cut_credit, full_credit)}
        elif estimator == 'cut':
            from .cut import record_settings
            from .options import PRESETS
            config = self.model.config
            preset = next(p for p, shape in PRESETS.items() if shape == (config.n_layer, config.n_embd))
            estimate = m.Cut(self.model, record_settings(m, preset, 4), 0, 0, 1, 1, None).estimate(self.x, self.y)
        else:
            raise ValueError(f'unknown estimator {estimator!r}')
        flat = lambda d, ns: torch.cat([(d[n] if d.get(n) is not None else torch.zeros_like(reference[n]))
                                        .flatten().float() for n in ns])
        summary.update({g: cosine(flat(estimate, ns), flat(reference, ns))
                        for g, ns in groups(list(reference), self.n_layer).items() if ns})
        return summary, {}

    def credit_dump(self):
        model, sites, credits, g_top = self.exact_credits()
        guess = h8.identity_guess(model, sites, g_top)
        summary, tensors = {}, {}
        for name, g, p in zip(site_names(self.n_layer), credits, guess):
            g, p = g[0].float(), p[0].float()
            residual = (g - p).square().sum(-1)
            summary[name] = dict(cosine=quantiles(per_token_cosine(g, p)),
                                 variance_ratio=round((residual.sum() / g.square().sum()).item(), 5),
                                 variance_ratio_per_token=quantiles(residual / g.square().sum(-1).clamp_min(1e-30)))
            tensors[name] = dict(exact=g.bfloat16().cpu(), guess=p.bfloat16().cpu())
        return summary, tensors

    def deep_readout_check(self, sigma=CHAIN_SIGMA, ridge=1e-3, chunk=4, precision='bf16'):
        """E5's diagnostic (module docstring). S0 keeps its own cycle in E5 (its probes also move x0, which enters
        every block), so the launch is at S1, whose own row is the probe itself."""
        model, sites, credits, g_top = self.exact_credits()
        guess = h8.identity_guess(model, sites, g_top)
        width, tokens = sites[1].shape[-1], sites[1].shape[1]
        signs = torch.randint(0, 2, (width,), generator=torch.Generator().manual_seed(0)).to(sites[1]) * 2 - 1
        basis = h8.hadamard(width, sites[1].device) * signs
        names = site_names(self.n_layer)[2:]
        cos = {n: {r: [] for r in (0.0, ridge)} for n in names}
        condition = {n: [] for n in names}
        # bf16 as the chain runs its products; float32 to separate the method from the replay's rounding: the replay
        # differences whole bf16 losses, which a 1% probe's change sinks under (linearity_check), where the chain's
        # kernels difference the changes themselves
        bf16 = lambda: (torch.autocast('cuda', dtype=torch.bfloat16) if sites[1].is_cuda and precision == 'bf16' else
                        exact() if precision == 'float32' else nullcontext())
        with bf16():
            replay = h8.Replay(model, self.x, self.y, chunk=chunk)
            base = replay(1, torch.zeros_like(sites[1]).unsqueeze(2))[..., 0]
        eye = torch.eye(width, device=basis.device)
        for t0 in range(0, tokens, chunk):
            sl = slice(t0, t0 + chunk)
            step = sigma * sites[1][:, sl].square().mean(-1, keepdim=True).sqrt()  # [1, c, 1]
            with bf16():
                moved, at = replay(1, step.unsqueeze(-1) * basis, sites=True, start=t0)  # the cycle: [1, c, W, W]
            d = (moved - base[:, sl, None]).float()[0].unsqueeze(-1)  # [c, W probes, 1]
            with exact():
                for k, name in enumerate(names, start=2):
                    A = at[k][0].float()  # [c, W probes, W]: each probe's change at S_k
                    U, S, Vh = torch.linalg.svd(A)
                    condition[name].append(S[:, 0] / S[:, -1].clamp_min(1e-30))
                    target = credits[k][0, sl].float()
                    plain = Vh.mT @ ((U.mT @ d) / S[..., None].clamp_min(1e-30))
                    cos[name][0.0].append(per_token_cosine(plain[..., 0], target))
                    lam = ridge * S.square().mean(-1)[:, None, None]
                    ridged = torch.linalg.solve(A.mT @ A + lam * eye, A.mT @ d + lam * guess[k][0, sl, :, None].float())
                    cos[name][ridge].append(per_token_cosine(ridged[..., 0], target))
        summary = {n: dict(condition=quantiles(torch.cat(condition[n])),
                           **{f'cosine ridge {r}': quantiles(torch.cat(v)) for r, v in cos[n].items()})
                   for n in names}
        summary['settings'] = dict(sigma=sigma, ridge=ridge, launch='S1', probes=width, tokens=tokens,
                                   precision=precision)
        return summary, {}

    def linearity_check(self, sigmas=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1), draws=2):
        """Per site and probe size: each token's loss under +v and -v (|v| = sigma |activation|, a random direction),
        through reforward. Quantiles over tokens and draws of |L+ + L- - 2 L0| / |L+ - L-| (0 when linear) and of
        |L+ - L-| (the signal), float32 and bf16."""
        model, sites, _, _ = self.exact_credits()
        generator = torch.Generator().manual_seed(0)
        summary = {}
        for precision in ('float32', 'bf16'):
            context = exact() if precision == 'float32' else (
                torch.autocast('cuda', dtype=torch.bfloat16) if sites[0].is_cuda else nullcontext())
            with context:
                replay = h8.Replay(model, self.x, self.y)
                for k, name in enumerate(site_names(self.n_layer)):
                    act = sites[k]
                    base = replay(k, torch.zeros_like(act).unsqueeze(2))[..., 0].double()
                    row = summary.setdefault(name, {})
                    for sigma in sigmas:
                        ratios, signals = [], []
                        for _ in range(draws):
                            v = torch.randn(act.shape, generator=generator).to(act)
                            v = v * (sigma * act.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True))
                            pm = replay(k, torch.stack([v, -v], 2)).double()
                            up, down = pm[..., 0], pm[..., 1]
                            ratios.append((up + down - 2 * base).abs() / (up - down).abs().clamp_min(1e-30))
                            signals.append((up - down).abs())
                        row[f'{precision} {sigma}'] = dict(nonlinearity=quantiles(torch.cat(ratios)),
                                                           signal=quantiles(torch.cat(signals)))
        return summary, {}

    def h8_validation(self, precision='bf16'):
        """H8's synthesis mode (identity scheme) against PR #40's Cut estimate on this checkpoint, the record's
        cross-token patches off, no future draws, every MLP unit routed (v2 H8, 'Validation before use'). precision
        'float32' runs both sides without bf16 (the record's chain_precision fp32, its capture in float32), to tell
        the record's own rounding from a difference in what H8 computes."""
        from .cut import record_settings
        from .options import H8, PRESETS
        m, config = self.m, self.model.config
        preset = next(p for p, shape in PRESETS.items() if shape == (config.n_layer, config.n_embd))
        s = record_settings(m, preset, 4)
        s.value_path = s.path_scalars = s.gate_path = False
        s.key_path_draws, s.future_draws, s.route_k = 0, 0, 4 * config.n_embd
        s.query_keys = 0  # every key logit of the own query, as H8's live query has (the record caps them at 256)
        exact32 = precision == 'float32'
        if exact32:
            s.chain_precision = 'fp32'
        model = self.model.requires_grad_(True)
        params = dict(model.named_parameters())
        scheme = h8.Scheme(dict(H8), config.n_embd, self.x.device, 0)  # the defaults: the identity scheme
        with torch.autocast(self.x.device.type, dtype=torch.bfloat16, enabled=self.x.is_cuda and not exact32):
            synth, _, _ = h8.synthesis_gradients(model, self.x, self.y, scheme, 1)
        model.requires_grad_(False)
        cut = m.Cut(model, s, 0, 0, 1, 1, None)
        if exact32:
            cut.capture.dtype = torch.float32
        record = cut.estimate(self.x, self.y)
        found = {n: cosine(synth[id(p)], record[n]) for n, p in params.items() if record[n].any()}
        return dict(cosines=found, below_099=sorted(n for n, c in found.items() if c < 0.99), precision=precision), {}

    def synth_vs_real(self, rows=128):
        """H6 item 6: H8 synthesis against the real chain's partial-row estimate with the same rows and signs. PR #40's
        Cut (cut.py's, with `rows` rows per token, both assignments) runs on the checkpoint and its decoded credit at
        each site is kept with the site's signed basis and each token's rows; H8's synthesized credit is the exact
        credit read through those same rows, (1/m) sum_j (v_j . g) v_j. Per site, quantiles over tokens of the cosine
        of the two (they differ by finite differences, bf16 and route truncation), and of each to the exact credit;
        with the record's routes and with every MLP unit routed, to separate the truncation."""
        from .cut import estimator_classes, record_settings
        from .options import PRESETS
        m, config = self.m, self.model.config
        preset = next(p for p, shape in PRESETS.items() if shape == (config.n_layer, config.n_embd))
        _, Cut = estimator_classes(m)
        seen = {}

        class Kept(Cut):
            def basis(self):
                H = super().basis()
                seen.setdefault('bases', []).append(H)
                return H

            def partial_chain_credits(self, e, *rest):
                seen['bases'] = []
                credits = super().partial_chain_credits(e, *rest)
                seen.update(credits=credits, index=self.row_index(e.shape[0]))
                return credits

        _, _, exact_credits, _ = self.exact_credits()
        exact_credits = [g[0].float() for g in exact_credits]
        names = site_names(self.n_layer)
        summary = dict(settings=dict(rows=rows, tokens=self.x.shape[1]))
        for assign in ('token', 'time'):
            for routes in ('record', 'all'):
                s = record_settings(m, preset, 4)
                if routes == 'all':  # every MLP unit and every own-query key logit, as H8 has them
                    s.route_k, s.query_keys = 4 * config.n_embd, 0
                estimator = Kept(self.model, s, 0, 0, 1, 1, None)
                estimator.rows, estimator.start = rows, (0 if assign == 'time' else None)
                estimator.estimate(self.x, self.y)
                index = seen['index']
                row = summary.setdefault(f'{assign} rows, {routes} routes', {})
                for k, (name, g, H) in enumerate(zip(names, exact_credits, seen['bases'])):
                    real = seen['credits']['emb' if k == 0 else k - 1].float()
                    with exact():
                        directions = H[index].float()  # [tokens, rows, width]
                        synth = torch.einsum('tp,tpw->tw', torch.einsum('tw,tpw->tp', g, directions), directions) / rows
                    row[name] = dict(real_vs_synth=quantiles(per_token_cosine(real, synth)),
                                     synth_vs_exact=quantiles(per_token_cosine(synth, g)),
                                     real_vs_exact=quantiles(per_token_cosine(real, g)))
        return summary, {}

def load(checkpoint, device, key):
    """The fixed model with a checkpoint's weights, in float32; the shape comes from the state dict (the benchmark's
    8 x 512, or a smoke test's tiny model)."""
    m = refs.trainer_module(key)
    um = h8.um()
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    n_layer, width = len(state['resid_lambdas']), state['transformer.wte.weight'].shape[1]
    config = um.GPTConfig(n_layer=n_layer, n_head=max(1, width // 64), n_kv_head=max(1, width // 64), n_embd=width)
    with torch.device('meta'):
        model = um.GPT(config)
    model.to_empty(device=device)
    model.load_state_dict({k: v.to(device).float() for k, v in state.items()}, strict=True)
    model.cos, model.sin = model._precompute_rotary(model.rotary_seq_len, width // config.n_head)  # not in the state
    return m, model.requires_grad_(False)


def synthesized_gradients(model, rows, k, assign, batch):
    """Per sequence, H8 synthesis's gradient of its mean loss for ANVIL II's matrices ({name: [S, rows, cols]}): the
    cut graph's credits read through a Scheme of width / k rows per token, assigned by `assign` (k = 1: the exact
    credits). Sequence s is read as part of update s // batch, so the rows a 'time' scheme reads rotate across
    updates as in training and are shared within one."""
    model = copy.deepcopy(model).float().requires_grad_(True)
    named = [(n, p) for n, p in model.named_parameters() if is_anvil_matrix(n, p)]
    width = model.config.n_embd
    scheme = h8.Scheme(dict(H8, rows=width // k, row_assign=assign), width, rows.device, 0)
    out = {n: [] for n, _ in named}
    for s, row in enumerate(rows.split(1)):
        grads, _, _ = h8.synthesis_gradients(model, row[:, :-1].contiguous(), row[:, 1:].contiguous(), scheme,
                                             s // batch)
        for n, p in named:
            out[n].append(grads[id(p)].float())
    return {n: torch.stack(gs) for n, gs in out.items()}


def snr_tool(model, split, checkpoint, o):
    """E8's diagnostic: snr.snr_spectrum for each estimator on the same sequences, the `sequences` training sequences
    the run would read after the checkpoint's update (from the epoch's start again past its end, as at 100%), with
    the momentum from the checkpoint's .optim.pt (e1.bp_b8 keeps it)."""
    saved = torch.load(checkpoint.with_suffix('.optim.pt'), map_location='cpu', weights_only=True)
    first = saved['step'] * o['batch']
    picks = (torch.arange(o['sequences']) + first) % len(split.train)
    device = next(model.parameters()).device
    rows = split.train[picks].to(device)
    summary = dict(step=saved['step'], first_sequence=first, rows=split.order[picks].tolist(), energy_below_1={})
    tensors = {}
    for name in o['estimators']:
        spec = ESTIMATORS[name]
        gradients = snr.exact_gradients if spec is None else \
            (lambda mdl, r, spec=spec: synthesized_gradients(mdl, r, *spec, o['batch']))
        with exact():
            summary[name], tensors[name] = snr.snr_spectrum(model, rows, saved, o['batch'], gradients)
        summary['energy_below_1'][name] = summary[name]['energy_below_1']  # per matrix type, side by side
    return summary, tensors


def checkpoint_path(ref):
    """'<job id>:<file>' (a finished run's checkpoint) or a path relative to the repo."""
    if ':' in ref:
        jid, name = ref.split(':', 1)
        experiment, variant, seed = jid.rsplit('.', 2)
        return paths.run_dir(experiment, variant, int(seed[1:])) / name
    return paths.REPO / ref


def run_task(spec, run_dir):
    o = diagnose_options(spec['args'])
    device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
    checkpoint = checkpoint_path(o['checkpoint'])
    # the reference trainer: PR #26's for its identity path, else PR #40's (both import the fixed utils/model.py, and
    # one process may load only one)
    identity = o['tool'] == 'cos_to_autograd' and o['estimator'] == 'identity'
    m, model = load(checkpoint, device, 'pr26' if identity else 'pr40')
    split = benchmark_split('10M', heldout=32)
    rows = split.heldout[o['row']:o['row'] + 1].to(device)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()
    run = Run(run_dir, spec, dict(options=o, checkpoint=str(checkpoint),
                                  heldout_rows=None if o['tool'] == 'snr_spectrum' else
                                  split.describe()['heldout_rows'][o['row']:o['row'] + 1]))
    if o['tool'] == 'ngram_tiers':  # E13's diagnostic: the 32 held-out sequences, the stream up to the checkpoint
        summary, tensors = ngram.tiers_tool(model, split.heldout.to(device), checkpoint)
    elif o['tool'] == 'snr_spectrum':  # E8's: training sequences after the checkpoint
        summary, tensors = snr_tool(model, split, checkpoint, o)
    else:
        kwargs = (dict(estimator=o['estimator']) if o['tool'] == 'cos_to_autograd' else
                  dict(precision=o['precision']) if o['tool'] in ('h8_validation', 'deep_readout_check') else {})
        summary, tensors = getattr(Toolkit(m, model, x, y), o['tool'])(**kwargs)
    (run_dir / f'{o["tool"]}.json').write_text(json.dumps(summary, indent=1) + '\n')
    if tensors:
        torch.save(tensors, run_dir / f'{o["tool"]}.pt')
    result = dict(tool=o['tool'], complete=True, summary_file=f'{o["tool"]}.json')
    if o['tool'] == 'h8_validation':  # numbers other jobs can gate on (expect: {min_cosine: {min: 0.98}, ...})
        cosines = list(summary['cosines'].values())
        result['min_cosine'] = min(cosines)
        result['share_099'] = sum(c >= 0.99 for c in cosines) / len(cosines)
    return run.finish(result, spec.get('expect'))

"""H6: the diagnostic toolkit. Offline measurements on H4 checkpoints, on held-out sequences (never trained on).
Autograd is allowed here: nothing feeds training. Task 'diagnose' runs one tool on one checkpoint and writes
<tool>.json (and <tool>.pt tensors) to its run directory.

Sites are the plan's probing sites: 'emb', the raw token embedding (the output of wte), and 'mid<i>', block i's
stream after attention and before the MLP (its input plus the attention output).

    cos_to_autograd   the identity-path estimate of the weight gradient against autograd's: cosine over all block
                      matrices, per matrix type, per block and per other parameter group (check_relay.py's comparison)
    credit_dump       per site: the true per-token credit (autograd), the identity path's guess and their per-token
                      cosine. The guess at mid<i> is the top-stream credit carried down the residual scalars; at emb,
                      PR #26's RMS-norm cycle of it
    token_map_svd     per site: singular values of each token's map from its raw embedding to the site, with the
                      attention outputs frozen at their clean values (torch.func.jacrev per token)
    direction_stats   per block matrix: the top singular directions of the pending update (from the optimizer state
                      the checkpoint saved with checkpoint_optimizer; else of the batch gradient), the curvature along
                      each (a central difference of the batch gradient) and the across-sequence variance along it
    linearity_check   per site, probe size and token: how far the token's own loss response to +v and -v is from
                      linear, with attention frozen, in float32 and in bf16

References are float32 throughout (no autocast, 'highest' matmul precision), as in check_relay.py; the estimator
runs as in training (bf16 autocast).
"""
from contextlib import contextmanager
import copy
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import paths, refs
from .data import benchmark_split
from .options import diagnose_options
from .runlog import Run

TYPES = ('attn.c_q', 'attn.c_k', 'attn.c_v', 'attn.c_proj', 'attn.ve_gate', 'mlp.c_fc', 'mlp.c_proj')


def site_names(n_layer):
    return ['emb'] + [f'mid{i}' for i in range(n_layer)]


@contextmanager
def sites(model, record=None, add=None):
    """Forward hooks at the sites: `record[site]` receives the activation (attention output for mid<i>, whose
    gradient is the site's credit) and `add[site]` (a tensor) is added to it."""
    def hook(name):
        def fn(module, args, out):
            if add and name in add:
                out = out + add[name]
            if record is not None:
                record[name] = out
            return out
        return fn
    handles = [model.transformer.wte.register_forward_hook(hook('emb'))]
    handles += [b.attn.register_forward_hook(hook(f'mid{i}')) for i, b in enumerate(model.transformer.h)]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def stream_values(model, x):
    """Each site's activation for one forward: the raw embedding, and block input + attention output."""
    inputs, outputs = {}, {}
    handles = [b.register_forward_pre_hook(lambda mod, args, i=i: inputs.__setitem__(i, args[0]))
               for i, b in enumerate(model.transformer.h)]
    try:
        with torch.no_grad(), sites(model, record=outputs):
            model(x)
    finally:
        for h in handles:
            h.remove()
    return {'emb': outputs['emb'], **{f'mid{i}': inputs[i] + outputs[f'mid{i}'] for i in inputs}}, outputs


@contextmanager
def exact():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision('highest')
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


def reference_gradients(model, x, y):
    """Autograd of the summed loss in float32: weight gradients and each site's per-token credit."""
    ref = copy.deepcopy(model).float().requires_grad_(True)
    record = {}
    with exact(), sites(ref, record=record):
        loss = ref(x, y, loss_reduction='sum')
        for value in record.values():
            value.retain_grad()
        loss.backward()
    grads = {n: (torch.zeros_like(p) if p.grad is None else p.grad).float() for n, p in ref.named_parameters()}
    credits = {s: v.grad.float()[0] for s, v in record.items()}
    return grads, credits


def groups(names, n_layer):
    block = [n for n in names if n.startswith('transformer.h.') and n.endswith('.weight')]
    out = {'block_matrices': block}
    out.update({kind: [n for n in block if n.endswith(f'.{kind}.weight')] for kind in TYPES})
    out.update({f'block_{i}': [n for n in block if n.startswith(f'transformer.h.{i}.')] for i in range(n_layer)})
    out.update(wte=['transformer.wte.weight'], value_embeds=[n for n in names if n.startswith('value_embeds.')],
               lm_head=['lm_head.weight'], resid_lambdas=['resid_lambdas'], x0_lambdas=['x0_lambdas'], all=list(names))
    return out


def per_token_cosine(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def quantiles(t):
    t = t.float().flatten()
    return {q: round(t.quantile(v).item(), 5) for q, v in (('p10', 0.1), ('median', 0.5), ('p90', 0.9))}


class Toolkit:
    def __init__(self, m, model, settings, x, y):
        self.m, self.model, self.settings, self.x, self.y = m, model, settings, x, y
        self.n_layer = model.config.n_layer

    def estimate(self, keep=None):
        """The identity-path estimate on this sequence, as in training; `keep` receives the top-stream credit and
        the embedding credit."""
        m = self.m
        estimator = m.FastRelay(self.model, self.settings, 0, 0, 1)
        probe, norm_cycle = estimator.probe, m.fast_norm_cycle
        if keep is not None:
            estimator.probe = lambda *a, **k: keep.setdefault('top', probe(*a, **k))
            m.fast_norm_cycle = lambda *a, **k: keep.setdefault('emb', norm_cycle(*a, **k))
        try:
            return estimator.estimate(self.x, self.y)
        finally:
            m.fast_norm_cycle = norm_cycle
            estimator.close()

    def cos_to_autograd(self):
        reference, _ = reference_gradients(self.model, self.x, self.y)
        estimate = self.estimate()
        flat = lambda d, ns: torch.cat([d[n].flatten().float() for n in ns])
        return {g: round(F.cosine_similarity(flat(estimate, ns), flat(reference, ns), dim=0).item(), 5)
                for g, ns in groups(list(reference), self.n_layer).items() if ns}, {}

    def credit_dump(self):
        _, true = reference_gradients(self.model, self.x, self.y)
        keep = {}
        self.estimate(keep)
        lam = self.model.resid_lambdas.float()
        guess, carried = {'emb': keep['emb']}, keep['top']
        for i in reversed(range(self.n_layer)):  # PR #26: d_out[L-1] = top credit, d_out[i] = lam[i+1] * d_out[i+1]
            guess[f'mid{i}'] = carried
            carried = lam[i] * carried
        summary, tensors = {}, {}
        for site in site_names(self.n_layer):
            cos = per_token_cosine(true[site], guess[site])
            summary[site] = quantiles(cos)
            tensors[site] = dict(true=true[site].bfloat16().cpu(), guess=guess[site].bfloat16().cpu(), cosine=cos.cpu())
        return summary, tensors

    def frozen(self, model):
        """Per token: each site's activation, the attention outputs [T, L, W], x0 = norm(raw embedding) and targets."""
        with exact():
            values, attn = stream_values(model, self.x)
        frozen = torch.stack([attn[f'mid{i}'][0] for i in range(self.n_layer)], 1)
        return values, frozen, self.m.norm(values['emb'][0]), self.y[0]

    def token_map_svd(self, chunk=32):
        model, norm = copy.deepcopy(self.model).float(), self.m.norm
        values, frozen, _, _ = self.frozen(model)
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()

        def to_sites(e, a):  # one token: raw embedding -> each block's mid stream, attention outputs frozen
            x0 = norm(e)
            x, out = x0, []
            for i, block in enumerate(model.transformer.h):
                mid = lam_r[i] * x + lam_0[i] * x0 + a[i]
                out.append(mid)
                x = mid + block.mlp(norm(mid))
            return torch.stack(out)

        jac = torch.func.vmap(torch.func.jacfwd(to_sites))  # forward mode: 512 inputs against L x 512 outputs
        singular = []
        with exact(), torch.no_grad():
            for t0 in range(0, values['emb'].shape[1], chunk):
                J = jac(values['emb'][0, t0:t0 + chunk].float(), frozen[t0:t0 + chunk])  # [t, L, W, W]
                singular.append(torch.linalg.svdvals(J))
        s = torch.cat(singular)  # [T, L, W], descending
        summary = {}
        for k in range(self.n_layer):
            sk = s[:, k]
            summary[f'mid{k}'] = dict(condition=quantiles(sk[:, 0] / sk[:, -1].clamp_min(1e-30)),
                                      condition_without_radial=quantiles(sk[:, 0] / sk[:, -2].clamp_min(1e-30)),
                                      largest=quantiles(sk[:, 0]), smallest=quantiles(sk[:, -1]))
        return summary, dict(singular_values=s.half().cpu())

    def direction_stats(self, rows, optimizer_state=None, directions=16, step=1e-3):
        """rows: [B >= 2, SEQ] sequences. Per block matrix, the top singular directions D (unit Frobenius norm) of the
        pending update before ANVIL II orthogonalizes it, g.lerp(w * fast + (1 - w) * slow, beta) with g the batch
        gradient, when the checkpoint saved the optimizer state; else of g alone. For each D: sigma (its singular
        value), curvature h = <D, (g(W + eD) - g(W - eD)) / 2e>, noise c = var_i <g_i, D> over the B sequences, and
        h^2 / c. The curvature is a central difference of autograd gradients: flex attention has no double backward."""
        if len(rows) < 2:
            raise ValueError('direction_stats needs at least two sequences for the noise')
        model = copy.deepcopy(self.model).float().requires_grad_(True)
        params = dict(model.named_parameters())
        names = [n for n, p in params.items() if n.startswith('transformer.h.') and p.ndim == 2]

        def loss(batch):
            with exact():
                return model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')

        per_sequence = []
        for i in range(len(rows)):
            grads = torch.autograd.grad(loss(rows[i:i + 1]), [params[n] for n in names])
            per_sequence.append(dict(zip(names, grads)))
        mean = {n: sum(g[n] for g in per_sequence) / len(rows) for n in names}
        source = 'gradient'
        if optimizer_state is not None:
            state, (beta, weight) = optimizer_state['state'], (optimizer_state['beta'], optimizer_state['fast_weight'])
            if all(n in state and {'fast', 'slow'} <= set(state[n]) for n in names):
                source = 'pending update'
        summary = dict(directions_from=source, sequences=len(rows), step_relative=step)
        for n in names:
            target = mean[n]
            if source == 'pending update':
                fast, slow = (state[n][k].to(target) for k in ('fast', 'slow'))
                target = target.lerp(weight * fast + (1 - weight) * slow, beta)
            u, sv, vh = torch.linalg.svd(target, full_matrices=False)
            w, e = params[n].detach().clone(), step * max(params[n].detach().norm().item(), 1.0)  # a zero matrix too
            out = []
            for k in range(min(directions, len(sv))):
                d = torch.outer(u[:, k], vh[k])
                sides = []
                for sign in (1, -1):
                    with torch.no_grad():
                        params[n].copy_(w + sign * e * d)
                    sides.append(torch.autograd.grad(loss(rows), params[n])[0] / len(rows))
                with torch.no_grad():
                    params[n].copy_(w)
                h = ((sides[0] - sides[1]) * d).sum().item() / (2 * e)
                c = torch.stack([(g[n] * d).sum() for g in per_sequence]).var().item()
                out.append(dict(sigma=sv[k].item(), curvature=h, noise=c, h2_over_c=h * h / c if c > 0 else None))
            summary[n] = out
        return summary, {}

    def linearity_check(self, sigmas=(1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1), draws=2):
        """Per token, as a Cut probe sees it: the token's own loss as a function of its activation at the site, with
        every attention output frozen at its clean value (so tokens do not interact), under +v and -v of norm
        sigma |activation|. Reports, over tokens and draws, quantiles of |L+ + L- - 2 L0| / |L+ - L-| (0 for a linear
        response) and of |L+ - L-| (the signal), in float32 and under the bf16 autocast the probes run in."""
        model, norm = copy.deepcopy(self.model).float(), self.m.norm
        values, frozen, x0_clean, targets = self.frozen(model)
        blocks, vocab = model.transformer.h, model.config.vocab_size
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()

        def token_loss(site, h, a, x0, y):
            if site == 'emb':
                x0 = norm(h)
                x, first = x0, 0
            else:
                k = int(site[3:])
                x, first = h + blocks[k].mlp(norm(h)), k + 1
            for i in range(first, self.n_layer):
                mid = lam_r[i] * x + lam_0[i] * x0 + a[i]
                x = mid + blocks[i].mlp(norm(mid))
            logits = 15 * torch.tanh(model.lm_head(norm(x))[:vocab].float() / 15)
            return -torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)

        generator = torch.Generator(self.x.device).manual_seed(0)
        summary = {}
        for site in site_names(self.n_layer):
            act = values[site][0]
            losses = torch.func.vmap(lambda h, a, x0, y: token_loss(site, h, a, x0, y))
            row = {}
            for precision in ('float32', 'bf16'):
                context = exact() if precision == 'float32' else torch.autocast(act.device.type, dtype=torch.bfloat16)
                with torch.no_grad(), context:
                    base = losses(act, frozen, x0_clean, targets).double()
                    for sigma in sigmas:
                        ratios, signals = [], []
                        for _ in range(draws):
                            v = torch.randn(act.shape, device=act.device, generator=generator)
                            v = v * (sigma * act.norm(dim=-1, keepdim=True) / v.norm(dim=-1, keepdim=True))
                            up = losses(act + v, frozen, x0_clean, targets).double()
                            down = losses(act - v, frozen, x0_clean, targets).double()
                            ratios.append((up + down - 2 * base).abs() / (up - down).abs().clamp_min(1e-30))
                            signals.append((up - down).abs())
                        row[f'{precision} {sigma}'] = dict(nonlinearity=quantiles(torch.cat(ratios)),
                                                           signal=quantiles(torch.cat(signals)))
            summary[site] = row
        return summary, {}


def load(checkpoint, device):
    """The fixed model with a checkpoint's weights (from PR #26's checkout, whose utils.model is the fixed model);
    the shape comes from the state dict (the benchmark's 8 x 512, or a smoke test's tiny model)."""
    m = refs.trainer_module('pr26')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    n_layer, width = len(state['resid_lambdas']), state['transformer.wte.weight'].shape[1]
    config = m.GPTConfig(sequence_len=m.SEQ, n_layer=n_layer, n_head=max(1, width // 64),
                         n_kv_head=max(1, width // 64), n_embd=width)
    model = m.make_model(device, 0, config)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    hidden = 4 * config.n_embd
    settings = m.Settings(fc_k=hidden // 8, route_k=hidden // 4, fast_mlp_mode='all', no_chain=True,
                          value_path=False, skip_mid=True)  # PR #26's main() defaults
    return m, model, settings


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
    m, model, settings = load(checkpoint, device)
    split = benchmark_split('10M', heldout=32)
    rows = split.heldout[o['row']:o['row'] + o['batch']].to(device)
    x, y = rows[:1, :-1].contiguous(), rows[:1, 1:].contiguous()
    run = Run(run_dir, spec, dict(options=o, checkpoint=str(checkpoint), heldout_rows=split.describe()['heldout_rows'][
        o['row']:o['row'] + o['batch']]))
    tools = Toolkit(m, model, settings, x, y)
    if o['tool'] == 'direction_stats':
        optim = checkpoint.with_suffix('.optim.pt')
        state = torch.load(optim, map_location=device, weights_only=True) if optim.exists() else None
        summary, tensors = tools.direction_stats(rows, state)
    else:
        summary, tensors = getattr(tools, o['tool'])()
    (run_dir / f'{o["tool"]}.json').write_text(json.dumps(summary, indent=1) + '\n')
    if tensors:
        torch.save(tensors, run_dir / f'{o["tool"]}.pt')
    return run.finish(dict(tool=o['tool'], complete=True, summary_file=f'{o["tool"]}.json'), spec.get('expect'))

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
                      attention outputs frozen at their clean values (torch.func.jacfwd per token)
    direction_stats   per ANVIL II matrix: the top singular directions of the pending update (from the optimizer state
                      the checkpoint saved with checkpoint_optimizer; else of the batch gradient), the curvature along
                      each (a central difference of the batch gradient) and the across-sequence variance along it
    linearity_check   per site, probe size and token: how far the token's own loss response to +v and -v is from
                      linear, with attention frozen, in float32 and in bf16
    credit_recovery   E5: one cycle of probes launched at the embedding, read at every site; per site and token, the
                      credit solved from the probes' changes there against the credit a direct cycle there measures

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
from .anvil import is_anvil_matrix
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


def hadamard(d, device):
    """The Sylvester Hadamard matrix of order d (a power of two), entries +-1, as PR #40 builds it."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def norm_difference(x, delta):
    """norm(x + delta) - norm(x) without the cancellation, as PR #40's chain forms it: x [t, 1, W], delta [t, P, W]."""
    width, eps = x.shape[-1], torch.finfo(torch.float32).eps
    sq = x.square().sum(-1, keepdim=True)
    inv0 = torch.rsqrt(sq / width + eps)
    inv = torch.rsqrt((sq + 2 * (x * delta).sum(-1, keepdim=True) + delta.square().sum(-1, keepdim=True)) / width + eps)
    return delta * inv + x * (inv - inv0)


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

    def identity_guess(self):
        """The identity path's per-token credit at each site: the top-stream credit carried down the residual scalars
        (PR #26: d_out[L-1] = top credit, d_out[i] = lam[i+1] * d_out[i+1]); at emb, its RMS-norm cycle."""
        keep = {}
        self.estimate(keep)
        lam = self.model.resid_lambdas.float()
        guess, carried = {'emb': keep['emb']}, keep['top']
        for i in reversed(range(self.n_layer)):
            guess[f'mid{i}'] = carried
            carried = lam[i] * carried
        return guess

    def credit_dump(self):
        _, true = reference_gradients(self.model, self.x, self.y)
        guess = self.identity_guess()
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

    def paths(self, model):
        """Tokens' own paths, every attention output frozen at its clean value (so tokens do not interact), batched
        over any leading dims. blocks(x, a, x0, first) runs blocks first.. from stream x (a: the frozen attention
        outputs [..., L, W]) and returns each block's mid stream and the final stream; loss(x, y) is the token's loss
        from the final stream; from_site(site, h, a, x0) is the final stream with the site's activation set to h."""
        norm, layers, vocab = self.m.norm, model.transformer.h, model.config.vocab_size
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()

        def blocks(x, a, x0, first=0):
            mids = []
            for i in range(first, self.n_layer):
                mid = lam_r[i] * x + lam_0[i] * x0 + a[..., i, :]
                mids.append(mid)
                x = mid + layers[i].mlp(norm(mid))
            return mids, x

        def loss(x, y):
            logits = 15 * torch.tanh(model.lm_head(norm(x))[..., :vocab].float() / 15)
            return -torch.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)

        def from_site(site, h, a, x0):
            if site == 'emb':
                return blocks(norm(h), a, norm(h))[1]
            k = int(site[3:])
            return blocks(h + layers[k].mlp(norm(h)), a, x0, k + 1)[1]

        return blocks, loss, from_site

    def token_map_svd(self, chunk=32):
        model, norm = copy.deepcopy(self.model).float(), self.m.norm
        values, frozen, _, _ = self.frozen(model)
        blocks, _, _ = self.paths(model)

        def to_sites(e, a):  # one token: raw embedding -> each block's mid stream
            return torch.stack(blocks(norm(e), a, norm(e))[0])

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
        names = [n for n, p in params.items() if is_anvil_matrix(n, p)]  # the matrices whose state is saved

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
        model = copy.deepcopy(self.model).float()
        values, frozen, x0_clean, targets = self.frozen(model)
        _, loss, from_site = self.paths(model)

        def token_loss(site, h, a, x0, y):
            return loss(from_site(site, h, a, x0), y)

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

    def credit_recovery(self, ridges=(0.0, 1e-5, 1e-4, 1e-3), sigma=0.01, route_k=512, chunk=16):
        """E5: one cycle launched at the embedding, read at every site, against a direct cycle at each site, both run
        as PR #40's chain runs its cycles (chain_chunk_eager): forward differences against the clean pass, each
        token's MLPs through its route_k strongest units (the record's routes: activation x |c_proj column|; None:
        every unit), the MLP and head products in float32 or, as the record runs them, in bf16. Every attention output
        is frozen, as the plan's diagnostic says; the chain's own-query response (a token's query moving its own
        attention output) is left out. Probes: rows of a signed Hadamard matrix times sigma x the token's RMS at the
        probed site (PR #40's chain_sigma step).
        At site k, A[i] is the launched probe's change there and d[i] the token's loss change; the recovered credit
        solves min |A c - d|^2 + lam |c - g|^2, lam = ridge x mean(diag A^T A), g the identity path's guess (ridge 0:
        A c = d). It is compared per token with the direct cycle's credit at the site (what the unchanged chain
        measures there), which is compared in turn with autograd (the token's own loss, attention and x0 held, every
        unit) and with autograd's full credit. The probe also moves x0 = norm(emb), which every later block reads,
        so d carries paths that skip site k: those, the conditioning of A and rounding are what this measures.
        A second launch, block 0's own direct cycle read at the later sites (x0 held, so every path passes each of
        them), says whether two cycles (the embedding's and mid0's) could stand in for nine. Condition numbers are
        reported with and without the smallest singular value (the embedding's RMS norm hides its radial direction
        to first order). One SVD per site and chunk gives every ridge's solution."""
        model, norm = copy.deepcopy(self.model).float(), self.m.norm
        values, frozen, x0_clean, targets = self.frozen(model)
        blocks, loss, from_site = self.paths(model)
        names, L, layers = site_names(self.n_layer), self.n_layer, model.transformer.h
        guess = self.identity_guess()
        _, full = reference_gradients(self.model, self.x, self.y)
        with exact():
            autograd = {site: torch.func.vmap(torch.func.grad(lambda h, a, x0, y, site=site: loss(
                from_site(site, h, a, x0), y)))(values[site][0], frozen, x0_clean, targets) for site in names}
            acts = [values[site][0] for site in names]  # each site's clean activation [T, W]
            final = blocks(x0_clean, frozen, x0_clean)[1]  # the clean top stream, before the final norm
            pre = [F.linear(norm(mid), b.mlp.c_fc.weight.float()) for mid, b in zip(acts[1:], layers)]
            hidden = pre[0].shape[-1]
            k = min(route_k or hidden, hidden)
            routes = [(F.relu(h).square() * b.mlp.c_proj.weight.float().norm(dim=0)).topk(k, -1).indices
                      for h, b in zip(pre, layers)]
            head = model.lm_head.weight.float()[:model.config.vocab_size]
            base = final @ head.T
        lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
        cap = lambda z: 15 * torch.tanh(z / 15)
        eps = torch.finfo(torch.float32).eps

        def changes(sl, site, v, dtype):
            """chain_chunk_eager for moves v [t, P, W] at site (0: the embedding, 1 + b: block b's mid stream) and
            tokens sl, without own-query responses: each site's change from there on and the loss change [t, P]."""
            xL, z0 = final[sl, None], base[sl, None]
            if site == 0:
                dx0 = norm(acts[0][sl, None] + v) - x0_clean[sl, None]
                delta, first, out = (lam_r[0] + lam_0[0]) * dx0, 0, [v]
            else:
                dx0, delta, first, out = None, v, site - 1, []
            for b in range(first, L):
                if b > first:
                    delta = lam_r[b] * delta + (lam_0[b] * dx0 if dx0 is not None else 0)
                out.append(delta)
                idx, mid = routes[b][sl], acts[b + 1][sl, None]
                fc = layers[b].mlp.c_fc.weight.to(dtype)[idx]  # [t, k, W]: each token's routed units
                wo = layers[b].mlp.c_proj.weight.to(dtype).T[idx]
                h = pre[b][sl].gather(-1, idx)[:, None]
                dh = torch.bmm(norm_difference(mid, delta).to(dtype), fc.mT).float()
                up, down = F.relu(h + dh), F.relu(h)
                delta = delta + torch.bmm(((up - down) * (up + down)).to(dtype), wo).float()
            y = targets[sl, None]
            w_y, z0_y = head[y], z0.gather(-1, y[..., None])[..., 0]
            inv = torch.rsqrt((xL + delta).square().mean(-1) + eps)
            inv0 = torch.rsqrt(xL.square().mean(-1) + eps)
            z = (z0 + (delta.to(dtype) @ head.T.to(dtype)).float()) * inv[..., None]
            moved = torch.logsumexp(cap(z), -1) - cap((z0_y + (delta * w_y).sum(-1)) * inv)
            clean = torch.logsumexp(cap(z0 * inv0[..., None]), -1) - cap(z0_y * inv0)
            return out, moved - clean

        tokens, width = acts[0].shape
        signs = torch.randint(0, 2, (width,), generator=torch.Generator().manual_seed(0)).to(acts[0]) * 2 - 1
        basis = hadamard(width, acts[0].device) * signs  # rows: the cycle's W orthogonal directions
        precisions = {'float32': torch.float32, 'bf16': torch.bfloat16}
        launches = {'emb': 0, 'mid0': 1}  # the plan's launch, and block 0's direct cycle read further up (x0 held)
        later = {launch: names[j + 1:] for launch, j in launches.items()}
        cos = {(p, l, s, r): [] for p in precisions for l in launches for s in later[l] for r in ridges}
        condition = {(p, l, s, kind): [] for p in precisions for l in launches for s in later[l]
                     for kind in ('all', 'without_radial')}
        direct = {(p, s): [] for p in precisions for s in names}
        with exact(), torch.no_grad():
            for precision, dtype in precisions.items():
                for t0 in range(0, tokens, chunk):
                    sl = slice(t0, t0 + chunk)
                    steps = [sigma * a[sl, None].square().mean(-1, keepdim=True).sqrt() for a in acts]  # [t, 1, 1]
                    cycles = [changes(sl, j, steps[j] * basis, dtype) for j in range(len(names))]
                    for site, (_, d), step in zip(names, cycles, steps):  # each site's credit, decoded as the chain does
                        direct[precision, site].append((d / step[..., 0]) @ basis / width)
                    for launch, j0 in launches.items():
                        moved, d = cycles[j0]
                        d = d.unsqueeze(-1)
                        for site, A in zip(later[launch], moved[1:]):  # moved[0] is the probe itself
                            U, S, Vh = torch.linalg.svd(A)
                            condition[precision, launch, site, 'all'].append(S[:, 0] / S[:, -1].clamp_min(1e-30))
                            condition[precision, launch, site, 'without_radial'].append(
                                S[:, 0] / S[:, -2].clamp_min(1e-30))
                            # min |A c - d|^2 + lam |c - g|^2: c = V (S^2 + lam)^-1 (S U^T d + lam V^T g)
                            Ud, Vg, S = U.mT @ d, Vh @ guess[site][sl, :, None].float(), S[..., None]
                            scale = S.square().mean(1, keepdim=True)  # mean(diag A^T A)
                            measured = direct[precision, site][-1]
                            for ridge in ridges:
                                lam = ridge * scale
                                c = Vh.mT @ ((S * Ud + lam * Vg) / (S.square() + lam).clamp_min(1e-30))
                                cos[precision, launch, site, ridge].append(per_token_cosine(c[..., 0], measured))
        summary, tensors = {}, {}
        for site in names:
            measured = torch.cat(direct['float32', site])
            row = dict(guess_vs_direct=quantiles(per_token_cosine(guess[site], measured)),
                       direct_vs_autograd=quantiles(per_token_cosine(measured, autograd[site])),
                       autograd_vs_full=quantiles(per_token_cosine(autograd[site], full[site])))
            for precision in precisions:
                for launch in launches:
                    if site not in later[launch]:
                        continue
                    for kind in ('all', 'without_radial'):
                        row[f'{precision} {launch} condition {kind}'] = quantiles(
                            torch.cat(condition[precision, launch, site, kind]))
                    for ridge in ridges:
                        per_token = torch.cat(cos[precision, launch, site, ridge])
                        row[f'{precision} {launch} ridge {ridge}'] = quantiles(per_token)
                        tensors[f'{site} {precision} {launch} ridge {ridge}'] = per_token.cpu()
            summary[site] = row
        summary['settings'] = dict(sigma=sigma, ridges=list(ridges), route_k=k, probes=width, tokens=tokens,
                                   launches=list(launches))
        return summary, tensors


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

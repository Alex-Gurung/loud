"""E8: signal-to-noise along the momentum's singular directions (v2). Does an orthogonalizing optimizer amplify
estimator noise, and does scaling its steps by measured signal-to-noise fix that?

Along a direction (u_i, v_i), each sequence's gradient G_s projects to p_s = u_i^T G_s v_i. Over S sequences with
mean m and variance s^2, m^2 - s^2 / S estimates mu^2 without the noise bias (for a direction chosen independently of
the sequences), and a batch of B averages the noise to sigma^2 / B.

    snr_spectrum    H6 item 4, offline on a checkpoint: per block matrix, along every singular direction of ANVIL II's
                    momentum (the checkpoint's fast and slow rails, blended by its fast weight: chosen before, so
                    independently of, the measured sequences), the SNR at batch B over S sequences (64 in the
                    configs), and the share of the momentum's energy (squared singular values) in directions below
                    SNR 1. `gradients` supplies the per-sequence gradients: exact backprop by default, or any
                    estimate (H8's synthesized partial rows).
    SNRScaling      the intervention, through loud/anvil.py's ShapedAnvil (the proxy) and ShapedBatchedAnvil (H8):
                    with U, V the singular directions of ANVIL II's momentum as it stood before this batch (its
                    blended rails, as the diagnostic takes them), the update becomes U diag(r) V^T, r_i =
                    max(0, m_i^2 - s_i^2 / B) / m_i^2 over the batch's B per-sequence gradients (H2), an estimate of
                    mu^2 / (mu^2 + sigma^2 / B). Directions taken from the update itself would hold the batch's own
                    gradient (weight 0.14 to 0.21 on the records' rails) and make its noise look like signal: on pure
                    noise, mean r 0.85 against 0.18 (review round 7). So the batch enters the step only through r;
                    the first update, whose rails are still zero, takes the update's own directions. ANVIL II's
                    per-lane gain and Frobenius renormalization would undo the scaling, so every arm runs without
                    them: 'control' (r = 1), 'snr', and 'rate' (a uniform r, each matrix's sqrt(mean r_i^2): the SNR
                    arm's Frobenius ratio, so that a gain cannot come from a smaller step alone). One SVD per block
                    matrix per update (development).
"""
import copy

import torch

from .anvil import is_anvil_matrix
from .pergrad import matrix_type


def projections(grads, u, vh):
    """p[s, i] = u_i^T G_s v_i for per-sequence gradients grads [S, rows, cols] and SVD factors u [rows, k],
    vh [k, cols]."""
    return torch.einsum('sab,ai,ib->si', grads, u, vh)


def moments(p):
    """(m^2, s^2): the squared mean and the variance (unbiased) over the sequences of projections p [S, k]."""
    return p.mean(0).square(), p.var(0)


def exact_gradients(model, rows):
    """Each sequence's gradient of the mean loss (float32, as training forms it) for ANVIL II's matrices:
    {name: [S, rows, cols]}."""
    model = copy.deepcopy(model).float().requires_grad_(True)
    named = [(n, p) for n, p in model.named_parameters() if is_anvil_matrix(n, p)]
    out = {n: [] for n, _ in named}
    for row in rows.split(1):
        loss = model(row[:, :-1].contiguous(), row[:, 1:].contiguous())
        for (n, _), g in zip(named, torch.autograd.grad(loss, [p for _, p in named])):
            out[n].append(g)
    return {n: torch.stack(gs) for n, gs in out.items()}


def momentum(rails, saved):
    """ANVIL II's blended momentum, w * fast + (1 - w) * slow, from a checkpoint's .optim.pt (loud/checkpoints.py)."""
    weight = saved['fast_weight']
    return weight * rails['fast'].float() + (1 - weight) * rails['slow'].float()


def snr_spectrum(model, rows, optimizer_state, batch=8, gradients=exact_gradients):
    """Per ANVIL II matrix: along each singular direction of the checkpoint's momentum, the SNR at `batch`,
    (m^2 - s^2 / S) / (s^2 / batch) over the S = len(rows) sequences, and the share of the momentum's energy in
    directions with SNR below 1; per matrix type, that share over its matrices. Returns (summary, tensors: per
    matrix the singular values and SNRs)."""
    grads = gradients(model, rows)
    summary, tensors, kinds = dict(sequences=len(rows), batch=batch), {}, {}
    with torch.no_grad():
        for name, g in grads.items():
            g = g.float()
            u, sv, vh = torch.linalg.svd(momentum(optimizer_state['state'][name], optimizer_state).to(g),
                                         full_matrices=False)
            m2, s2 = moments(projections(g, u, vh))
            snr = (m2 - s2 / len(g)) / (s2 / batch).clamp_min(1e-30)
            energy = sv.square()
            low = (energy * (snr < 1)).sum().item()
            summary[name] = dict(energy_below_1=low / energy.sum().item(), snr_median=snr.median().item(),
                                 snr_top16=snr[:16].tolist())
            tensors[name] = dict(singular=sv.cpu(), snr=snr.cpu())
            row = kinds.setdefault(matrix_type(name), [0.0, 0.0])
            row[0], row[1] = row[0] + low, row[1] + energy.sum().item()
    summary['energy_below_1'] = {kind: low / total for kind, (low, total) in kinds.items()}
    return summary, tensors


class SNRScaling:
    """E8's shaper for ShapedAnvil (lane_gain False: no per-lane gain or renormalization after it). The proxy hands it
    the update's per-sequence gradients (H2) before ANVIL II steps; report() logs, per matrix type, the mean r and the
    Frobenius ratio sqrt(mean r^2) of the last update."""
    lane_gain = False

    def __init__(self, arm, model):
        self.arm = arm
        self.names = {id(p): n for n, p in model.named_parameters() if is_anvil_matrix(n, p)}
        self.pergrad, self.ratios = None, {}

    def before_step(self, step, pergrad):
        self.pergrad, self.ratios = pergrad, {}

    def direction(self, p, pending, before=None):
        basis = pending if before is None or not before.any() else before
        u, _, vh = torch.linalg.svd(basis, full_matrices=False)
        if self.arm == 'control':
            return u @ vh
        name = self.names[id(p)]
        grads = self.pergrad.stack(name).float()
        m2, s2 = moments(projections(grads, u, vh))
        r = (m2 - s2 / len(grads)).clamp_min(0) / m2.clamp_min(1e-30)
        frobenius = r.square().mean().sqrt()
        self.ratios.setdefault(matrix_type(name), []).append(torch.stack([r.mean(), frobenius]))
        if self.arm == 'rate':
            r = frobenius.expand_as(r)
        return (u * r) @ vh

    def report(self, step):
        """Per matrix type: the mean r and the Frobenius ratio of this update (one sync)."""
        if self.arm == 'control' or not self.ratios:
            return {}
        means = {kind: torch.stack(v).mean(0) for kind, v in self.ratios.items()}
        values = torch.stack(list(means.values())).tolist()
        return dict(snr_ratios={kind: dict(r_mean=round(a, 5), frobenius=round(b, 5))
                                for kind, (a, b) in zip(means, values)})

"""Proxy variants: each changes one thing about how the backprop proxy forms or takes an update (the plan's E2, E4
and E10). Each is off unless its option is set (loud/options.py, PROXY); the plain proxy never enters this module.

    Blur          E2: the gradient of the loss blurred at width tau, from an exact antithetic pair of displaced passes.
    StepControl   E4: undo an update that raised its batch's loss; optionally scale every rate by the outcomes.
    TokenShaping  E10: per-token credits capped, or weight-gradient entries kept only where the tokens agree in sign.
"""
import contextlib

import torch
from torch import nn

CHAIN_SIGMA = 0.01  # PR #40's Settings.chain_sigma: the Cut chain's probe step, relative to each token's RMS
BLUR_SEED = 7907    # the blur draws from its own generator, seeded with the job seed + BLUR_SEED


class Blur:
    """E2. The probing sites are the raw embedding (wte's output) and each block's mid stream (after attention, before
    the MLP), nine for the benchmark model. At each site, token t gets r[t]: a uniformly random direction of length
    tau * CHAIN_SIGMA * RMS(a[t]) * sqrt(width), with a[t] its clean activation there. That is the Cut chain's probe
    with a random direction: a Hadamard row of +-1 entries (norm sqrt(width)) times the step chain_sigma * RMS(a[t]),
    where PR #40's chain_credits takes the RMS per token. This answers the plan's open question: tau counts per-token
    steps. Two forwards add +r and -r (the same r) at every site at once. The mean of their gradients has no odd-order
    terms in r, and over the draws of r it averages to the gradient of the loss blurred at width tau. The displacement
    is added in the activation's dtype (bf16 on the GPU), as the chain's bf16 probes are. A mid-stream displacement is
    added to the attention output, which is what a block adds to its input there."""

    def __init__(self, model, tau, seed, device):
        self.model, self.width = model, model.config.n_embd
        self.scale = tau * CHAIN_SIGMA * self.width ** 0.5
        self.generator = torch.Generator(device).manual_seed(seed + BLUR_SEED)
        self.shifts = None

    @contextlib.contextmanager
    def _sites(self, visit):
        """For one forward, call visit(site, output, block_input) at each site: output writes the site's activation
        (wte's output; a block's attention output, added to block_input). A returned tensor replaces the output."""
        inputs, handles = {}, []

        def embedding(module, args, output):
            return visit(0, output, None)

        def block_input(i):
            def hook(module, args):
                inputs[i] = args[0]
            return hook

        def attention(i):
            def hook(module, args, output):
                return visit(i + 1, output, inputs.pop(i))
            return hook

        try:
            handles.append(self.model.transformer.wte.register_forward_hook(embedding))
            for i, block in enumerate(self.model.transformer.h):
                handles.append(block.register_forward_pre_hook(block_input(i)))
                handles.append(block.attn.register_forward_hook(attention(i)))
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def measure(self, forward):
        """Run the clean forward (forward() returns its loss) and draw this micro-batch's displacements from its
        per-token RMS at every site. Returns the clean loss."""
        seen = []

        def visit(site, output, block_input):
            a = output.float() if block_input is None else block_input.float() + output.float()
            seen.append((a.square().mean(-1, keepdim=True).sqrt(), output.dtype))

        with self._sites(visit):
            loss = forward()
        self.shifts = []
        for rms, dtype in seen:  # sites in forward order: the embedding, then each block's mid stream
            u = torch.randn(rms.shape[:-1] + (self.width,), device=rms.device, generator=self.generator)
            self.shifts.append(u.mul_(self.scale * rms / u.norm(dim=-1, keepdim=True)).to(dtype))
        return loss

    def shifted(self, sign):
        """A context in which forwards add sign * r (the last measure()'s) at every site."""
        def visit(site, output, block_input):
            return output + self.shifts[site] if sign > 0 else output - self.shifts[site]
        return self._sites(visit)


class Snapshot:
    """The weights and the optimizer state (every state tensor; ANVIL II's schedule clock is left to the loop), copied into buffers that
    are allocated once per tensor and reused, so a rejected update can be undone exactly."""

    def __init__(self, model, optimizer):
        self.params, self.optimizers = list(model.parameters()), optimizer.optimizers
        self.buffers = {}

    def _tensors(self):
        held = list(self.params)
        for o in self.optimizers:
            for state in o.state.values():
                held += [v for v in state.values() if torch.is_tensor(v)]
        return held

    def save(self):
        self.saved = self._tensors()
        for t in self.saved:
            if id(t) not in self.buffers:
                self.buffers[id(t)] = torch.empty_like(t)
        torch._foreach_copy_([self.buffers[id(t)] for t in self.saved], self.saved)

    def restore(self):
        """Undo the step: weights, rails, energy and Adam's moments and counts. ANVIL II's clock `t` is not a record
        of the data but its schedule (rail-beta ramp, engage, cooldown), which follows the loop's update count as the
        warmdown does, so it stays advanced."""
        torch._foreach_copy_(self.saved, [self.buffers[id(t)] for t in self.saved])
        known = {id(t) for t in self.saved}
        created = [t for t in self._tensors() if id(t) not in known]
        if created:  # the undone step was the first: zero state is what a fresh optimizer starts from
            torch._foreach_zero_(created)


class StepControl:
    """E4. After each update, the batch's loss at the new weights, from one no-grad forward over the same
    micro-batches as the gradient, so the two losses compare like for like. If it rose, the weights and the whole
    optimizer state go back to before the update, and the batch feeds no update. Version one ('reject') stops there.
    'adapt' (version two) also keeps a global multiplier on every group's rate, on top of the hand schedule: times
    accept_up after an accept, times accept_down after a reject. It settles where the acceptance rate a satisfies
    a * ln(up) + (1 - a) * ln(down) = 0; for the plan's 1.05 and 0.8 that is a = 0.82, not the 0.6 it targets."""

    def __init__(self, model, optimizer, o):
        self.adapt, self.up, self.down = o['step_control'] == 'adapt', o['accept_up'], o['accept_down']
        self.multiplier = 1.0
        self.snapshot = Snapshot(model, optimizer)

    def save(self):
        """Before the step: keep what it will change."""
        self.snapshot.save()

    def scale(self, optimizer):
        """Apply the multiplier to the rates set_rate just set (they are recomputed from initial_lr every update)."""
        if self.adapt:
            for group in optimizer.param_groups:
                group['lr'] *= self.multiplier

    def judge(self, loss, trial):
        """After the step: undo it if the batch's loss rose from `loss` to `trial`. Returns the metrics fields."""
        accepted = bool(trial <= loss)  # one sync per update (the loop waits for the loss anyway)
        if not accepted:
            self.snapshot.restore()
        record = dict(accepted=accepted, trial_loss=trial.item())
        if self.adapt:
            record['lr_multiplier'] = self.multiplier  # the multiplier this update used
            self.multiplier *= self.up if accepted else self.down
        return record


class TokenShaping:
    """E10, on the weight gradient of every nn.Linear in the model: the block matrices, the value-embedding gates and
    the LM head. Each of these gradients is a sum over the update's tokens of credit (the loss gradient at the layer's
    output) times input, which is the sum the plan reshapes. Embedding tables (one-hot inputs) and the mixing scalars
    keep their ordinary gradients.

    cap: before the outer products, each token's credit is scaled down to a norm of at most the cap_quantile of the
    per-token credit norms over the whole update batch, per layer. The quantile needs every micro-batch's credits, so
    each layer's credits and inputs are kept (bf16, about 0.3 GB per sequence for the benchmark model) until the batch
    is done; a second backward for the norms would cost a full pass more.
    agree: each entry of the ordinary gradient is kept only where at least agree_threshold of the tokens share one sign
    of their contribution credit[i] * input[j]: of the tokens with a nonzero contribution (agree_over 'active', the
    default), or of all the update's tokens (agree_over 'all', the plan's literal reading, under which an entry whose
    input or credit is zero for more than 1 - threshold of the tokens, common after the ReLU^2, is always dropped).
    The counts are sign matmuls accumulated over the micro-batches, exact for +-1 and 0 entries in float32 (TF32 on the
    GPU) with float32 sums, so nothing per token is kept.

    While shaping, these weights skip autograd's own weight gradient (requires_grad off); a hook on each layer's output
    forms it from the captured input, with the same matmul as autograd."""

    def __init__(self, model, accumulators, o):
        slot = {id(p): a for p, a in zip(model.parameters(), accumulators)}
        self.modules = [m for m in model.modules() if isinstance(m, nn.Linear)]
        self.sums = [slot[id(m.weight)] for m in self.modules]  # the proxy's float32 gradient accumulators
        self.mode, self.quantile, self.threshold = o['token_shaping'], o['cap_quantile'], o['agree_threshold']
        self.over_all, self.tokens = o['agree_over'] == 'all', 0  # tokens: this update's, per layer (shapes, no sync)
        self.kept = [[] for _ in self.modules]  # cap: (credit, input) of each micro-batch, per layer
        self.counts = ([(torch.zeros_like(s), torch.zeros_like(s)) for s in self.sums] if self.mode == 'agree'
                       else None)  # agree: (sum of sign products, count of nonzero products), per layer

    @contextlib.contextmanager
    def batch(self):
        """Hooks for one update's backward passes; finish() then writes the shaped sums."""
        if self.counts is not None:
            torch._foreach_zero_([c for pair in self.counts for c in pair])
            self.tokens = 0
        handles = []

        def forward_hook(k):
            def hook(module, args, output):
                if output.requires_grad:
                    x = args[0].to(output.dtype)  # the cast autocast feeds the matmul
                    output.register_hook(lambda g: self._credit(k, x, g))
            return hook

        try:
            for k, module in enumerate(self.modules):
                module.weight.requires_grad_(False)
                handles.append(module.register_forward_hook(forward_hook(k)))
            yield
        finally:
            for handle in handles:
                handle.remove()
            for kept in self.kept:
                kept.clear()

    def _credit(self, k, x, g):
        g, x = g.reshape(-1, g.shape[-1]), x.reshape(-1, x.shape[-1])
        if self.mode == 'cap':
            self.kept[k].append((g, x))
            return
        if k == 0:
            self.tokens += g.shape[0]  # every layer sees each micro-batch's tokens
        self.sums[k].add_(torch.mm(x.t(), g).t())  # autograd's weight gradient of the linear map
        same, nonzero = self.counts[k]
        sg, sx = g.sign().float(), x.sign().float()
        same.addmm_(sg.t(), sx)
        nonzero.addmm_(sg.abs_().t(), sx.abs_())

    def finish(self):
        """Write each layer's shaped gradient sum into its accumulator (before the proxy divides by the passes)."""
        if self.mode == 'agree':
            for total, (same, nonzero) in zip(self.sums, self.counts):
                majority = same.abs_().add_(nonzero)  # twice the number of products with the majority sign
                total.mul_(majority >= (2 * self.threshold * self.tokens if self.over_all
                                        else nonzero.mul_(2 * self.threshold)))
            return
        for total, kept in zip(self.sums, self.kept):
            norms = [g.float().norm(dim=-1) for g, _ in kept]
            cap = torch.quantile(torch.cat(norms), self.quantile)
            for (g, x), norm in zip(kept, norms):
                scale = (cap / norm.clamp_min(1e-30)).clamp_max_(1)
                total.add_(torch.mm(x.t(), (g.float() * scale[:, None]).to(g.dtype)).t())

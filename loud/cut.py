"""H5: the record's Boost -> Cut chain (Noise PR #40, 10M record #14: 3.676 in 2101 s on 8xH100) on one GPU.

PR #40's archive/10M/train/val/run.py (@4ef285e) is imported unchanged: its model, init, settings, Boost and Cut
estimators, optimizers, schedule and end merge. That script runs on 8 GPUs at batch 4, each sequence split over two
GPUs: the two take complementary halves of every cycle of moves and draws, add their parts with an all-reduce inside
the estimate (group_sum), and a second all-reduce sums the 8 GPUs' gradients for the step. Here the 8 GPUs are virtual
ranks with the record's seeds and roles, measured in rank order. The GPUs of a sequence are threads that take turns
(Turns): one runs at a time and hands over at each group_sum, where the parts are added as the all-reduce adds them.
So the run draws the 8-GPU run's numbers; the step's sum over ranks is taken in rank order (NCCL's order differs,
which changes float32 rounding only). A CPU test runs the record's own train_stages on 8 gloo processes, its
all-reduces summing in rank order, and gets bit-identical losses and weights.

Schedule as the record's: stage_schedule (Boost for the first 20% of updates, then Cut), the optimizer handoff at the
switch, the rates. The record's nine trajectory means are loud/train.py's (TrajectoryMeans keeps them by the same rule:
from 60% of the update sequences, every 5 updates and at the last), and the end merge runs after the last update inside
the clock, scoring its candidates on the 32 held-out sequences rank by rank as the 8-GPU all-reduce sums them.

E6 and E7 change only the Cut phase's chain (chain_credits: the embedding and the 8 mid streams). Boost, the head-only
top probe (probe()) and the key, gate, query and future probes keep their budgets. With fewer than the full 512 rows per
token at a site, a token's credit there is read from its rows of the site's signed Hadamard basis and scaled by
512 / rows. That is unbiased: the basis's random column signs make the outer products of any set of its rows average
to the set's size times the identity. The full cycle runs the record's chain_credits itself.
  - E6 (rows, row_offset_per_step): update s reads rows [(s mod K) * rows, ...) at every site, K = 512 / rows, from
    bases held for the K updates of cycle s // K (one per site), so each cycle measures every direction once, spread
    over K updates for momentum to stitch (the Cut phase's first cycle may start part way through). Without the
    offset: rows [0, rows) of fresh bases (the control).
  - E7 (rows_start, rows_full_at): the rows per token grow linearly from rows_start at the first Cut update to 512 at
    the fraction rows_full_at of the run's updates, in steps of 64 (so few chain shapes compile, all in the warmup).
    Token t reads the window that starts at (t * rows) mod 512 of fresh bases: consecutive tokens tile the cycle, so
    every row is read in every update.
Differences from PR #40, all deliberate: the job's seed replaces the module's SEED (seed 0 is the record's); the shared
loop (loud/train.py): H4, untimed diagnostics, official scoring in a fresh process; the means are updated in the
loop's untimed pause instead of the timed region (a 9-way lerp every 5 updates).
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
import threading
import time

import torch

from . import refs
from .identity import DRIFT_SEED, attach, cosine
from .options import CUT_SWITCH, PRESETS, ROW_QUANTUM, cut_options
from .train import current_rates, rate_labels, train


class Turns:
    """The GPUs of one sequence as threads that take turns: one runs at a time and hands the turn to the next part at
    each group_sum, where the parts meet as the GPUs' all-reduce: the sum in part order is written to every part's
    tensor (two GPUs' sum is a + b, the same in either order). Part 0 runs on the calling thread. Grad mode and
    autocast are per thread, but the estimators set both themselves (estimate runs under no_grad, the capture under its
    own autocast), so every part sees the state the record's processes do."""

    def __init__(self, parts):
        self.parts, self.lock = parts, threading.Condition()
        self.pool = ThreadPoolExecutor(parts - 1, thread_name_prefix='part')

    def run(self, jobs):
        """Run jobs[p]() as part p; returns their results in part order."""
        self.turn, self.arrived, self.done, self.failed = 0, [None] * self.parts, [False] * self.parts, None
        futures = [self.pool.submit(self._part, p, job) for p, job in enumerate(jobs) if p]
        try:
            first = self._part(0, jobs[0])
        except BaseException as exc:  # also an interrupt while waiting: wake the other parts so they stop
            self._fail(exc)
            raise
        finally:
            others = [f.exception() or f.result() for f in futures]  # waits for every part, failed or not
        for result in others:
            if isinstance(result, BaseException):
                raise result
        return [first] + others

    def group_sum(self, part, tensor):
        with self.lock:
            self.arrived[part] = tensor
            if all(t is not None for t in self.arrived):
                total = self.arrived[0].clone()
                for t in self.arrived[1:]:
                    total += t
                for t in self.arrived:
                    t.copy_(total)
                self.arrived = [None] * self.parts
            self._pass(part)
            self._wait(part)
        return tensor

    def _part(self, part, job):
        with self.lock:
            self._wait(part)
        try:
            result = job()
        except BaseException as exc:
            self._fail(exc)
            raise
        with self.lock:
            self.done[part] = True
            if any(t is not None for t in self.arrived):
                self.failed = RuntimeError('the parts of a sequence reached different collectives')
            self._pass(part)
        return result

    def _fail(self, exc):
        with self.lock:
            self.failed = self.failed or exc
            self.lock.notify_all()

    def _pass(self, part):
        """The turn to the next part (in order, cyclically) that has not finished."""
        for k in range(1, self.parts + 1):
            if not self.done[(part + k) % self.parts]:
                self.turn = (part + k) % self.parts
                break
        self.lock.notify_all()

    def _wait(self, part):
        self.lock.wait_for(lambda: self.turn == part or self.failed is not None)
        if self.failed is not None:
            raise RuntimeError('another part of the sequence failed') from self.failed


def estimator_classes(m):
    """PR #40's Boost and Cut with group_sum through Turns, and Cut's chain with E6/E7's row counts."""

    class Parts:
        turns = None  # the trainer's Turns; group_sum needs it only when a sequence has several parts

        def group_sum(self, tensor):
            return tensor if self.split == 1 else self.turns.group_sum(self.part, tensor)

    class Boost(Parts, m.Boost):
        pass

    class Cut(Parts, m.Cut):
        rows = None    # rows per token at each chain site, the sequence's GPUs together (None: the full cycle)
        start = 0      # E6: every token's window starts here; None: E7's windows tiled across tokens
        cycle = None   # E6: the cycle whose bases are held (None: fresh bases each update)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.held = (None, {})  # (cycle, {site: basis})

        def site_basis(self, site):
            """A chain site's signed basis, drawn as the record draws it (one per site, in site order, so the random
            stream does not depend on the rows); with a cycle set, the cycle's first draw is held through it."""
            fresh = self.basis()
            if self.cycle is None:
                return fresh
            if self.held[0] != self.cycle:
                self.held = (self.cycle, {})
            return self.held[1].setdefault(site, fresh)

        def row_index(self, n):
            """[n, rows / split]: each token's rows of the cycle, this GPU's share (the first or second half of the
            token's window)."""
            share = self.rows // self.split
            if self.start is None:
                first = torch.arange(n, device=self.device) * self.rows
            else:
                first = torch.full((n,), self.start, device=self.device)
            return (first[:, None] + self.part * share + torch.arange(share, device=self.device)) % self.width

        def chain_credits(self, e, x0, mids, outs, routes, y):
            if self.rows is None:
                return super().chain_credits(e, x0, mids, outs, routes, y)
            return self.partial_chain_credits(e, x0, mids, outs, routes, y)

        def partial_chain_credits(self, e, x0, mids, outs, routes, y):
            """PR #40's chain_credits with each token reading `rows` rows of each site's basis (its `index`) instead of
            this GPU's half of the whole basis, and the credit decoded over those rows (divided by rows, not width).
            Every other line is the record's (its token-chunk size m is m_ here, as m is the module)."""
            model, s = self.model, self.s
            L = model.config.n_layer
            vocab = model.config.vocab_size
            weight = model.lm_head.weight.float()[:vocab]
            weight_t = weight.T.to(torch.bfloat16).contiguous()
            w_fc = [blk.mlp.c_fc.weight.to(s.chain_dtype) for blk in model.transformer.h]
            w_out_t = [blk.mlp.c_proj.weight.to(s.chain_dtype).T.contiguous() for blk in model.transformer.h]
            lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
            xL = outs[-1]
            base = xL @ weight.T
            target_w = weight[y.clamp(min=0)]
            n, m_, h = e.shape[0], s.chain_tokens, s.scalar_step
            probes = []
            index = self.row_index(n)  # changed: each token's rows, not this GPU's half of all of them
            P, scalars_here = index.shape[1], self.part == 0
            for site in range(-1, L):
                H = self.site_basis(site)  # changed: the whole basis (rows are picked per token below)
                probed = e if site < 0 else mids[site]
                steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                credit = torch.empty(n, self.width, device=self.device)
                lam = torch.zeros(2, device=self.device)
                probes.append((site, H, steps, credit, lam))
            length = self.capture.length if s.own_query else n
            for start in range(0, n, length):
                own = None
                if s.own_query:
                    own = self.own_query_inputs(start // length)
                    if s.query_keys:
                        own = self.strong_keys(own, s.query_keys)
                for t0 in range(start, start + length, m_):
                    t1 = min(t0 + m_, start + length)
                    sl = slice(t0, t1)
                    own_sl = None
                    if own is not None:
                        begin, end = t0 - start, t1 - start
                        if own[0] == 'routed':
                            local = slice(begin, end)
                            own_sl = ('routed', own[1][local], [z[local] for z in own[2]], own[3]) + tuple(
                                [z[local] for z in part] for part in own[4:8]) + own[8:]
                        else:
                            own_sl = self.own_query_window(own, begin, end)
                    kernel = m.chain_chunk_own if own_sl is not None and own_sl[0] == 'window' else m.chain_chunk
                    selected_fc = [w[r[0][sl]] for w, r in zip(w_fc, routes)]
                    selected_out = [w[r[0][sl]] for w, r in zip(w_out_t, routes)]
                    for site, H, steps, credit, lam in probes:
                        directions = H[index[sl]]  # changed: [tokens, P, width], each token's rows
                        move = steps[sl].unsqueeze(-1) * directions
                        scale = steps[sl].expand(-1, P)
                        if site >= 0 and scalars_here:  # symmetric probes of both linear mixing scalars
                            prev = (x0 if site == 0 else outs[site - 1])[sl].unsqueeze(1)
                            move = torch.cat([move, h * prev, h * x0[sl].unsqueeze(1),
                                              -h * prev, -h * x0[sl].unsqueeze(1)], 1)
                            scale = torch.cat([scale, torch.full_like(scale[:, :2], h),
                                               torch.full_like(scale[:, :2], -h)], 1)
                        derivative = kernel(
                            site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl],
                            [r[0][sl] for r in routes], [r[1][sl] for r in routes], lam_r, lam_0, selected_fc,
                            selected_out, weight_t, base[sl], target_w[sl], y[sl], own_sl, packed=True)
                        # changed: decoded over the token's rows, scaled by width / rows
                        credit[sl] = torch.einsum('tp,tpw->tw', derivative[:, :P], directions) / self.rows
                        if site >= 0 and scalars_here:
                            lam += (derivative[:, P:P + 2] + derivative[:, P + 2:]).sum(0) / 2
                    del selected_fc, selected_out
                del own_sl, own
            credits = {}
            for site, _, _, credit, lam in probes:
                credits['emb' if site < 0 else site] = self.group_sum(credit)
                if site >= 0:
                    credits['lam', site] = self.group_sum(lam)
            return credits

    return Boost, Cut


def record_settings(m, preset, batch):
    """PR #40's Settings for a model preset: the record's at '38m'; smaller presets (smoke tests only) scale the
    budgets that the record ties to the width (input-K pairs = width, future draws = width / 4, routed units = hidden
    / 4) and keep at most their blocks as future sources."""
    n_layer, width = PRESETS[preset]
    config = m.model_config()
    config.n_layer, config.n_head, config.n_kv_head, config.n_embd = n_layer, width // 64, width // 64, width
    return m.Settings(model=config, boost_batch=batch, cut_batch=batch, key_path_draws=width,
                      future_draws=width // 4, route_k=width, future_layers=min(4, n_layer))


def virtual_holdout_loss(ranks):
    """The record's holdout_loss as `ranks` GPUs compute it: each rank its rows[rank::ranks] in batches of 8, the
    all-reduce a float64 sum. Its other lines are the record's (with its CUDA autocast, a no-op on a CPU)."""

    @torch.no_grad()
    def holdout_loss(model, rows, rank, world):
        device = next(model.parameters()).device
        total = torch.zeros(2, dtype=torch.float64, device=device)
        for r in range(ranks):
            for batch in rows[r::ranks].split(8):
                batch = batch.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
                total[0] += loss.double()
                total[1] += batch[:, 1:].numel()
        return (total[0] / total[1]).item()

    return holdout_loss


class CutChain:
    def __init__(self, o, device, seed, steps, heldout):
        m = self.m = refs.trainer_module('pr40')
        self.o, self.device, self.seed, self.steps, self.heldout = o, device, seed, steps, heldout
        self.settings = s = record_settings(m, o['preset'], o['batch'])
        if s.switch != CUT_SWITCH:
            raise RuntimeError(f'PR #40 switches at {s.switch}; loud.options.CUT_SWITCH says {CUT_SWITCH}')
        self.width = s.model.n_embd
        self.world = o['virtual_gpus']
        self.split = max(1, self.world // o['batch'])  # GPUs per sequence, as train_stages
        self.per = o['batch'] // (self.world // self.split)  # sequences per GPU
        self.schedule = m.stage_schedule(steps * o['batch'], s)
        self.horizons = {phase: horizon for phase, _, _, _, horizon in self.schedule}
        self.first_cut = next((i + 1 for i, b in enumerate(self.schedule) if b[0] == 'cut'), None)
        self.Boost, self.Cut = estimator_classes(m)
        self.turns = Turns(self.split) if self.split > 1 else None
        m.holdout_loss = virtual_holdout_loss(self.world)  # the end merge's scorer (joint_parts calls it by name)
        self.finish = self.end_merge if o['end_merge'] else None
        self.groups, self.drift = [], None
        self.probe_row = heldout[:1].to(device)  # the drift check's sequence: held out, so no momentum holds it
        self.reset()

    def describe(self):
        return dict(name='pr40', file=refs.TRAINERS['pr40'], settings=asdict(self.settings),
                    virtual_gpus=self.world, gpus_per_sequence=self.split,
                    boost_updates=sum(phase == 'boost' for phase, *_ in self.schedule),
                    chain_rows=self.o['rows'] or self.width, row_offset_per_step=self.o['row_offset_per_step'],
                    rows_start=self.o['rows_start'], rows_full_at=self.o['rows_full_at'])

    def estimators(self, phase, seed, ranks):
        cls = self.Boost if phase == 'boost' else self.Cut
        out = [cls(self.model, self.settings, seed, rank, self.world, self.split, None) for rank in ranks]
        for estimator in out:
            estimator.close()  # measure() attaches the hooks of the sequence it measures
            estimator.turns = self.turns
        return out

    def reset(self, phase=None):
        """A fresh model, the phase's optimizer and estimators (the record's first phase unless given)."""
        m, s = self.m, self.settings
        self.phase = phase or self.schedule[0][0]
        self.model = m.make_model(self.device, self.seed, s.model, s.projection_std, s.head_std)
        self.named = list(self.model.named_parameters())
        self.optimizer = m.make_optimizer(self.model, s, self.horizons[self.phase])
        self.labels = rate_labels(self.optimizer, s.lr)
        ranks = self.estimators(self.phase, self.seed, range(self.world))
        self.groups = [ranks[i:i + self.split] for i in range(0, self.world, self.split)]
        if self.o['drift_every']:  # one sequence's GPUs, with their own seed so training draws are untouched
            self.drift = self.estimators('cut', self.seed + DRIFT_SEED, range(self.split))

    def switch(self, phase, phase_step, horizon):
        """Boost -> Cut as train_stages: a new optimizer for the phase, the state handed over, new estimators."""
        optimizer = self.m.make_optimizer(self.model, self.settings, horizon)
        self.m.handoff_optimizer(self.optimizer, optimizer, phase_step)
        self.optimizer, self.phase = optimizer, phase
        self.labels = rate_labels(optimizer, self.settings.lr)
        ranks = self.estimators(phase, self.seed, range(self.world))
        self.groups = [ranks[i:i + self.split] for i in range(0, self.world, self.split)]

    def chain_rows(self, step):
        """(rows, start, cycle) of a Cut update's chain: rows None is the full cycle; start None tiles windows across
        tokens (E7); cycle names the bases held (E6)."""
        o, width = self.o, self.width
        if o['rows'] is not None:
            k = width // o['rows']
            if o['row_offset_per_step']:
                return o['rows'], (step % k) * o['rows'], step // k
            return o['rows'], 0, None
        if o['rows_start'] is not None:
            full_at = o['rows_full_at'] * self.steps
            progress = min(1.0, max(0.0, (step - self.first_cut) / max(1.0, full_at - self.first_cut)))
            rows = o['rows_start'] + (width - o['rows_start']) * progress
            rows = int(rows // ROW_QUANTUM) * ROW_QUANTUM
            return (None if rows >= width else rows), None, None
        return None, 0, None

    def measure(self, group, x, y, total=None):
        """Add each of one sequence's GPUs' float32 gradients (divided by the sequence's token count, as Boost.step)
        to `total` in rank order, as the step's all-reduce; returns the total and each GPU's clean loss."""
        for estimator in group:
            attach(estimator.capture, self.model)
        try:
            if len(group) == 1:
                gradients = [group[0].estimate(x, y)]
            else:
                gradients = self.turns.run([partial(estimator.estimate, x, y) for estimator in group])
        finally:
            for estimator in group:
                estimator.close()
        valid = (y != -1).sum()
        for g in gradients:
            flat = torch.cat([(g[name] / valid).float().reshape(-1) for name, _ in self.named])
            total = flat if total is None else total.add_(flat)
        losses = []
        for estimator in group:
            losses.append(estimator.capture.clean_loss[y != -1].mean())
            estimator.maps.clear()
            for store in (estimator.capture.inputs, estimator.capture.outputs, estimator.capture.blocks):
                store.clear()  # this sequence's activations are no longer needed
        return total, losses

    def update(self, step, batch):
        phase, phase_step, _, _, horizon = self.schedule[step - 1]
        if phase != self.phase:
            self.switch(phase, phase_step, horizon)
        self.m.set_rate(self.optimizer, phase_step, horizon, self.settings.warmdown, self.settings.final)
        fields = dict(phase=phase, phase_step=phase_step, **current_rates(self.optimizer, self.labels))
        if phase == 'cut':
            rows, start, cycle = self.chain_rows(step)
            for group in self.groups:
                for estimator in group:
                    estimator.rows, estimator.start, estimator.cycle = rows, start, cycle
            fields['chain_rows'] = rows or self.width
            if rows is not None and start is not None:
                fields['chain_window'] = [start, start + rows]
        total, losses = None, []
        for g, group in enumerate(self.groups):  # group g: the GPUs of the batch's g-th sequences
            seqs = batch[g * self.per:(g + 1) * self.per]
            total, loss = self.measure(group, seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous(), total)
            losses += loss
        total /= len(self.groups)  # Boost.step: the all-reduced sum over the GPUs, divided by the sequences
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError('Non-finite estimate')
        self.optimizer.zero_grad(set_to_none=True)
        offset = 0
        for _, p in self.named:
            p.grad = total[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        self.optimizer.step()
        return sum(losses) / len(losses), fields  # average_ranks: the all-reduced sum over the GPUs, divided by them

    def warmup(self, batch):
        """As the record: one update of each phase on random tokens, each from a fresh model, so every shape compiles
        before the clock starts; here also one at each partial row count the run's chain will use."""
        steps, seen = [], set()
        for step, (phase, *_) in enumerate(self.schedule, 1):
            shape = (phase, self.chain_rows(step)[0] if phase == 'cut' else None)
            if shape not in seen:
                seen.add(shape)
                steps.append(step)
        for step in steps:
            self.reset(self.schedule[step - 1][0])
            self.update(step, batch)

    @torch.no_grad()
    def diagnose(self, step):
        """E6's drift check, in the Cut phase: a full-cycle estimate on a held-out sequence against the momentum ANVIL
        II steps along, weight * fast + (1 - weight) * slow, and against fast and slow alone, over all block matrices.
        One sequence's estimate is noisy: read the cosines across K, not against an absolute threshold."""
        if self.drift is None or self.phase != 'cut' or step % self.o['drift_every']:
            return {}
        row = self.probe_row
        flat, _ = self.measure(self.drift, row[:, :-1].contiguous(), row[:, 1:].contiguous())
        anvil = self.optimizer.optimizers[0]
        weight = anvil.fast_weight if anvil.t >= anvil.engage else 1.0
        full, fast, slow, offset = [], [], [], 0
        for _, p in self.named:
            if p in anvil.state:
                full.append(flat[offset:offset + p.numel()])
                fast.append(anvil.state[p]['fast'].reshape(-1))
                slow.append(anvil.state[p]['slow'].reshape(-1))
            offset += p.numel()
        full, fast, slow = torch.cat(full), torch.cat(fast), torch.cat(slow)
        return dict(drift=dict(blend=cosine(full, weight * fast + (1 - weight) * slow), fast=cosine(full, fast),
                               slow=cosine(full, slow)))

    def end_merge(self, means):
        """The record's end merge of the trajectory means, on the held-out sequences; returns its result fields."""
        if means.means is None:  # as the record: no means, no merge
            return {}
        start = time.perf_counter()
        out = self.m.end_merge(self.model, means.means, self.heldout, 0, 1)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        return dict(out, merge_seconds=round(time.perf_counter() - start, 2))


def run(spec, run_dir):
    o = cut_options(spec.get('args', {}))
    return train(spec, run_dir, o, lambda device, seed, steps, heldout: CutChain(o, device, seed, steps, heldout),
                 method='Boost -> Cut chain (PR #40)')

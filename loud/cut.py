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
  - E7 (rows_start, rows_full_at; rows_tiled for a fixed count): the rows per token grow linearly from rows_start at
    the first Cut update to 512 at the fraction rows_full_at of the run's updates, in steps of 64 (so few chain shapes
    compile, all in the warmup). Token t reads the window that starts at (t * rows) mod 512 of fresh bases:
    consecutive tokens tile the cycle, so every row is read in every update. At the mid sites the rows measure the
    residual of the identity path's guess (the top stream's credit carried down the residual scalars, H6's
    credit_dump guess): the credit is the guess plus the decoded residual, a control variate whose variance shrinks
    with the guess's error. That is what makes few rows cheap while the guess is good, the plan's premise for E7; at the
    full cycle it is the record's credit exactly. The embedding site has no guess (PR #26 measures it with its own
    cycle) and keeps the plain scaled estimate.
  - E7's self-check (rows_check): part 0 also measures CHECK_ROWS rows past each mid-site window. The estimate predicts
    them by the guess alone (they are orthogonal to the window), so their relative error says how good the guess is
    where nothing was measured; above rows_check, the rows rise by ROW_QUANTUM for the rest of the run.
E11 (target_damping): for each c_proj writer (attn.c_proj writes block i's mid stream, mlp.c_proj its output), four
moves of its output stream along minus its own credit (the record's: the measured mid credit plus path credits;
mlp.c_proj's the next mid's carried back), at 2, 4, 8 and 16 chain probe lengths (chain_sigma x RMS x sqrt(width)),
scored by the chain's kernel (its from_output path for the MLP's output). The runway is how far the token's loss keeps
falling, the target move D the move there; each writer's gradient becomes minus the batch's ridge fit
D^T A (A^T A + lam I)^-1 to its inputs A, lam = target_damping x mean(diag A^T A). Part 0 of a sequence runs them.
E8 (curvature_damping): in the Cut phase, ANVIL II's step along each block matrix's tracked directions is scaled right
after its orthogonalization (loud/curvature.py's 'step'), the noise taken from the sequences' estimates.
Differences from PR #40, all deliberate: the job's seed replaces the module's SEED (seed 0 is the record's); the shared
loop (loud/train.py): H4, untimed diagnostics, official scoring in a fresh process; the means are updated in the
loop's untimed pause instead of the timed region (a 9-way lerp every 5 updates).
"""
from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import asdict
from functools import partial
import threading

import torch

from . import checkpoints, refs
from .anvil import shaped_optimizer
from .curvature import CurvatureDamping
from .identity import DRIFT_SEED, attach, cosine
from .options import CUT_SWITCH, PRESETS, ROW_QUANTUM, cut_options
from .train import current_rates, rate_labels, train

CHECK_ROWS = 8          # E7's self-check: rows past each mid-site window that part 0 also measures, per token
RUNWAY = (2, 4, 8, 16)  # E11: the runway probes' distances, in chain probe lengths
RUNWAY_EVERY = 10       # E11: Cut updates per logged runway distribution


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
        # each part runs in a copy of the caller's context: torch >= 2.13 keeps Dynamo's config (the raised recompile
        # limit) per context, so a bare pool thread would compile under the defaults
        futures = [self.pool.submit(contextvars.copy_context().run, self._part, p, job)
                   for p, job in enumerate(jobs) if p]
        try:
            first = self._part(0, jobs[0])
        except BaseException as exc:  # also an interrupt while waiting: wake the other parts so they stop
            self._fail(exc)
        finally:
            others = [f.exception() or f.result() for f in futures]  # waits for every part, failed or not
        if self.failed is not None:  # the first failure, wherever it happened (the others only followed it)
            raise self.failed
        return [first] + others

    def group_sum(self, part, tensor):
        with self.lock:
            if any(self.done):  # a part that has finished never meets this collective
                self.failed = RuntimeError('the parts of a sequence reached different collectives')
                self.lock.notify_all()
                raise self.failed
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
        rows = None      # rows per token at each chain site, the sequence's GPUs together (None: the full cycle)
        start = 0        # E6: every token's window starts here; None: E7's windows, tiled across tokens
        cycle = None     # E6: the cycle whose bases are held (None: fresh bases each update)
        check = 0        # E7's self-check: rows past each mid-site window that part 0 also measures (0: off)
        targets = False  # E11: part 0 measures each c_proj writer's runway and keeps its fit's statistics

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.held = (None, {})  # (cycle, {site: basis})
            self.top = self.extra = self.inputs = None  # this estimate's top credit, path credits, chain inputs
            self.misses = self.fit = None  # E7's self-check sums and E11's statistics, for the trainer to collect

        @torch.no_grad()
        def estimate(self, x, y):
            gradients = super().estimate(x, y)
            if self.targets and self.part == 0:
                self.fit = self.runway_fit()
            self.top = self.extra = self.inputs = None
            return gradients

        def probe(self, x, y, mix=None, probed=None):
            """The record's probe; Cut calls it once, for the top stream's credit (E7's guess carries it down)."""
            self.top = super().probe(x, y, mix, probed)
            return self.top

        def path_credits(self, chain, targets, raw_values=None):
            """The record's path credits, kept for E11: its writers' credits add them, as the record's gradients do."""
            self.extra, inputs = super().path_credits(chain, targets, raw_values)
            return self.extra, inputs

        def chain_credits(self, e, x0, mids, outs, routes, y):
            if self.targets:
                self.inputs = (e, x0, mids, outs, routes, y)
            if self.rows is None:
                return super().chain_credits(e, x0, mids, outs, routes, y)
            return self.partial_chain_credits(e, x0, mids, outs, routes, y)

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

        def guesses(self):
            """The identity path's credit at each mid site: the top stream's credit carried down the residual scalars
            (H6's credit_dump guess, which E7's diagnostic scores)."""
            lam, carried, out = self.model.resid_lambdas.float(), self.top, {}
            for i in reversed(range(self.model.config.n_layer)):
                out[i] = carried
                carried = lam[i] * carried
            return out

        def chain_pass(self, e, x0, mids, outs, routes, y, probes):
            """The record's chain_credits loop (its kernels, token chunks and own-query context) over `probes`:
            (site, from_output, moves), moves(sl) giving a token slice's (move [t, P, W], scale [t, P]). Returns each
            probe's kernel outputs [n, P]: each token's loss change along each move, divided by its scale."""
            model, s = self.model, self.s
            vocab = model.config.vocab_size
            weight = model.lm_head.weight.float()[:vocab]
            weight_t = weight.T.to(torch.bfloat16).contiguous()
            w_fc = [blk.mlp.c_fc.weight.to(s.chain_dtype) for blk in model.transformer.h]
            w_out_t = [blk.mlp.c_proj.weight.to(s.chain_dtype).T.contiguous() for blk in model.transformer.h]
            lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
            xL = outs[-1]
            base = xL @ weight.T
            target_w = weight[y.clamp(min=0)]
            n, chunk, out = e.shape[0], s.chain_tokens, [None] * len(probes)
            length = self.capture.length if s.own_query else n
            for start in range(0, n, length):
                own = None
                if s.own_query:
                    own = self.own_query_inputs(start // length)
                    if s.query_keys:
                        own = self.strong_keys(own, s.query_keys)
                for t0 in range(start, start + length, chunk):
                    t1 = min(t0 + chunk, start + length)
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
                    for k, (site, from_output, moves) in enumerate(probes):
                        move, scale = moves(sl)
                        derivative = kernel(
                            site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl],
                            [r[0][sl] for r in routes], [r[1][sl] for r in routes], lam_r, lam_0, selected_fc,
                            selected_out, weight_t, base[sl], target_w[sl], y[sl], own_sl, from_output=from_output,
                            packed=True)
                        if out[k] is None:
                            out[k] = torch.empty(n, derivative.shape[1], device=self.device)
                        out[k][sl] = derivative
                    del selected_fc, selected_out
                del own_sl, own
            return out

        def window_moves(self, H, steps, index, prev, x0, sl):
            """One site's moves for a token slice: the record's (steps times each row; on part 0 of a mid site, the
            symmetric moves of both mixing scalars), each token's rows of H picked by `index`."""
            move = steps[sl].unsqueeze(-1) * H[index[sl]]
            scale = steps[sl].expand(-1, move.shape[1])
            if prev is not None:  # their scales do not depend on the rows (the record's assume 2 or more)
                h, prev, z = self.s.scalar_step, prev[sl].unsqueeze(1), x0[sl].unsqueeze(1)
                move = torch.cat([move, h * prev, h * z, -h * prev, -h * z], 1)
                scale = torch.cat([scale, scale.new_full((len(scale), 2), h), scale.new_full((len(scale), 2), -h)], 1)
            return move, scale

        def partial_chain_credits(self, e, x0, mids, outs, routes, y):
            """PR #40's chain_credits at `rows` rows per token per site: each token reads its rows of the site's basis
            (this GPU's share of its window) and its credit is decoded over them, scaled by width / rows. With tiled
            windows (E7), the mid sites measure the identity path's guess's residual: the credit is the guess plus the
            decoded residual (at the full cycle, the record's credit either way), and on part 0 the self-check also
            measures `check` rows past each window, where the estimate predicts the guess's own loss changes."""
            s, L = self.s, self.model.config.n_layer
            n, first = e.shape[0], self.part == 0
            index = self.row_index(n)
            P = index.shape[1]
            check = self.check if first else 0
            past = (index[:, :1] + self.rows + torch.arange(check, device=self.device)) % self.width
            guesses = self.guesses() if self.start is None else {}
            sites, bases, probes = range(-1, L), [], []
            for site in sites:
                H = self.site_basis(site)
                probed = e if site < 0 else mids[site]
                steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                rows = torch.cat([index, past], 1) if site in guesses else index
                prev = (x0 if site == 0 else outs[site - 1]) if site >= 0 and first else None
                bases.append(H)
                probes.append((site, False, partial(self.window_moves, H, steps, rows, prev, x0)))
            measured = self.chain_pass(e, x0, mids, outs, routes, y, probes)
            credits, self.misses = {}, {}
            for site, H, d in zip(sites, bases, measured):
                guess = guesses.get(site)
                credit = torch.empty(n, self.width, device=self.device)
                for t0 in range(0, n, s.chain_tokens):
                    sl = slice(t0, t0 + s.chain_tokens)
                    directions = H[index[sl]]  # [tokens, P, width]
                    residual = d[sl, :P] if guess is None else (
                        d[sl, :P] - torch.einsum('tw,tpw->tp', guess[sl], directions))
                    credit[sl] = torch.einsum('tp,tpw->tw', residual, directions) / self.rows
                if guess is not None and first:
                    credit += guess
                if guess is not None and check:
                    miss = d[:, P:P + check] - torch.einsum('tw,tjw->tj', guess, H[past])
                    self.misses[site] = torch.stack([miss.square().sum(), d[:, P:P + check].square().sum()])
                credits['emb' if site < 0 else site] = self.group_sum(credit)
                if site >= 0:
                    lam = (d[:, -4:-2] + d[:, -2:]).sum(0) / 2 if first else torch.zeros(2, device=self.device)
                    credits['lam', site] = self.group_sum(lam)
            return credits

        @staticmethod
        def runway_moves(distances, unit, sl):
            """E11's moves for a token slice: each distance times the token's probe-length step, loss changes
            unscaled."""
            return distances[:, None] * unit[sl].unsqueeze(1), unit.new_ones(unit[sl].shape[0], len(distances))

        def runway_fit(self):
            """E11 on this sequence: for each c_proj writer, four moves of its output stream along minus its credit
            (the record's: the mid credit plus path credits for attn.c_proj, the next mid's carried back for
            mlp.c_proj), at RUNWAY times the chain's probe length; the runway, where the token's loss stops falling;
            the target move D there; and the fit's statistics A^T A and D^T A, A the writer's inputs."""
            e, x0, mids, outs, routes, y = self.inputs
            chain, top, s = self.measured['chain'], self.measured['g_top'], self.s
            lam_r, L = self.model.resid_lambdas.float(), self.model.config.n_layer
            writers = []  # (name, site, from_output, its output stream, its credit)
            for i in range(L):
                extra = 0 if self.extra is None else self.extra[i]
                after = top if i == L - 1 else lam_r[i + 1] * chain[i + 1]
                writers.append((f'transformer.h.{i}.attn.c_proj', i, False, mids[i], chain[i] + extra))
                writers.append((f'transformer.h.{i}.mlp.c_proj', i, True, outs[i], after + extra))
            distances = torch.tensor(RUNWAY, dtype=torch.float32, device=self.device)
            probes, unit_moves = [], []
            for _, site, from_output, stream, credit in writers:
                rms = stream.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                direction = credit / credit.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                unit = -s.chain_sigma * rms * self.width ** .5 * direction  # one chain probe length along -credit
                unit_moves.append(unit)
                probes.append((site, from_output, partial(self.runway_moves, distances, unit)))
            changes = self.chain_pass(e, x0, mids, outs, routes, y, probes)
            stats, counts = {}, {}
            for (name, *_), unit, change in zip(writers, unit_moves, changes):
                walked = runway(change)
                target = torch.cat([distances.new_zeros(1), distances])[walked, None] * unit
                inputs = self.capture.inputs[name].flatten(0, 1).float()
                stats[name] = (inputs.T @ inputs, target.T @ inputs)
                counts[name] = torch.bincount(walked, minlength=len(RUNWAY) + 1)
            return stats, counts

    return Boost, Cut


def runway(changes):
    """E11: how many of the RUNWAY distances each token's loss keeps falling over, from its loss changes [n, 4] at
    them (0 if the first already rises)."""
    falling = torch.cat([changes[:, :1] < 0, changes[:, 1:] < changes[:, :-1]], 1)
    return falling.int().cumprod(1).sum(1)


def ridge_fit(ata, dta, damping):
    """E11's damped least-squares fit of a writer to its targets from the batch's sums A^T A and D^T A:
    dW = D^T A (A^T A + lam I)^-1, lam = damping x mean(diag A^T A), solved in float64."""
    ridge = damping * ata.diagonal().mean().double()
    system = ata.double() + ridge * torch.eye(len(ata), device=ata.device, dtype=torch.float64)
    return torch.linalg.solve(system, dta.double().T).T.float()


def record_settings(m, preset, batch):
    """PR #40's Settings for a model preset: the record's at '38m'; smaller presets (smoke tests only) scale the
    budgets that the record ties to the width (input-K pairs = width, future draws = width / 4, routed units = hidden
    / 4) and keep at most their blocks as future sources."""
    n_layer, width = PRESETS[preset]
    config = m.model_config()
    config.n_layer, config.n_head, config.n_kv_head, config.n_embd = n_layer, width // 64, width // 64, width
    return m.Settings(model=config, boost_batch=batch, cut_batch=batch, key_path_draws=width,
                      future_draws=width // 4, route_k=width, future_layers=min(4, n_layer))


def merge_loss(ranks, one_batch):
    """The record's holdout_loss as `ranks` GPUs compute it: each rank its rows[rank::ranks] in batches of 8, the
    all-reduce a float64 sum; with one_batch (merge_batch), all rows in one forward, which changes only the bf16
    rounding of the losses. Its other lines are the record's (with its CUDA autocast, a no-op on a CPU)."""

    @torch.no_grad()
    def loss(model, rows):
        device = next(model.parameters()).device
        total = torch.zeros(2, dtype=torch.float64, device=device)
        for part in [rows] if one_batch else [rows[r::ranks] for r in range(ranks)]:
            for batch in part.split(len(part) if one_batch else 8):
                batch = batch.to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = model(batch[:, :-1].contiguous(), batch[:, 1:].contiguous(), loss_reduction='sum')
                total[0] += loss.double()
                total[1] += batch[:, 1:].numel()
        return (total[0] / total[1]).item()

    return loss


class SequenceEstimates:
    """Each sequence's estimate (its GPUs' parts summed), stacked per parameter as H2's PerSequenceGrads.stack: E8's
    noise along its tracked directions."""

    def __init__(self, slots, flats):
        self.slots, self.flats = slots, flats

    def stack(self, name):
        offset, shape = self.slots[name]
        return torch.stack([flat[offset:offset + shape.numel()].view(shape) for flat in self.flats])


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
        self.finish = self.end_merge if o['end_merge'] else None
        self.groups, self.drift = [], None
        self.probe_row = heldout[:1].to(device)  # the drift check's sequence: held out, so no momentum holds it
        self.reset()
        self.planned = self.planned_rows()
        self.graphs = self.chain_graphs(self.planned)
        self.recompile_limit = max(64, 2 * len(self.graphs))  # the record's 64 holds only its own 17 graphs
        raise_recompile_limit(self.recompile_limit)

    def describe(self):
        return dict(name='pr40', file=refs.TRAINERS['pr40'], settings=asdict(self.settings),
                    virtual_gpus=self.world, gpus_per_sequence=self.split,
                    boost_updates=sum(phase == 'boost' for phase, *_ in self.schedule),
                    chain_rows=self.o['rows'] or self.width, row_offset_per_step=self.o['row_offset_per_step'],
                    rows_tiled=self.o['rows_tiled'] or self.o['rows_start'] is not None,
                    rows_start=self.o['rows_start'], rows_full_at=self.o['rows_full_at'],
                    rows_check=self.o['rows_check'], target_damping=self.o['target_damping'],
                    curvature_damping=self.o['curvature_damping'], merge_batch=self.o['merge_batch'],
                    reverse_sum=self.o['reverse_sum'], chain_graphs=len(self.graphs),
                    recompile_limit=self.recompile_limit)

    def estimators(self, phase, seed, ranks):
        cls = self.Boost if phase == 'boost' else self.Cut
        out = [cls(self.model, self.settings, seed, rank, self.world, self.split, None) for rank in ranks]
        for estimator in out:
            estimator.close()  # measure() attaches the hooks of the sequence it measures
            estimator.turns = self.turns
        return out

    def make_optimizer(self, phase, horizon):
        """The phase's optimizers; in the Cut phase with E8 on, ANVIL II with a new curvature shaper for this model,
        acting right after its orthogonalization (curvature.py's damp_where 'step')."""
        self.shaper = None
        if phase == 'cut' and self.o['curvature_damping']:
            self.shaper = CurvatureDamping(dict(self.o, damp_where='step'), self.model, self.settings.anvil_lr)
            return shaped_optimizer(self.m, self.model, self.settings, horizon, self.shaper, 'BatchedAnvil')
        return self.m.make_optimizer(self.model, self.settings, horizon)

    def reset(self, phase=None):
        """A fresh model, the phase's optimizer and estimators (the record's first phase unless given)."""
        m, s = self.m, self.settings
        self.phase = phase or self.schedule[0][0]
        self.model = m.make_model(self.device, self.seed, s.model, s.projection_std, s.head_std)
        self.named = list(self.model.named_parameters())
        offsets = [0]
        for _, p in self.named:
            offsets.append(offsets[-1] + p.numel())
        self.slots = {name: (offset, p.shape) for (name, p), offset in zip(self.named, offsets)}
        self.optimizer = self.make_optimizer(self.phase, self.horizons[self.phase])
        self.labels = rate_labels(self.optimizer, s.lr)
        ranks = self.estimators(self.phase, self.seed, range(self.world))
        self.groups = [ranks[i:i + self.split] for i in range(0, self.world, self.split)]
        if self.o['drift_every']:  # one sequence's GPUs, with their own seed so training draws are untouched
            self.drift = self.estimators('cut', self.seed + DRIFT_SEED, range(self.split))
        self.floor, self.runways = 0, {}  # E7's self-check: rows it has raised the chain to; E11: runway counts

    def switch(self, phase, phase_step, horizon):
        """Boost -> Cut as train_stages: a new optimizer for the phase, the state handed over, new estimators."""
        optimizer = self.make_optimizer(phase, horizon)
        self.m.handoff_optimizer(self.optimizer, optimizer, phase_step)
        self.optimizer, self.phase = optimizer, phase
        self.labels = rate_labels(optimizer, self.settings.lr)
        ranks = self.estimators(phase, self.seed, range(self.world))
        self.groups = [ranks[i:i + self.split] for i in range(0, self.world, self.split)]

    def chain_rows(self, step):
        """(rows, start, cycle) of a Cut update's chain: rows None is the full cycle; start None tiles windows across
        tokens around the identity path's guess (E7); cycle names the bases held (E6)."""
        o, width = self.o, self.width
        if o['rows'] is not None and not o['rows_tiled']:
            k = width // o['rows']
            if o['row_offset_per_step']:
                return o['rows'], (step % k) * o['rows'], step // k
            return o['rows'], 0, None
        if o['rows'] is not None:
            rows = o['rows']
        elif o['rows_start'] is not None:
            full_at = o['rows_full_at'] * self.steps
            progress = min(1.0, max(0.0, (step - self.first_cut) / max(1.0, full_at - self.first_cut)))
            rows = int((o['rows_start'] + (width - o['rows_start']) * progress) // ROW_QUANTUM) * ROW_QUANTUM
        else:
            return None, 0, None
        rows = max(rows, self.floor)  # E7's self-check only raises the rows
        return (None if rows >= width else rows), None, None

    def measure(self, group, x, y, total=None, sequences=None):
        """Add each of one sequence's GPUs' float32 gradients (divided by the sequence's token count, as Boost.step)
        to `total` in rank order, as the step's all-reduce (reversed with reverse_sum, the calibration of its
        rounding); returns the total and each GPU's clean loss. With a `sequences` list, also appends the
        sequence's own estimate, its parts summed (E8's noise)."""
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
        sequence = None
        for g in gradients[::-1] if self.o['reverse_sum'] else gradients:
            flat = torch.cat([(g[name] / valid).float().reshape(-1) for name, _ in self.named])
            if sequences is not None:
                sequence = flat.clone() if sequence is None else sequence.add_(flat)
            total = flat if total is None else total.add_(flat)
        if sequences is not None:
            sequences.append(sequence)
        losses = []
        for estimator in group:
            losses.append(estimator.capture.clean_loss[y != -1].mean())
            estimator.maps.clear()
            estimator.measured = {}  # Cut's chain credits and targets, kept by the record until the next estimate
            for store in (estimator.capture.inputs, estimator.capture.outputs, estimator.capture.blocks):
                store.clear()  # this sequence's activations are no longer needed
        return total, losses

    def update(self, step, batch):
        phase, phase_step, _, _, horizon = self.schedule[step - 1]
        if phase != self.phase:
            self.switch(phase, phase_step, horizon)
        self.m.set_rate(self.optimizer, phase_step, horizon, self.settings.warmdown, self.settings.final)
        fields = dict(phase=phase, phase_step=phase_step, **current_rates(self.optimizer, self.labels))
        rows = check = None
        targets = phase == 'cut' and bool(self.o['target_damping'])
        if phase == 'cut':
            rows, start, cycle = self.chain_rows(step)
            check = CHECK_ROWS if self.o['rows_check'] and start is None and rows is not None else 0
            for group in self.groups:
                for estimator in group:
                    estimator.rows, estimator.start, estimator.cycle = rows, start, cycle
                    estimator.check, estimator.targets = check, targets
            fields['chain_rows'] = rows or self.width
            if rows is not None and start is not None:
                fields['chain_window'] = [start, start + rows]
        sequences = [] if self.shaper is not None else None
        total, losses, misses, fits = None, [], {}, {}
        order = list(enumerate(self.groups))  # group g: the GPUs of the batch's g-th sequences
        for g, group in order[::-1] if self.o['reverse_sum'] else order:
            seqs = batch[g * self.per:(g + 1) * self.per]
            total, loss = self.measure(group, seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous(), total, sequences)
            losses += loss
            if phase == 'cut':
                self.collect(group, misses, fits)
        total /= len(self.groups)  # Boost.step: the all-reduced sum over the GPUs, divided by the sequences
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError('Non-finite estimate')
        if targets:
            self.fit_targets(total, fits)
        if check:
            fields['rows_check'] = self.self_check(rows, misses)
        self.optimizer.zero_grad(set_to_none=True)
        for name, p in self.named:
            offset, shape = self.slots[name]
            p.grad = total[offset:offset + p.numel()].view(shape).to(p.dtype)
        if self.shaper is not None:
            estimates = SequenceEstimates(self.slots, sequences)
            self.shaper.before_step(step - self.first_cut + 1, estimates)  # refreshes from the first Cut update
        self.optimizer.step()
        if self.shaper is not None:
            self.shaper.after_step(self.model, batch, estimates)
        return sum(losses) / len(losses), fields  # average_ranks: the all-reduced sum over the GPUs, divided by them

    def collect(self, group, misses, fits):
        """Add what one sequence's part 0 measured for E7's self-check and E11 to the batch's sums, in rank order as an
        all-reduce would; E11's runway counts accumulate until diagnose logs them."""
        for estimator in group:
            for site, miss in (estimator.misses or {}).items():
                misses[site] = misses[site] + miss if site in misses else miss
            if estimator.fit is not None:
                stats, counts = estimator.fit
                for name, (ata, dta) in stats.items():
                    fits[name] = (fits[name][0] + ata, fits[name][1] + dta) if name in fits else (ata, dta)
                for name, count in counts.items():
                    self.runways[name] = self.runways.get(name, 0) + count
            estimator.misses = estimator.fit = None

    def fit_targets(self, total, fits):
        """E11: each c_proj writer's fit to the batch's per-token targets (ridge_fit); minus the fit replaces its
        gradient (ANVIL II orthogonalizes it)."""
        for name, (ata, dta) in fits.items():
            offset, shape = self.slots[name + '.weight']
            total[offset:offset + shape.numel()] = -ridge_fit(ata, dta, self.o['target_damping']).reshape(-1)

    def self_check(self, rows, misses):
        """E7's self-check over the batch: the relative error of the estimate's prediction on the CHECK_ROWS rows past
        each mid-site window (the guess's own: the rows are orthogonal to the window), per site and pooled. Above
        rows_check, the chain's rows rise to the next step of ROW_QUANTUM for the rest of the run."""
        error = (sum(m[0] for m in misses.values()) / sum(m[1] for m in misses.values()).clamp_min(1e-30)).sqrt().item()
        record = dict(error=round(error, 4), sites={site: round((m[0] / m[1].clamp_min(1e-30)).sqrt().item(), 4)
                                                     for site, m in misses.items()})
        if error > self.o['rows_check']:
            self.floor = (rows // ROW_QUANTUM + 1) * ROW_QUANTUM
            record['raised_to'] = min(self.floor, self.width)
        return record

    def planned_rows(self):
        """The chain's row counts over the Cut phase (None: the full cycle), with every count E7's self-check may
        raise them to (the next multiples of ROW_QUANTUM) and the drift check's full cycle."""
        rows = {self.chain_rows(step)[0] for step, (phase, *_) in enumerate(self.schedule, 1) if phase == 'cut'}
        if self.o['rows_check'] and rows - {None}:
            low = min(rows - {None})
            rows |= set(range((low // ROW_QUANTUM + 1) * ROW_QUANTUM, self.width, ROW_QUANTUM)) | {None}
        if self.o['drift_every'] and rows:
            rows.add(None)
        return rows

    def chain_graphs(self, rows):
        """chain_chunk's graphs for these row counts, as Dynamo specializes it: one per (site, moves per token,
        from_output). Part 0 adds the mixing scalars' 4 moves at the mid sites (and E7's self-check rows); the other
        parts do not; E11 adds its 4 runway moves at every block's mid and output streams."""
        L = self.settings.model.n_layer
        check = CHECK_ROWS if self.o['rows_check'] else 0
        keys = set()
        for r in rows:
            share = (r or self.width) // self.split
            keys.add((-1, share, False))
            keys |= {(site, share + 4 + (check if r is not None else 0), False) for site in range(L)}
            if self.split > 1:
                keys |= {(site, share, False) for site in range(L)}
        if self.o['target_damping']:
            keys |= {(site, len(RUNWAY), out) for site in range(L) for out in (False, True)}
        return keys

    def warmup(self, batch):
        """As the record: one update of each phase on random tokens, each from a fresh model, so every shape compiles
        before the clock starts; here one at each row count the chain may use, and a drift check. Fails unless the
        chain's kernel compiled every planned graph (below Dynamo's limit): a missing shape would run eagerly."""
        config = torch._dynamo.config
        if compiled_graphs(self.m.chain_chunk_eager) is None and hasattr(config, 'fail_on_recompile_limit_hit'):
            config.fail_on_recompile_limit_hit = True  # this torch cannot count the graphs: fail at the limit instead
        cut = [step for step, (phase, *_) in enumerate(self.schedule, 1) if phase == 'cut']
        runs = [(1, 0)] if self.schedule[0][0] == 'boost' else []
        runs += [(cut[0], self.width if r is None else r) for r in self.planned] if cut else []
        for step, floor in runs:  # the floor sets a tiled chain's rows at the first Cut update; E6 ignores it
            self.reset(self.schedule[step - 1][0])
            self.floor = floor
            self.update(step, batch)
        if self.drift is not None and cut:
            self.drift_check()
        compiled = compiled_graphs(self.m.chain_chunk_eager)
        if compiled is not None and not torch._dynamo.config.disable:
            print(f'chain_chunk: {compiled} graphs compiled; {len(self.graphs)} planned, limit '
                  f'{self.recompile_limit}', flush=True)
            if compiled < len(self.graphs) or compiled >= self.recompile_limit:
                raise RuntimeError(f'chain_chunk compiled {compiled} of its {len(self.graphs)} planned graphs (limit '
                                   f'{self.recompile_limit}): the missing shapes would run eagerly')

    @torch.no_grad()
    def diagnose(self, step):
        """Untimed records in the Cut phase: E6's drift check every drift_every updates, E11's runway distribution
        every RUNWAY_EVERY and E8's damping summary after each refresh."""
        if self.phase != 'cut':
            return {}
        record = self.shaper.report(step) if self.shaper is not None else {}
        if self.runways and (step - self.first_cut + 1) % RUNWAY_EVERY == 0:
            record['runway'] = {'.'.join(name.split('.')[2:]): [round(c / count.sum().item(), 4)
                                                                for c in count.tolist()]
                                for name, count in self.runways.items()}
            self.runways = {}
        if self.drift is not None and step % self.o['drift_every'] == 0:
            record['drift'] = self.drift_check()
        return record

    def drift_check(self):
        """E6's drift check: a full-cycle estimate on a held-out sequence against ANVIL II's fast and slow momenta,
        over all block matrices. PR #40 steps along the fast one alone (fast_weight 1.0), so there is no blend to
        report. One sequence's estimate is noisy: read the cosines across K, not against an absolute threshold."""
        row = self.probe_row
        flat, _ = self.measure(self.drift, row[:, :-1].contiguous(), row[:, 1:].contiguous())
        anvil = self.optimizer.optimizers[0]
        full, fast, slow = [], [], []
        for name, p in self.named:
            if p in anvil.state:
                offset, shape = self.slots[name]
                full.append(flat[offset:offset + shape.numel()])
                fast.append(anvil.state[p]['fast'].reshape(-1))
                slow.append(anvil.state[p]['slow'].reshape(-1))
        full, fast, slow = torch.cat(full), torch.cat(fast), torch.cat(slow)
        return dict(fast=cosine(full, fast), slow=cosine(full, slow))

    def end_merge(self, means, heldout):
        """The record's end merge (loud.checkpoints.end_merge), each candidate scored as the 8-GPU run scores it
        (merge_loss), or in one batch with merge_batch."""
        return checkpoints.end_merge(self.m, self.model, means, heldout, self.steps,
                                     merge_loss(self.world, self.o['merge_batch']))


def raise_recompile_limit(limit):
    """Dynamo's graphs per function raised to `limit` (the record sets 64), and its total to match, under the
    current and the older names."""
    config = torch._dynamo.config
    for name in ('recompile_limit', 'cache_size_limit'):
        if hasattr(config, name):
            setattr(config, name, max(getattr(config, name), limit))
    for name in ('accumulated_recompile_limit', 'accumulated_cache_size_limit'):
        if hasattr(config, name):
            setattr(config, name, max(getattr(config, name), limit + 256))


def compiled_graphs(function):
    """Dynamo's compiled graphs of a function (a debug API; None where this torch lacks it)."""
    try:
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    except ImportError:
        return None
    return len(_debug_get_cache_entry_list(function.__code__))


def run(spec, run_dir):
    o = cut_options(spec.get('args', {}))
    return train(spec, run_dir, o, lambda device, seed, steps, heldout: CutChain(o, device, seed, steps, heldout),
                 method='Boost -> Cut chain (PR #40)')

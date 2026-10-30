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

The probe-budget flags (v2's H5: the winners of the H8 experiments, confirmed on the real chain) change only the Cut
phase's chain (chain_credits: the embedding and the 8 mid streams). Boost, the head-only top probe (probe()) and the
key, gate, query and future probes keep their budgets. With rows = m below the full 512, a token's credit at a site is
decoded from m rows of the site's freshly signed Hadamard basis and scaled by 512 / m. That is unbiased: the basis's
random column signs make the outer products of any set of its rows average to the set's size times the identity. The
full cycle runs the record's chain_credits itself.
  - row_assign 'time': every token of update s reads rows [(s mod K) m, (s mod K + 1) m), K = 512 / m, so the rows
    rotate across updates; 'token': token t reads rows from (t mod K) m, so every update covers the full basis across
    its tokens.
  - prior 'identity' (E7): at the mid sites the rows measure the residual of the identity path's guess (the top
    stream's credit carried down the residual scalars, as H6's credit_dump guesses it): the credit is the guess plus
    (1 / m) sum_j (v_j . (g - guess)) v_j, unbiased like the plain estimate, with a variance that shrinks with the
    guess's error. The embedding site S0 has no guess (PR #26 measures it with its own cycle) and keeps the plain
    estimate.
  - row_budget (E13's hook; 'ngram' is E13's rule, NGramBudget): a strategy from ROW_BUDGETS gives each token its
    own row count up to `rows` (token
    windows of its count); the rows past a token's count are left out of its decode. The chain still evaluates them,
    so a budget changes the estimate, not yet the chain's cost: grouping tokens by count is the step that saves time.
  - deep_readout (E5, "one launch, eight readings"): S0 keeps its own cycle; S1's full cycle runs through a copy of
    the record's chain kernel (sites_kernel) that also returns each move's change at every later mid site, where the
    record's own cycles would move the stream; for each token and later site the 512 changes and loss changes form a
    512 x 512 system, solved in float32 with a ridge toward the identity path's guess (1e-3 times the mean diagonal
    of A^T A, v2's), and the cycles at S2..S8 are not run. The changes are stored in bf16. The mixing scalars keep the
    record's 4 central moves at every mid site (the solved credit is least sure along the stream itself, the
    direction the residual scalar's move takes: on the CPU test model, credit cosine 0.995 but its dot product with
    the stream off by 3x).
    A token's system needs all its moves but the tokens are independent, so the GPUs of a sequence split S1's tokens
    (and the scalar moves') instead of its moves. The bases are drawn as the record draws them, the skipped sites' too.
Differences from PR #40, all deliberate: the job's seed replaces the module's SEED (seed 0 is the record's); the shared
loop (loud/train.py): H4, untimed diagnostics, official scoring in a fresh process; the means are updated in the
loop's untimed pause instead of the timed region (a 9-way lerp every 5 updates).
"""
from concurrent.futures import ThreadPoolExecutor
import contextvars
from dataclasses import asdict
from functools import partial, wraps
import threading
import time

import torch

from . import checkpoints, refs
from .identity import attach
from .options import PRESETS, cut_options
from .train import current_rates, rate_labels, train

# E13's per-token row budgets, by name (the row_budget option). ROW_BUDGETS[name](o, vocab) is built by every reset,
# so the run's own comes after the warmup. Its rows(capture) returns each token's row count [n] (a power of two up to
# o['rows']) for the sequence the capture holds (capture.x, capture.y and capture.clean_loss, the clean forward's
# per-token loss), and its observe(batch) sees every update's batch after the update, so a statistic such as E13's
# n-gram counts learns only from data already trained on.
class NGramBudget:
    """E13's rule on the chain ('ngram'): a token whose loss under the streaming Witten-Bell n-gram model of the
    training stream so far (loud/ngram.py) and under the network (the clean forward) are both at most budget_theta
    nats reads LOW_ROWS rows, the rest `rows` (the full cycle at 512), as H8's budget_theta."""

    def __init__(self, o, vocab):
        from .h8 import LOW_ROWS
        from .ngram import NGram
        self.theta, self.high, self.low = o['budget_theta'], o['rows'], min(LOW_ROWS, o['rows'])
        self.table = NGram(vocab)

    def rows(self, capture):
        x, y = (capture.x, capture.y) if capture.x.dim() == 2 else (capture.x[None], capture.y[None])
        rows = torch.cat([x, y[:, -1:]], 1).cpu().numpy()  # each sequence with its own last target
        ngram = torch.from_numpy(self.table.losses(rows)).float().flatten()
        low = torch.maximum(ngram, capture.clean_loss.flatten().float().cpu()) <= self.theta
        return torch.where(low, self.low, self.high)

    def observe(self, batch):
        self.table.insert(batch.cpu().numpy())


ROW_BUDGETS = {'ngram': NGramBudget}


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


class Profile:
    """Seconds per part of the Cut estimate, the GPU synchronized around each call (profile: true; a timing aid, the
    syncs cost a little). Inclusive: a part's time includes the parts it calls (internals holds query_errors,
    exact_key_errors and exact_values; path_credits holds key_input_credits and gate_input_credits; estimate all)."""
    METHODS = ('estimate', 'probe', 'chain_credits', 'path_credits', 'key_input_credits', 'gate_input_credits',
               'value_credit', 'hidden_credit', 'internals', 'query_errors', 'exact_key_errors', 'exact_values',
               'select_mlp_routes', 'own_query_inputs')

    def __init__(self):
        self.on, self.seconds, self.installed = False, {}, set()

    def wrap(self, name, fn):
        @wraps(fn)
        def timed(*args, **kwargs):
            if not self.on:
                return fn(*args, **kwargs)
            sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
            sync()
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                sync()
                self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - start
        return timed

    def install(self, m, Cut):
        """Time the Cut class's parts, the capture's forward and head readout, and the future probes."""
        for name in self.METHODS:
            setattr(Cut, name, self.wrap(name, getattr(Cut, name)))
        if m not in self.installed:  # module-level: once per process
            self.installed.add(m)
            for name in ('capture', 'head_error'):
                setattr(m.Capture, name, self.wrap(name, getattr(m.Capture, name)))
            m.add_future_gradients = self.wrap('add_future_gradients', m.add_future_gradients)

    def take(self):
        out, self.seconds = {k: round(v, 3) for k, v in sorted(self.seconds.items())}, {}
        return out


PROFILE = Profile()
SITES = None  # E5's kernel (eager, compiled static, compiled dynamic), built with the estimators
CHANGES_DTYPE = torch.bfloat16  # E5: how the changes are stored for the solve
RIDGE = 1e-3  # E5: the solve's ridge toward the identity path's guess, times the mean diagonal of A^T A (v2's)


def sites_kernel(m):
    """E5's kernel: the record's chain_chunk_eager from a mid site (packed units, from the stream, not an MLP output)
    that also returns, for the first `rows` moves, each move's change at every later mid site [m, rows, W] in bf16:
    the stream after that block's residual scaling and own-query response, where the record's cycle at that site
    would move it. Returns (loss changes / scale [m, P], [changes at site + 1, ..., L - 1])."""
    import torch.nn.functional as F

    def eager(site, move, scale, e, x0, mids, xL, idx, hs, lam_r, lam_0, w_fc, w_out_t, weight_t, base,
              target_logit_w, y, own=None, rows=0):
        L, delta, changes = len(mids), move, []
        for b in range(site, L):
            if b > site:
                delta = lam_r[b] * delta
                if own is not None:
                    if own[0] == 'routed':
                        _, pos, inputs, w_q, top_keys, top_values, top_p, rest, w_proj_t, cos_sin = own
                        delta = delta + m.query_change_routed(inputs[b], delta, pos, w_q[b], top_keys[b],
                                                              top_values[b], top_p[b], rest[b], w_proj_t[b],
                                                              cos_sin[0], cos_sin[1])
                    else:
                        _, pos, rels, inputs, w_q, keys, values, w_proj_t, cos, sin, windows = own
                        delta = delta + m.query_change(inputs[b], delta, pos, rels[b], w_q[b], keys[b], values[b],
                                                       w_proj_t[b], cos, sin, windows[b])
                changes.append(delta[:, :rows].to(CHANGES_DTYPE))
            dz = m.norm_difference(mids[b], delta)
            fc, wo = w_fc[b], w_out_t[b]
            dh = torch.bmm(dz.to(fc.dtype), fc.transpose(1, 2)).float()
            h = hs[b].unsqueeze(1)
            up, down = F.relu(h + dh), F.relu(h)
            delta = delta + torch.bmm(((up - down) * (up + down)).to(wo.dtype), wo).float()
        sq = xL.square().sum(-1, keepdim=True)
        dots = torch.einsum('md,mpd->mp', xL, delta)
        inv = torch.rsqrt((sq + 2 * dots + delta.square().sum(-1)) / xL.shape[-1] + m.EPS)
        z = (base.unsqueeze(1) + (delta.to(torch.bfloat16) @ weight_t).float()) * inv.unsqueeze(-1)
        zy = (base.gather(1, y.clamp(min=0).unsqueeze(1)) + (delta * target_logit_w.unsqueeze(1)).sum(-1)) * inv
        moved = torch.exp(15 * torch.tanh(z / 15)).sum(-1).log() - 15 * torch.tanh(zy / 15)
        inv0 = torch.rsqrt(sq / xL.shape[-1] + m.EPS)
        clean = torch.exp(15 * torch.tanh(base * inv0 / 15)).sum(-1, keepdim=True).log() - 15 * torch.tanh(
            base.gather(1, y.clamp(min=0).unsqueeze(1)) * inv0 / 15)
        return (moved - clean) / scale * (y >= 0).unsqueeze(1), changes

    return eager, torch.compile(eager, dynamic=False), torch.compile(eager, dynamic=True)


def solve_credits(changes, derivative, steps, guess):
    """E5: per token, the credit g [W] at one site from its moves' changes there (changes [t, P, W], each from a move of
    `steps` [t, 1] times a basis row) and the loss changes along the rows (derivative [t, P]): the least-squares
    solution of (changes / steps) g = derivative, with RIDGE toward the identity path's guess [t, W], by Cholesky.
    Returns g and each system's status (nonzero: not positive definite), for one check per estimate."""
    with torch.autocast(changes.device.type, enabled=False):
        A = changes.float() / steps.unsqueeze(-1)
        gram = A.transpose(1, 2) @ A
        ridge = RIDGE * gram.diagonal(dim1=-2, dim2=-1).mean(-1)
        gram.diagonal(dim1=-2, dim2=-1).add_(ridge.unsqueeze(-1))
        rhs = (A.transpose(1, 2) @ derivative.unsqueeze(-1)).squeeze(-1) + ridge.unsqueeze(-1) * guess
        factor, status = torch.linalg.cholesky_ex(gram)  # the ridge makes it positive definite; no host sync
        return torch.cholesky_solve(rhs.unsqueeze(-1), factor).squeeze(-1), status


def estimator_classes(m):
    """PR #40's Boost and Cut with group_sum through Turns, and Cut's chain with the probe-budget flags."""

    global SITES
    SITES = sites_kernel(m)

    class Parts:
        turns = None  # the trainer's Turns; group_sum needs it only when a sequence has several parts

        def group_sum(self, tensor):
            return tensor if self.split == 1 else self.turns.group_sum(self.part, tensor)

    class Boost(Parts, m.Boost):
        pass

    class Cut(Parts, m.Cut):
        rows = None     # rows per token at each chain site, the sequence's GPUs together (None: the full cycle)
        start = None    # 'time': every token's window starts here; None: token windows
        prior = False   # E7: the mid sites measure the identity-path guess's residual
        budget = None   # E13: the run's ROW_BUDGETS strategy (token windows of each token's own count)
        key_blocks = None  # E17: the blocks whose input-key probes run (None: every block, the record's)
        deep = False    # E5: one S1 launch reads S2..S8 (deep_chain_credits)
        transport = False  # E17: the K-branch input credit from the raw key errors through c_k, no probes
        gate_transport = False  # the gate's input credit from its own error through ve_gate, no probes
        top = None      # this estimate's top-stream credit (the guess E7 carries down)

        @torch.no_grad()
        def estimate(self, x, y):
            gradients = super().estimate(x, y)
            self.top = None
            return gradients

        def key_input_credits(self, targets):
            """The record's input-key probes at key_blocks only; the other blocks' K-branch input credit is zero, and
            their c_k weights take the key errors the record uses without these probes (exact_key_errors)."""
            if self.transport:
                return self.transported_key_credits(targets)
            if self.key_blocks is None:
                return super().key_input_credits(targets)
            chosen = {i: t for i, t in targets.items() if i in self.key_blocks}
            credits = super().key_input_credits(chosen) if chosen else {}
            errors = dict(self._input_key_errors or {})
            errors.update(self.exact_key_errors(targets, [i for i in targets if i not in chosen]))
            self._input_key_errors = errors
            d = self.capture
            zero = torch.zeros(d.batch * d.length, self.width, device=self.device)
            return {i: credits.get(i, zero) for i in targets}

        def gate_input_credits(self, targets, raw_values=None):
            if not self.gate_transport:
                return super().gate_input_credits(targets, raw_values)
            return self.transported_gate_credits(targets, raw_values)

        def transported_gate_credits(self, targets, raw_values=None):
            """The gate's input credit without its probes: from the block input x the gate is RMSNorm, the first
            channels through ve_gate (linear), then 2 sigmoid scaling the value embedding, so the input's credit along
            it is the gate's own error (exact_values': the record's central slope times the value credit dotted with
            the embedding, what its ve_gate weight takes) through ve_gate's weight, zero past the channels it reads,
            and one forward RMSNorm step (norm_move, the record's value-path convention)."""
            d, heads, credits = self.capture, self.model.config.n_head, {}
            for i, target in targets.items():
                if not m.has_ve(i, self.model.config.n_layer):
                    continue
                prefix = f'transformer.h.{i}.attn.'
                raw = None if raw_values is None else raw_values[i]
                error = self.exact_values(i, target, None, raw)[prefix + 've_gate']  # [b, t, heads]
                weight = self.model.transformer.h[i].attn.ve_gate.weight.float()  # [heads, channels]
                g = torch.zeros(error.shape[0] * error.shape[1], self.width, device=self.device)
                g[:, :weight.shape[1]] = error.reshape(-1, heads) @ weight
                credits[i] = m.norm_move(d.blocks[i].float().flatten(0, 1), g)
            return credits

        def transported_key_credits(self, targets):
            """E17's key transport: from the block input x the K branch is RMSNorm, then c_k (linear) to the raw key,
            so the input's credit along it is the raw key's credit (the record's single-key experiments, which its
            c_k weights use anyway; this GPU's share of each head's coordinates) times c_k's weight, through one
            forward RMSNorm step (norm_move, the record's value-path convention), summed over the sequence's GPUs as
            the probes' estimate is. No input-key probes run."""
            d, credits, errors = self.capture, {}, {}
            for i, target in targets.items():
                raw = torch.empty(d.batch, d.length, self.width, device=self.device)
                for r in range(d.batch):
                    context = self.key_context(i, r, target)
                    raw[r] = self.key_errors_from_context(i, context).transpose(0, 1).reshape(d.length, self.width)
                w_key = self.model.transformer.h[i].attn.c_k.weight.float()
                # through the K branch's RMSNorm as the record's value path goes (PR #23's norm_move: a central
                # forward response of the norm along the credit, which its symmetric Jacobian makes the pullback)
                credits[i] = self.group_sum(m.norm_move(d.blocks[i].float().flatten(0, 1),
                                                        (raw @ w_key).flatten(0, 1)))
                errors[f'transformer.h.{i}.attn.c_k'] = raw
            self._input_key_errors = errors
            return credits

        def probe(self, x, y, mix=None, probed=None):
            """The record's probe; Cut calls it once, for the top stream's credit (E7's guess carries it down)."""
            self.top = super().probe(x, y, mix, probed)
            return self.top

        def chain_credits(self, e, x0, mids, outs, routes, y):
            if self.deep:
                return self.deep_chain_credits(e, x0, mids, outs, routes, y)
            if self.rows is None:
                return super().chain_credits(e, x0, mids, outs, routes, y)
            return self.partial_chain_credits(e, x0, mids, outs, routes, y)

        def row_index(self, n, counts=None):
            """[n, rows / split]: each token's rows of the basis, this GPU's share (the first or second half of the
            token's window). Windows start at `start` for every token ('time'), else at (t * count) mod width, the
            count being `rows` or the token's own budget ('token')."""
            share = self.rows // self.split
            if self.start is not None:
                first = torch.full((n,), self.start, device=self.device)
            else:
                first = torch.arange(n, device=self.device) * (self.rows if counts is None else counts) % self.width
            return (first[:, None] + self.part * share + torch.arange(share, device=self.device)) % self.width

        def guesses(self):
            """The identity path's credit at each mid site: the top stream's credit carried down the residual scalars
            (H6's credit_dump guess)."""
            lam, carried, out = self.model.resid_lambdas.float(), self.top, {}
            for i in reversed(range(self.model.config.n_layer)):
                out[i] = carried
                carried = lam[i] * carried
            return out

        def chain_pass(self, e, x0, mids, outs, routes, y, probes):
            """The record's chain_credits loop (its kernels, token chunks and own-query context) over `probes`:
            (site, moves), moves(sl) giving a token slice's (move [t, P, W], scale [t, P]). Returns each probe's kernel
            outputs [n, P]: each token's loss change along each move, divided by its scale."""
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
                    for k, (site, moves) in enumerate(probes):
                        move, scale = moves(sl)
                        derivative = kernel(
                            site, move, scale, e[sl], x0[sl], [z[sl] for z in mids], xL[sl],
                            [r[0][sl] for r in routes], [r[1][sl] for r in routes], lam_r, lam_0, selected_fc,
                            selected_out, weight_t, base[sl], target_w[sl], y[sl], own_sl, packed=True)
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

        def deep_chain_credits(self, e, x0, mids, outs, routes, y):
            """E5 (module docstring), one pass over each sequence's token chunks with one own-query context, as the
            record's chain_credits: S0's cycle as the record's (this GPU's share of the moves, every token); S1's full
            cycle and its scalar moves through sites_kernel, and the later sites' scalar moves through the record's
            kernel, for this GPU's share of the chunk's tokens (each token's system needs all its moves, and the
            tokens are independent, so the GPUs of a sequence split the tokens and stay balanced); S1's credit decoded
            as the record's, each later mid site's solved per token. The bases are drawn as the record draws them, the
            skipped sites' too, so the run stays paired with the record's draw for draw."""
            model, s, L = self.model, self.s, self.model.config.n_layer
            vocab = model.config.vocab_size
            weight = model.lm_head.weight.float()[:vocab]
            weight_t = weight.T.to(torch.bfloat16).contiguous()
            w_fc = [blk.mlp.c_fc.weight.to(s.chain_dtype) for blk in model.transformer.h]
            w_out_t = [blk.mlp.c_proj.weight.to(s.chain_dtype).T.contiguous() for blk in model.transformer.h]
            lam_r, lam_0 = model.resid_lambdas.float(), model.x0_lambdas.float()
            xL, h = outs[-1], s.scalar_step
            base, target_w = xL @ weight.T, weight[y.clamp(min=0)]
            lo, hi = self.share(self.width)
            H0 = self.basis()[lo:hi]  # S0's, this GPU's share
            H = self.basis()          # S1's, whole
            for _ in range(L - 1):    # the sites whose cycles E5 skips: drawn and dropped, as the record draws them
                self.basis()
            P0, P = H0.shape[0], H.shape[0]
            steps0 = s.chain_sigma * e.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            steps = s.chain_sigma * mids[0].square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
            guesses = self.guesses()
            n, chunk = e.shape[0], s.chain_tokens
            emb = torch.zeros(n, self.width, device=self.device)
            out = {k: torch.zeros(n, self.width, device=self.device) for k in range(L)}
            lam = {k: torch.zeros(2, device=self.device) for k in range(L)}
            status = torch.zeros((), dtype=torch.int32, device=self.device)
            length = self.capture.length if s.own_query else n
            for start in range(0, n, length):
                own = None
                if s.own_query:
                    own = self.own_query_inputs(start // length)
                    if s.query_keys:
                        own = self.strong_keys(own, s.query_keys)

                def context(sl):
                    """The kernel and the token slice's own-query context, its selected units and clean inputs."""
                    own_sl = None
                    if own is not None:
                        begin, end = sl.start - start, sl.stop - start
                        if own[0] == 'routed':
                            local = slice(begin, end)
                            own_sl = ('routed', own[1][local], [z[local] for z in own[2]], own[3]) + tuple(
                                [z[local] for z in part] for part in own[4:8]) + own[8:]
                        else:
                            own_sl = self.own_query_window(own, begin, end)
                    window = own_sl is not None and own_sl[0] == 'window'
                    return window, (e[sl], x0[sl], [z[sl] for z in mids], xL[sl], [r[0][sl] for r in routes],
                                    [r[1][sl] for r in routes], lam_r, lam_0,
                                    [w[r[0][sl]] for w, r in zip(w_fc, routes)],
                                    [w[r[0][sl]] for w, r in zip(w_out_t, routes)], weight_t, base[sl], target_w[sl],
                                    y[sl], own_sl)

                for t0 in range(start, start + length, chunk):
                    t1 = min(t0 + chunk, start + length)
                    sl = slice(t0, t1)
                    window, args = context(sl)
                    kernel = m.chain_chunk_own if window else m.chain_chunk
                    d = kernel(-1, steps0[sl].unsqueeze(-1) * H0.unsqueeze(0), steps0[sl].expand(-1, P0), *args,
                               packed=True)
                    emb[sl] = d @ H0 / self.width
                    del args
                    share = (t1 - t0) // self.split  # this GPU's tokens of the chunk
                    mine = slice(t0 + self.part * share, t1 if self.part == self.split - 1 else t0 + (self.part + 1) * share)
                    window, args = context(mine)
                    z0 = x0[mine].unsqueeze(1)  # S1's scalars both scale x0 (block 0's input)
                    move = torch.cat([steps[mine].unsqueeze(-1) * H.unsqueeze(0), h * z0, h * z0, -h * z0, -h * z0], 1)
                    unit = torch.ones_like(steps[mine].expand(-1, 2))
                    scale = torch.cat([steps[mine].expand(-1, P), h * unit, -h * unit], 1)
                    derivative, changes = (SITES[2] if window else SITES[1])(0, move, scale, *args, rows=P)
                    out[0][mine] = derivative[:, :P] @ H / self.width
                    lam[0] += (derivative[:, P:P + 2] + derivative[:, P + 2:]).sum(0) / 2
                    for k, change in enumerate(changes, 1):
                        g, info = solve_credits(change, derivative[:, :P], steps[mine], guesses[k][mine].float())
                        out[k][mine] = g
                        status = torch.maximum(status, info.max().to(status.dtype))
                    del changes, derivative
                    kernel = m.chain_chunk_own if window else m.chain_chunk
                    for k in range(1, L):  # the later sites' mixing scalars: the record's 4 central moves each
                        prev = outs[k - 1][mine].unsqueeze(1)
                        d = kernel(k, torch.cat([h * prev, h * z0, -h * prev, -h * z0], 1),
                                   torch.cat([h * unit, -h * unit], 1), *args, packed=True)
                        lam[k] += (d[:, 0:2] + d[:, 2:4]).sum(0) / 2
                    del args
                del own
            if int(status):  # one check per estimate: a failed factorization would leave garbage, not an error
                raise RuntimeError(f'deep readout: a 512 x 512 system was not positive definite (status {int(status)})')
            credits = {'emb': self.group_sum(emb)}
            for k in range(L):
                credits[k] = self.group_sum(out[k])
                credits['lam', k] = self.group_sum(lam[k])
            return credits

        def partial_chain_credits(self, e, x0, mids, outs, routes, y):
            """PR #40's chain_credits at `rows` rows per token per site (or each token's budget): each token reads its
            rows of the site's freshly signed basis (this GPU's share of its window), decoded over them and scaled by
            1 / its row count; with the prior, the mid sites decode the identity-path guess's residual and add the
            guess back (on part 0, so the parts' sum holds it once)."""
            s, L = self.s, self.model.config.n_layer
            n, first = e.shape[0], self.part == 0
            counts = None if self.budget is None else self.budget.rows(self.capture).to(self.device)
            index = self.row_index(n, counts)
            share = index.shape[1]
            if counts is None:
                mask, scale = None, 1 / self.rows
            else:
                mask = self.part * share + torch.arange(share, device=self.device) < counts[:, None]
                scale = 1 / counts.float()[:, None]
            guesses = self.guesses() if self.prior else {}
            sites, bases, probes = range(-1, L), [], []
            for site in sites:
                H = self.basis()  # one per site, in site order, as the record draws them
                probed = e if site < 0 else mids[site]
                steps = s.chain_sigma * probed.square().mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
                prev = (x0 if site == 0 else outs[site - 1]) if site >= 0 and first else None
                bases.append(H)
                probes.append((site, partial(self.window_moves, H, steps, index, prev, x0)))
            measured = self.chain_pass(e, x0, mids, outs, routes, y, probes)
            credits = {}
            for site, H, d in zip(sites, bases, measured):
                guess = guesses.get(site)
                credit = torch.empty(n, self.width, device=self.device)
                for t0 in range(0, n, s.chain_tokens):
                    sl = slice(t0, t0 + s.chain_tokens)
                    directions = H[index[sl]]  # [tokens, share, width]
                    residual = d[sl, :share] if guess is None else (
                        d[sl, :share] - torch.einsum('tw,tpw->tp', guess[sl], directions))
                    if mask is not None:
                        residual = residual * mask[sl]
                    credit[sl] = torch.einsum('tp,tpw->tw', residual, directions) * (
                        scale if mask is None else scale[sl])
                if guess is not None and first:
                    credit += guess
                credits['emb' if site < 0 else site] = self.group_sum(credit)
                if site >= 0:
                    lam = (d[:, -4:-2] + d[:, -2:]).sum(0) / 2 if first else torch.zeros(2, device=self.device)
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


class CutChain:
    def __init__(self, o, device, seed, steps, heldout):
        m = self.m = refs.trainer_module('pr40')
        self.o, self.device, self.seed, self.steps, self.heldout = o, device, seed, steps, heldout
        self.settings = s = record_settings(m, o['preset'], o['batch'])
        if not o['key_probes']:  # E17: the record without its input-key probes (path_credits skips them)
            s.key_path_draws = 0
        self.width = s.model.n_embd
        self.world = o['virtual_gpus']
        self.split = max(1, self.world // o['batch'])  # GPUs per sequence, as train_stages
        self.per = o['batch'] // (self.world // self.split)  # sequences per GPU
        self.schedule = m.stage_schedule(steps * o['batch'], s)
        self.horizons = {phase: horizon for phase, _, _, _, horizon in self.schedule}
        self.Boost, self.Cut = estimator_classes(m)
        if o['profile']:
            PROFILE.install(m, self.Cut)
        self.turns = Turns(self.split) if self.split > 1 else None
        self.finish = self.end_merge if o['end_merge'] else None
        if o['row_budget'] is not None and o['row_budget'] not in ROW_BUDGETS:
            raise ValueError(f'unknown row_budget {o["row_budget"]!r}; have {sorted(ROW_BUDGETS)}')
        self.reset()
        self.graphs = self.chain_graphs()
        self.recompile_limit = max(64, 2 * len(self.graphs))  # the record's 64 holds its own 17 graphs
        raise_recompile_limit(self.recompile_limit)

    def describe(self):
        return dict(name='pr40', file=refs.TRAINERS['pr40'], settings=asdict(self.settings),
                    virtual_gpus=self.world, gpus_per_sequence=self.split,
                    boost_updates=sum(phase == 'boost' for phase, *_ in self.schedule),
                    chain_rows=self.o['rows'] or self.width, row_assign=self.o['row_assign'], prior=self.o['prior'],
                    row_budget=self.o['row_budget'], key_blocks=self.o['key_blocks'] or 'all', key_transport=self.o['key_transport'], gate_transport=self.o['gate_transport'], deep_readout=self.o['deep_readout'], merge_batch=self.o['merge_batch'],
                    reverse_sum=self.o['reverse_sum'], chain_graphs=len(self.graphs),
                    recompile_limit=self.recompile_limit)

    def estimators(self, phase):
        cls = self.Boost if phase == 'boost' else self.Cut
        out = [cls(self.model, self.settings, self.seed, rank, self.world, self.split, None)
               for rank in range(self.world)]
        for estimator in out:
            estimator.close()  # measure() attaches the hooks of the sequence it measures
            estimator.turns = self.turns
        return [out[i:i + self.split] for i in range(0, self.world, self.split)]

    def reset(self, phase=None):
        """A fresh model, the phase's optimizer and estimators (the record's first phase unless given), and a fresh row
        budget, so the run's never saw the warmup's random tokens."""
        m, s = self.m, self.settings
        self.phase = phase or self.schedule[0][0]
        budget = self.o['row_budget']
        self.budget = None if budget is None else ROW_BUDGETS[budget](self.o, s.model.vocab_size)
        self.model = m.make_model(self.device, self.seed, s.model, s.projection_std, s.head_std)
        self.named = list(self.model.named_parameters())
        self.optimizer = m.make_optimizer(self.model, s, self.horizons[self.phase])
        self.labels = rate_labels(self.optimizer, s.lr)
        self.groups = self.estimators(self.phase)

    def switch(self, phase, phase_step, horizon):
        """Boost -> Cut as train_stages: a new optimizer for the phase, the state handed over, new estimators."""
        optimizer = self.m.make_optimizer(self.model, self.settings, horizon)
        self.m.handoff_optimizer(self.optimizer, optimizer, phase_step)
        self.optimizer, self.phase = optimizer, phase
        self.labels = rate_labels(optimizer, self.settings.lr)
        self.groups = self.estimators(phase)

    def window_start(self, step):
        """'time': where every token's window starts at update `step`; None for token windows or the full cycle."""
        if self.o['rows'] is None or self.o['row_assign'] != 'time':
            return None
        return step % (self.width // self.o['rows']) * self.o['rows']

    def measure(self, group, x, y, total=None):
        """Add each of one sequence's GPUs' float32 gradients (divided by the sequence's token count, as Boost.step)
        to `total` in rank order, as the step's all-reduce (reversed with reverse_sum, the calibration of its
        rounding); returns the total and each GPU's clean loss."""
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
        for g in gradients[::-1] if self.o['reverse_sum'] else gradients:
            flat = torch.cat([(g[name] / valid).float().reshape(-1) for name, _ in self.named])
            total = flat if total is None else total.add_(flat)
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
        if phase == 'cut':
            start = self.window_start(step)
            for group in self.groups:
                for estimator in group:
                    estimator.rows, estimator.start = self.o['rows'], start
                    estimator.prior, estimator.budget = self.o['prior'] == 'identity', self.budget
                    estimator.key_blocks = set(self.o['key_blocks']) if self.o['key_blocks'] else None
                    estimator.deep = self.o['deep_readout']
                    estimator.transport = self.o['key_transport']
                    estimator.gate_transport = self.o['gate_transport']
            fields['chain_rows'] = self.o['rows'] or self.width
            if start is not None:
                fields['chain_window'] = [start, start + self.o['rows']]
        total, losses = None, []
        PROFILE.on = self.o['profile'] and phase == 'cut'
        order = list(enumerate(self.groups))  # group g: the GPUs of the batch's g-th sequences
        for g, group in order[::-1] if self.o['reverse_sum'] else order:
            seqs = batch[g * self.per:(g + 1) * self.per]
            total, loss = self.measure(group, seqs[:, :-1].contiguous(), seqs[:, 1:].contiguous(), total)
            losses += loss
        total /= len(self.groups)  # Boost.step: the all-reduced sum over the GPUs, divided by the sequences
        if PROFILE.on:
            PROFILE.on = False
            fields['cut_profile'] = PROFILE.take()
        if not bool(torch.isfinite(total).all()):
            raise FloatingPointError('Non-finite estimate')
        self.optimizer.zero_grad(set_to_none=True)
        offset = 0
        for _, p in self.named:
            p.grad = total[offset:offset + p.numel()].view_as(p).to(p.dtype)
            offset += p.numel()
        self.optimizer.step()
        if self.budget is not None:  # every update's batch, after it, as H8's E13 table learns the stream
            self.budget.observe(batch)
        return sum(losses) / len(losses), fields  # average_ranks: the all-reduced sum over the GPUs, divided by them

    def chain_graphs(self):
        """chain_chunk's graphs, as Dynamo specializes it: one per (site, moves per token), for the Cut phase's row
        count. Part 0 adds the mixing scalars' 4 moves at the mid sites; the other parts do not."""
        if not any(phase == 'cut' for phase, *_ in self.schedule):
            return set()
        share = (self.o['rows'] or self.width) // self.split
        if self.o['deep_readout']:  # E5: chain_chunk runs S0's cycle and the later sites' 4 scalar moves
            return {(-1, share)} | {(site, 4) for site in range(1, self.settings.model.n_layer)}
        keys = {(-1, share)} | {(site, share + 4) for site in range(self.settings.model.n_layer)}
        if self.split > 1:
            keys |= {(site, share) for site in range(self.settings.model.n_layer)}
        return keys

    def warmup(self, batch):
        """As the record: one update of each phase on random tokens, each from a fresh model, so every shape compiles
        before the clock starts. Fails unless the chain's kernel compiled every planned graph (below Dynamo's
        limit): a missing shape would run eagerly."""
        config = torch._dynamo.config
        if compiled_graphs(self.m.chain_chunk_eager) is None and hasattr(config, 'fail_on_recompile_limit_hit'):
            config.fail_on_recompile_limit_hit = True  # this torch cannot count the graphs: fail at the limit instead
        firsts = {}
        for step, (phase, *_) in enumerate(self.schedule, 1):
            firsts.setdefault(phase, step)
        for phase, step in firsts.items():
            self.reset(phase)
            self.update(step, batch)
        compiled = compiled_graphs(self.m.chain_chunk_eager)
        if compiled is not None and not torch._dynamo.config.disable:
            print(f'chain_chunk: {compiled} graphs compiled; {len(self.graphs)} planned, limit '
                  f'{self.recompile_limit}', flush=True)
            if compiled < len(self.graphs) or compiled >= self.recompile_limit:
                raise RuntimeError(f'chain_chunk compiled {compiled} of its {len(self.graphs)} planned graphs (limit '
                                   f'{self.recompile_limit}): the missing shapes would run eagerly')

    def diagnose(self, step):
        return {}

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

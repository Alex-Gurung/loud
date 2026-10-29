"""CPU checks of the training methods, one per process (each loads its own reference trainer; see refs.trainer_module).

    python tests/method_checks.py <check> <scratch dir>

Two test-only patches make them run on a CPU: dense attention with the same causal window instead of flex attention
(which has no CPU backward, and on CPU sees float32 values under autocast), and sequences cut to SEQ tokens. The Cut
chain's checks also shrink the vocabulary (CUT_VOCAB): its head matmuls are bfloat16, which this CPU runs slowly.
"""
import copy
import json
import os
import socket
import subprocess
import sys
import types
from pathlib import Path

import torch

from loud import data, refs, train

SEQ = 129


def patch_attention():
    um = sys.modules['utils.model']

    def dense_attention(q, k, v, window):  # written out, so it also has a double backward
        i = torch.arange(q.shape[1])
        mask = (i[:, None] >= i[None, :]) & (i[:, None] - i[None, :] <= window)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.to(q.dtype).transpose(1, 2)
        scores = (q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5).masked_fill(~mask, float('-inf'))
        return (scores.softmax(-1) @ v).transpose(1, 2)
    um.attention = dense_attention


def patch_sequences(*loops, length=SEQ, vocab=None):
    real = data.benchmark_split

    def short(*args, **kwargs):
        split = real(*args, **kwargs)
        split.train, split.heldout = split.train[:, :length].contiguous(), split.heldout[:, :length].contiguous()
        if vocab is not None:
            split.train, split.heldout = split.train % vocab, split.heldout % vocab
        return split
    for loop in (train, *loops):
        loop.benchmark_split, loop.SEQ = short, length


def proxy_gradient(scratch):
    """micro_batch 1 is the float32 mean of per-sequence gradients; micro_batch = batch is one direct backward."""
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    o = proxy_options(dict(preset='2m', batch=4, micro_batch=1))
    trainer = Backprop(o, reference, torch.device('cpu'), 0, 10)
    model, params = trainer.model, list(trainer.model.parameters())
    batch = torch.randint(0, model.config.vocab_size, (4, SEQ), generator=torch.Generator().manual_seed(1))

    def direct(rows):
        model.zero_grad(set_to_none=True)  # the trainer leaves .grad aliasing its accumulators
        model.requires_grad_(True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = model(rows[:, :-1].contiguous(), rows[:, 1:].contiguous())
        loss.backward()
        grads = [torch.zeros_like(p, dtype=torch.float32) if p.grad is None else p.grad.float().clone() for p in params]
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
        return loss.detach().float(), grads

    per_sequence = [direct(batch[i:i + 1]) for i in range(4)]
    expected = [sum(g[j] for _, g in per_sequence) / 4 for j in range(len(params))]
    loss = trainer.gradient(batch)
    assert torch.allclose(loss, torch.stack([l for l, _ in per_sequence]).mean())
    for p, e in zip(params, expected):
        assert torch.equal(p.grad, e.to(p.dtype)), 'micro_batch 1 is not the float32 mean of per-sequence gradients'

    whole_loss, whole = direct(batch)
    trainer.o = dict(o, micro_batch=4)
    loss = trainer.gradient(batch)
    assert torch.equal(loss, whole_loss)
    for p, e in zip(params, whole):
        assert torch.equal(p.grad, e.to(p.dtype)), 'micro_batch = batch is not one direct backward'

    for steps in (606, 610, 1212):  # backprop_preset.py: lr = initial * min(1, left / warmdown), one-based steps
        group = dict(lr=1.0)
        optimizer = type('O', (), dict(param_groups=[group]))
        for step in (1, steps // 2, steps - 1, steps):
            train.set_rate(optimizer, step, steps, 0.3)
            assert group['lr'] == min(1.0, ((steps - step + 1) / steps) / 0.3)


def proxy_run(scratch):
    """The whole loop on a tiny model: every H1-H4 output exists and is consistent."""
    from loud.proxy import run
    refs.trainer_module('boostcut')
    patch_attention()
    patch_sequences()
    spec = dict(experiment='cpu', variant='proxy', seed=0, task='proxy', args=dict(
        preset='2m', sequences=48, heldout=4, batch=4, checkpoints=True, checkpoint_optimizer=True, heldout_every=2,
        means=True, save_means=True, per_sequence_grads=True, grad_noise_every=1, official_eval=False))
    out = Path(scratch) / 'proxy'
    result = run(spec, out)
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and result['updates'] == 11 and len(metrics) == 11
    assert all({'trace_cov', 'true_sq', 'b_simple'} <= set(row) for m in metrics for row in m['grad_noise'].values())
    assert {f'ckpt_{f:02d}.pt' for f in (0, 10, 20, 50, 100)} <= {p.name for p in out.iterdir()}
    optim = torch.load(out / 'ckpt_50.optim.pt', weights_only=True)
    assert optim['state'] and all({'fast', 'slow'} <= set(v) for v in optim['state'].values())
    # means from 60% of the epoch: updates 10 and 11 (every 5, and the last); each then differs from the final weights
    assert result['means_updates'] == 2 and len(result['means_heldout']) == 9 and (out / 'means.pt').exists()
    assert all(v != result['heldout_loss'] for v in result['means_heldout'].values())
    heldout = [json.loads(line) for line in (out / 'heldout.jsonl').read_text().splitlines()]
    assert [h['step'] for h in heldout if 'mean_decay' not in h].count(11) == 1  # the last update is scored once


def proxy_merge(scratch):
    """The proxy with PR #40's recipe and its end merge: timed, chosen on the held-out rows, the plain weights kept
    and scored apart."""
    from loud.proxy import run
    refs.trainer_module('pr40')
    patch_attention()
    patch_sequences()
    spec = dict(experiment='cpu', variant='merge', seed=0, task='proxy', args=dict(
        recipe='pr40', preset='2m', sequences=48, heldout=4, batch=4, means=True, end_merge=True, official_eval=False))
    out = Path(scratch) / 'merge'
    result = run(spec, out)
    assert result['complete'] and result['merge_seconds'] > 0 and (out / 'final_plain.pt').exists()
    assert result['holdout_merged'] <= result['holdout_plain'] and result['merge_choices']
    merged, plain = (torch.load(out / f, weights_only=True) for f in ('final.pt', 'final_plain.pt'))
    assert (result['holdout_merged'] < result['holdout_plain']) == any(
        not torch.equal(merged[k], plain[k]) for k in merged)




def identity_step(scratch):
    """An IdentityPath update over two virtual GPUs is PR #26's: one FastRelay per rank (its own seeds) on one
    sequence each, then Boost.step's sum of the per-sequence gradients over token counts, divided by the ranks."""
    from loud.identity import IdentityPath
    from loud.options import identity_options
    m = refs.trainer_module('pr26')
    patch_attention()
    o = identity_options(dict(preset='tiny', batch=2))
    ours = IdentityPath(o, torch.device('cpu'), 0, 10, torch.zeros(1, SEQ, dtype=torch.long))
    model = m.make_model(torch.device('cpu'), 0, ours.config)
    optimizer = m.make_optimizer(model, ours.recipe, 10)
    ranks = [m.FastRelay(model, ours.settings, 0, rank, 2) for rank in range(2)]
    batch = torch.randint(0, ours.config.vocab_size, (2, SEQ), generator=torch.Generator().manual_seed(4))
    loss, _ = ours.update(1, batch)
    flats, losses = [], []
    for rank, estimator in enumerate(ranks):  # Boost.step on each rank, its all-reduce as a sum
        x, y = batch[rank:rank + 1, :-1].contiguous(), batch[rank:rank + 1, 1:].contiguous()
        gradients = estimator.estimate(x, y)
        flats.append(torch.cat([(gradients[n] / (y != -1).sum()).float().reshape(-1)
                                for n, _ in model.named_parameters()]))
        losses.append(estimator.capture.clean_loss[y != -1].mean())
    flat, offset = sum(flats) / 2, 0
    for p in model.parameters():
        p.grad = flat[offset:offset + p.numel()].view_as(p).to(p.dtype)
        offset += p.numel()
    optimizer.step()
    assert abs(loss.item() - torch.stack(losses).mean().item()) < 1e-6
    for (name, a), b in zip(ours.model.named_parameters(), model.parameters()):
        assert torch.equal(a, b), f'{name} differs after one update'


def identity_run(scratch):
    """The whole loop: H3, H4 and the result."""
    from loud.identity import run
    refs.trainer_module('pr26')
    patch_attention()
    patch_sequences()
    spec = dict(experiment='cpu', variant='identity', seed=0, task='identity', args=dict(
        preset='tiny', sequences=24, heldout=4, batch=4, heldout_every=2, checkpoints=True, official_eval=False))
    out = Path(scratch) / 'identity'
    result = run(spec, out)
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and len(metrics) == 5 and 'heldout_loss' in result
    assert {f'ckpt_{f:02d}.pt' for f in (0, 10, 20, 50, 100)} <= {p.name for p in out.iterdir()}


CUT_SEQ, CUT_VOCAB = 65, 512  # the Cut checks: 64 predictions per sequence, a 512-token vocabulary
CUT_ROWS = 72                   # their data: 40 sequences feed 10 updates of 4 (2 Boost), 32 are held out


def cut_module():
    """PR #40 run eagerly (both sides of a comparison) with dense attention, and loud.cut.record_settings shrunk to
    CUT_SEQ tokens and CUT_VOCAB."""
    import torch._dynamo
    torch._dynamo.config.disable = True
    m = refs.trainer_module('pr40')
    patch_attention()
    m.attention = sys.modules['utils.model'].attention  # the script's own reference to the flex attention
    import loud.cut
    real = loud.cut.record_settings

    def small(*args):
        s = real(*args)
        s.model.sequence_len, s.model.vocab_size = CUT_SEQ - 1, CUT_VOCAB
        return s
    loud.cut.record_settings = small
    return m


def cut_rows_data():
    return torch.randint(0, CUT_VOCAB, (CUT_ROWS, CUT_SEQ), generator=torch.Generator().manual_seed(6))


def cut_rank(scratch, rank):
    """One of 8 gloo processes running PR #40's own train_stages (its loop, groups and all-reduces) for 4 updates:
    2 Boost, then the switch and 2 Cut. Each all-reduce sums in rank order, as the port does, so the two can agree to
    the bit. Its CUDA calls become no-ops; load_rows returns cut_rows_data()."""
    import argparse
    import loud.cut
    m = cut_module()
    dist = torch.distributed
    dist.init_process_group('gloo', rank=int(rank), world_size=8)

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        assert op == dist.ReduceOp.SUM and not async_op
        parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(parts, tensor.contiguous(), group=group)
        total = parts[0].clone()
        for part in parts[1:]:
            total += part
        tensor.copy_(total)
    dist.all_reduce = all_reduce
    for name in ('synchronize', 'reset_peak_memory_stats'):
        setattr(torch.cuda, name, lambda *a, **k: None)
    torch.cuda.max_memory_allocated = lambda *a, **k: 0
    m.load_rows = lambda path, rows: cut_rows_data()
    args = argparse.Namespace(data_dir=Path(scratch), output=Path(scratch) / 'record', steps=None, stop_after_step=4,
                              resume=None, eval_every=0, checkpoint_every=0)
    m.train_stages(args, loud.cut.record_settings(m, '2m', 4), torch.device('cpu'), int(rank), 8)
    torch.distributed.destroy_process_group()


def cut_ranks(scratch):
    """The single-process port (8 virtual GPUs, two per sequence taking turns) against PR #40's train_stages on 8 real
    gloo processes, both eager in float32: bit-identical per-update losses and weights after 2 Boost and 2 Cut
    updates (the processes' all-reduces sum in rank order; on GPUs NCCL's order changes the rounding)."""
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = dict(os.environ, MASTER_ADDR='127.0.0.1', MASTER_PORT=str(port), TORCHDYNAMO_DISABLE='1')
    procs = [subprocess.Popen([sys.executable, __file__, 'cut_rank', scratch, str(r)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for r in range(8)]
    m = cut_module()
    from loud.cut import CutChain
    from loud.options import cut_options
    rows = cut_rows_data()
    order = rows[torch.randperm(len(rows), generator=torch.Generator().manual_seed(1))]  # the record's order
    training, heldout = order[:-32], order[-32:]
    o = cut_options(dict(preset='2m', batch=4, sequences=CUT_ROWS, heldout=32))
    ours = CutChain(o, torch.device('cpu'), 0, len(training) // 4, heldout)
    losses = [ours.update(step, training[(step - 1) * 4:step * 4])[0].item() for step in range(1, 5)]
    for proc in procs:
        out, _ = proc.communicate(timeout=1800)
        assert proc.returncode == 0, out[-4000:]
    record = [json.loads(line) for line in (Path(scratch) / 'record' / 'metrics.jsonl').read_text().splitlines()]
    assert [r['phase'] for r in record] == ['boost', 'boost', 'cut', 'cut']
    theirs = torch.load(Path(scratch) / 'record' / 'checkpoint.pt', weights_only=True)['model']
    for step, (a, b) in enumerate(zip(losses, [r['train_loss'] for r in record]), 1):
        assert a == b, f'update {step}: loss {a} here, {b} on 8 processes'
    for name, p in ours.model.state_dict().items():
        assert torch.equal(p, theirs[name]), f'{name} differs by {(p.float() - theirs[name].float()).abs().max()}'
    # reverse_sum (verify.cut_prefix_reversed's calibration): the same run up to the rounding of the 8-way sum
    o = cut_options(dict(preset='2m', batch=4, sequences=CUT_ROWS, heldout=32, reverse_sum=True))
    reversed_sum = CutChain(o, torch.device('cpu'), 0, len(training) // 4, heldout)
    for step in (1, 2):
        loss = reversed_sum.update(step, training[(step - 1) * 4:step * 4])[0].item()
        assert abs(loss - losses[step - 1]) < 1e-5, f'update {step}: reversed sum {loss}, rank order {losses[step - 1]}'


def cut_windows(scratch):
    """rows = full is PR #40's Cut estimate. At rows = width / K, with one signed basis for every site: the K 'time'
    windows average to the full cycle's chain credits, with the identity prior too (whatever its guess, and exactly
    the credit, window by window, when the guess is exact); 'token' windows give each token its window's credit; a
    row budget of `rows` for every token is the token windows, a budget of half reads each token's first half; the
    GPUs of a sequence split each token's window into halves, and through the trainer, two virtual GPUs per sequence
    meeting in group_sum give what one GPU measures."""
    m = cut_module()
    from loud.cut import estimator_classes, record_settings
    s = record_settings(m, '2m', 4)
    width, L = s.model.n_embd, s.model.n_layer
    model = m.make_model(torch.device('cpu'), 0, s.model, s.projection_std, s.head_std)
    _, Cut = estimator_classes(m)
    row = cut_rows_data()[:1]
    x, y = row[:, :-1].contiguous(), row[:, 1:].contiguous()
    ours, theirs = Cut(model, s, 0, 0, 1, 1, None), m.Cut(model, s, 0, 0, 1, 1, None)
    a, b = ours.estimate(x, y), theirs.estimate(x, y)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a), 'the full cycle is not the record'

    saved = {}

    def keep(self, *args):
        saved['args'] = args
        return Cut.chain_credits(self, *args)
    ours.chain_credits = types.MethodType(keep, ours)
    ours.estimate(x, y)
    del ours.chain_credits
    fixed = ours.basis()
    ours.basis = lambda: fixed  # one signed basis, so windows compare exactly with the full cycle
    credits = lambda: ours.chain_credits(*saved['args'])
    full = credits()
    K, rows, n = 4, width // 4, x.shape[1]
    tokens = torch.arange(n)
    ours.rows = rows

    def windows():
        out = []
        for k in range(K):
            ours.start = k * rows
            out.append(credits())
        ours.start = None
        return out
    time = windows()
    tiled = credits()
    for key, value in full.items():
        mean = sum(w[key] for w in time) / K
        assert torch.allclose(mean, value, rtol=1e-4, atol=1e-6), f'{key}: the windows do not average to the cycle'
        if not isinstance(key, tuple):  # ('lam', site): the mixing scalars' probes do not use the rows
            each = torch.stack([w[key] for w in time])[tokens % K, tokens]
            assert torch.allclose(tiled[key], each, rtol=1e-4, atol=1e-6), f'{key}: a token misses its window'
    ours.prior, ours.top = True, torch.randn(n, width, generator=torch.Generator().manual_seed(3))  # any guess
    primed = windows()
    for i in range(L):
        mean = sum(w[i] for w in primed) / K
        assert torch.allclose(mean, full[i], rtol=1e-4, atol=1e-5), f'mid {i}: the prior biases the windows'
    assert torch.allclose(primed[0]['emb'], time[0]['emb']), 'S0 keeps the plain estimate'
    ours.guesses = lambda: {i: full[i] for i in range(L)}  # the exact credit as the guess: no residual left
    for w in windows():
        assert all(torch.allclose(w[i], full[i], rtol=1e-4, atol=1e-6) for i in range(L)), 'an exact guess moves'
    del ours.guesses
    ours.prior = False

    class Budget:
        def __init__(self, count):
            self.count = count

        def rows(self, capture):
            return torch.full((capture.x.numel(),), self.count)
    ours.budget = Budget(rows)
    same = credits()
    ours.budget = Budget(rows // 2)
    halved = credits()
    ours.budget, ours.rows = None, rows // 2
    plain_half = credits()
    ours.rows = rows
    for key in tiled:
        if not isinstance(key, tuple):
            assert torch.allclose(same[key], tiled[key]), f'{key}: a full budget is not the token windows'
            assert torch.allclose(halved[key], plain_half[key], rtol=1e-4, atol=1e-6), f'{key}: a half budget'

    halves = [Cut(model, s, 0, part, 2, 2, None) for part in range(2)]
    for start in (0, None):
        for half in halves:
            half.rows, half.start = rows, start
        index = torch.cat([h.row_index(8) for h in halves], 1)
        first = torch.zeros(8, dtype=torch.long) if start == 0 else torch.arange(8) * rows
        assert torch.equal(index, (first[:, None] + torch.arange(rows)) % width), 'the halves do not tile a window'

    # the same through the trainer with two virtual GPUs per sequence, meeting in group_sum: the K windows of one
    # basis average to the record's chain credits, and rows = 2 (one row per GPU) sum to what one GPU measures
    from loud.cut import CutChain, merge_loss
    from loud.options import cut_options

    def chain_credits(gpus, settings):
        """Part 0's chain credits (complete after group_sum) of one measure per setting, every one from the same
        generator states, so every setting reads the same bases."""
        o = cut_options(dict(preset='2m', batch=1, virtual_gpus=gpus, sequences=CUT_ROWS, heldout=32))
        trainer = CutChain(o, torch.device('cpu'), 0, 40, cut_rows_data()[-32:])
        trainer.reset('cut')
        group, out = trainer.groups[0], []
        states = [[g.get_state() for g in (e.generator, e.capture.generator, e.capture.shared_generator)]
                  for e in group]
        real = type(group[0]).chain_credits

        def keep(self, *args):
            credits = real(self, *args)
            if self.part == 0:
                out.append({k: v.clone() for k, v in credits.items()})
            return credits
        type(group[0]).chain_credits = keep
        for rows_, start in settings:
            for e, state in zip(group, states):
                for g, saved_state in zip((e.generator, e.capture.generator, e.capture.shared_generator), state):
                    g.set_state(saved_state)
                e.rows, e.start = rows_, start
            trainer.measure(group, x, y)
        type(group[0]).chain_credits = real
        return out
    split = chain_credits(2, [(None, None)] + [(rows, k * rows) for k in range(K)] + [(2, 0)])
    single = chain_credits(1, [(2, 0)])[0]
    for key, value in split[0].items():
        mean = sum(w[key] for w in split[1:1 + K]) / K
        assert torch.allclose(mean, value, rtol=1e-4, atol=1e-6), f'{key}: split windows do not average to the cycle'
        assert torch.allclose(split[-1][key], single[key], rtol=1e-4, atol=1e-6), f'{key}: rows = split differs'

    # merge_batch: the end merge's candidates scored in one batch are the rank-by-rank scores up to rounding
    rows8 = cut_rows_data()[-32:]
    assert abs(merge_loss(8, True)(model, rows8) - merge_loss(8, False)(model, rows8)) < 1e-5


def cut_run(scratch):
    """The whole loop on the 2-layer model (width 128), two virtual GPUs on one sequence per update: Boost, the switch,
    64 'time' rows rotating across updates with the identity prior, H4, the means (updates 10 and 12) and the
    record's end merge, timed; and chain_chunk runs exactly the planned shapes (the ones the warmup compiles)."""
    m = cut_module()
    import loud.cut
    from loud.options import cut_options
    patch_sequences(length=CUT_SEQ, vocab=CUT_VOCAB)
    calls, kernel = set(), m.chain_chunk

    def recorded(site, move, *args, **kwargs):  # what Dynamo specializes chain_chunk on
        calls.add((site, move.shape[1]))
        return kernel(site, move, *args, **kwargs)
    m.chain_chunk = recorded
    args = dict(preset='2m', sequences=20, heldout=8, batch=1, virtual_gpus=2, rows=64, row_assign='time',
                prior='identity', checkpoints=True, heldout_every=4, official_eval=False)
    out = Path(scratch) / 'cut'
    result = loud.cut.run(dict(experiment='cpu', variant='cut', seed=0, task='cut', args=args), out)
    plan = loud.cut.CutChain(cut_options(args), torch.device('cpu'), 0, 12, torch.zeros(8, CUT_SEQ, dtype=torch.long))
    assert calls == plan.graphs, f'chain_chunk ran {sorted(calls - plan.graphs)} unplanned, missed ' \
                                 f'{sorted(plan.graphs - calls)}'
    metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
    assert result['complete'] and len(metrics) == 12
    assert [r['phase'] for r in metrics] == ['boost'] * 2 + ['cut'] * 10
    assert [r.get('chain_window') for r in metrics] == [None] * 2 + [[s % 2 * 64, s % 2 * 64 + 64]
                                                                      for s in range(3, 13)]
    assert {'holdout_plain', 'holdout_merged', 'merge_choices', 'merge_seconds'} <= set(result)
    assert result['holdout_merged'] <= result['holdout_plain'] and result['means_updates'] == 2
    assert result['train_seconds'] >= result['merge_seconds']
    assert {'final.pt', 'final_plain.pt', 'ckpt_100.pt'} <= {p.name for p in out.iterdir()}
    calls_ = [json.loads(line) for line in (out / 'heldout.jsonl').read_text().splitlines()]
    assert [c['call'] for c in calls_] == list(range(1, len(calls_) + 1))
    assert sum(c['reason'] == 'merge' for c in calls_) > 80  # the plain weights and every candidate the merge scored


def cut_budget(scratch):
    """E13's hook through the trainer: a ROW_BUDGETS strategy is built fresh by every reset (the run's never sees the
    warmup's random tokens), gives each Cut estimate's tokens their counts (once per GPU of the sequence) and sees
    each Cut update's batch after the update; Boost never asks it."""
    cut_module()
    import loud.cut
    from loud.options import cut_options
    built = []

    class Recorder:
        def __init__(self, o):
            self.count, self.asked, self.observed = o['rows'] // 2, 0, []
            built.append(self)

        def rows(self, capture):
            self.asked += 1
            return torch.full((capture.x.numel(),), self.count)

        def observe(self, batch):
            self.observed.append(batch.clone())
    loud.cut.ROW_BUDGETS['recorder'] = Recorder
    o = cut_options(dict(preset='2m', batch=1, virtual_gpus=2, sequences=CUT_ROWS, heldout=8, rows=64,
                         row_budget='recorder', end_merge=False, means=False))
    data = cut_rows_data()
    chain = loud.cut.CutChain(o, torch.device('cpu'), 0, 10, data[-8:])
    first_cut = next(step for step, (phase, *_) in enumerate(chain.schedule, 1) if phase == 'cut')
    chain.warmup(data[-1:])
    warm = built[-1]
    assert warm.asked == 2 and len(warm.observed) == 1, 'the Cut warmup update did not use its budget'
    chain.reset()
    fresh = built[-1]
    assert fresh is not warm and fresh.asked == 0 and not fresh.observed, 'the run inherits the warmup budget'
    for step in range(1, first_cut + 2):
        chain.update(step, data[step - 1:step])
    assert fresh.asked == 4 and len(fresh.observed) == 2, (fresh.asked, len(fresh.observed))
    assert all(torch.equal(seen, data[s - 1:s]) for seen, s in zip(fresh.observed, (first_cut, first_cut + 1)))
    try:
        loud.cut.CutChain(dict(o, row_budget='missing'), torch.device('cpu'), 0, 10, data[-8:])
    except ValueError as exc:
        assert 'unknown row_budget' in str(exc)
    else:
        raise AssertionError('an unknown row_budget was accepted')


def tokens(n, seed=1, vocab=4096):
    return torch.randint(0, vocab, (n, SEQ), generator=torch.Generator().manual_seed(seed))


def variants_blur(scratch):
    """E2 in float32 (no autocast): r has the documented length at all nine sites, + and - add the same r, and the
    antithetic mean cancels the odd orders: far closer to the clean gradient than one displaced gradient, and closer
    by about 4x when r halves. Then the trainer: the clean loss, and the mean of all 2 x micro-batches gradients."""
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    from loud.variants import CHAIN_SIGMA, Blur
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    cpu = torch.device('cpu')
    model = reference.make_model(cpu, 0)
    width, params = model.config.n_embd, list(model.parameters())
    rows = tokens(2)
    x, y = rows[:, :-1].contiguous(), rows[:, 1:].contiguous()

    def gradient():
        model.requires_grad_(True)
        grads = torch.autograd.grad(model(x, y), params)
        model.requires_grad_(False)
        return torch.cat([g.reshape(-1) for g in grads])

    sites, inputs = [], {}  # the clean activations at the sites, read with the test's own hooks
    hooks = [model.transformer.wte.register_forward_hook(lambda m, a, out: sites.append(out))]
    for i, block in enumerate(model.transformer.h):
        hooks.append(block.register_forward_pre_hook(lambda m, a, i=i: inputs.__setitem__(i, a[0])))
        hooks.append(block.attn.register_forward_hook(lambda m, a, out, i=i: sites.append(inputs[i] + out)))
    with torch.no_grad():
        clean = model(x, y)
    for h in hooks:
        h.remove()
    clean_grad = gradient()

    tau = 10.0
    blur = Blur(model, tau, 0, cpu)
    assert torch.equal(blur.measure(lambda: model(x, y)), clean)
    assert len(blur.shifts) == len(sites) == 1 + model.config.n_layer
    for a, r in zip(sites, blur.shifts):
        expected = tau * CHAIN_SIGMA * a.square().mean(-1).sqrt() * width ** 0.5
        assert torch.allclose(r.norm(dim=-1), expected, rtol=1e-5), 'r does not have the documented length'
    for sign in (1, -1):  # each site's writer output moves by exactly sign * r
        raw, moved = [], []
        hooks = [model.transformer.wte.register_forward_hook(lambda m, a, out: raw.append(out), prepend=True)]
        hooks += [b.attn.register_forward_hook(lambda m, a, out: raw.append(out), prepend=True)
                  for b in model.transformer.h]
        with blur.shifted(sign), torch.no_grad():
            hooks += [model.transformer.wte.register_forward_hook(lambda m, a, out: moved.append(out))]
            hooks += [b.attn.register_forward_hook(lambda m, a, out: moved.append(out)) for b in model.transformer.h]
            model(x, y)
        for h in hooks:
            h.remove()
        for before, after, r in zip(raw, moved, blur.shifts):
            assert torch.allclose(after - before, sign * r, atol=1e-6), 'the two signs do not add the same r'

    def displaced(sign):
        with blur.shifted(sign):
            return gradient()

    plus, minus = displaced(1), displaced(-1)
    one_sided, antithetic = (plus - clean_grad).norm(), ((plus + minus) / 2 - clean_grad).norm()
    assert antithetic < 0.2 * one_sided, (antithetic, one_sided)
    blur.shifts = [r / 2 for r in blur.shifts]
    half = ((displaced(1) + displaced(-1)) / 2 - clean_grad).norm()
    assert half < 0.4 * antithetic, (half, antithetic)  # second order: a quarter, up to higher orders

    o = proxy_options(dict(preset='2m', batch=2, smooth_tau=3.0))
    trainer = Backprop(o, reference, cpu, 0, 10)
    loss = trainer.gradient(rows)
    got = [p.grad.clone() for p in trainer.model.parameters()]
    replay = Blur(trainer.model, 3.0, 0, cpu)  # the same generator stream, replayed by hand
    sums, clean_losses, blurred = [torch.zeros_like(p, dtype=torch.float32) for p in got], [], []
    for i in range(2):
        micro = rows[i:i + 1]
        clean_losses.append(replay.measure(lambda: trainer.loss_of(micro)))
        for sign in (1, -1):
            trainer.model.requires_grad_(True)
            with replay.shifted(sign):
                step_loss = trainer.loss_of(micro)
                for s, g in zip(sums, torch.autograd.grad(step_loss, list(trainer.model.parameters()))):
                    s += g.float()
            trainer.model.requires_grad_(False)
            blurred.append(step_loss.detach())
    assert torch.equal(loss, torch.stack(clean_losses).mean()), 'the logged loss is not the clean loss'
    assert torch.allclose(trainer.blurred, torch.stack(blurred).mean())
    for g, s, p in zip(got, sums, trainer.model.parameters()):
        assert torch.equal(g, (s / 4).to(p.dtype)), 'not the mean over sequences and both signs'
    assert not trainer.model.transformer.wte._forward_hooks, 'the blur left a hook behind'


def optimizer_state(trainer):
    """Copies of the weights, every optimizer state tensor (in parameter order) and the optimizers' clocks."""
    held = [p.detach().clone() for p in trainer.model.parameters()]
    for o in trainer.optimizer.optimizers:
        for p in trainer.model.parameters():
            held += [v.clone() for _, v in sorted(o.state.get(p, {}).items()) if torch.is_tensor(v)]
    return held, [getattr(o, 't', None) for o in trainer.optimizer.optimizers]


def same_tensors(a, b):
    return len(a[0]) == len(b[0]) and all(torch.equal(u, v) for u, v in zip(a[0], b[0]))


def variants_control(scratch, recipe='boostcut'):
    """E4: a rejected update leaves the weights and every optimizer tensor exactly as before (after the first update:
    as a fresh optimizer), while ANVIL II's schedule clock advances with the loop, so the next update equals a fresh
    trainer's at that clock; adapt's multiplier follows the outcomes; warmdown 0 holds the rates constant."""
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    reference = Reference(recipe, 1.0, '2m')
    patch_attention()
    cpu, rows = torch.device('cpu'), tokens(2)
    reject, accept = (lambda batch: torch.tensor(float('inf'))), (lambda batch: torch.tensor(-float('inf')))
    o = proxy_options(dict(recipe=recipe, preset='2m', batch=2, step_control='reject'))
    trainer = Backprop(o, reference, cpu, 0, 10)
    before = optimizer_state(trainer)
    trainer.batch_loss = reject
    _, fields = trainer.update(1, rows)
    assert fields['accepted'] is False and fields['trial_loss'] == float('inf')
    after = optimizer_state(trainer)
    weights = len(list(trainer.model.parameters()))
    assert all(torch.equal(u, v) for u, v in zip(before[0], after[0][:weights]))
    assert all(not t.any() for t in after[0][weights:]), 'not a fresh optimizer'
    assert after[1][0] == before[1][0] + 1, "ANVIL II's clock must follow the loop's updates"
    del trainer.batch_loss  # the real trial forward again
    _, fields = trainer.update(2, rows)
    fresh = Backprop(proxy_options(dict(recipe=recipe, preset='2m', batch=2)), reference, cpu, 0, 10)
    fresh.optimizer.optimizers[0].t = 1
    fresh.update(2, rows)
    assert fields['accepted'] and fields['trial_loss'] < 10, fields
    assert all(torch.equal(a, b) for a, b in zip(trainer.model.parameters(), fresh.model.parameters()))

    trainer.update(3, tokens(2, seed=2))
    held = optimizer_state(trainer)
    trainer.batch_loss = reject
    trainer.update(4, tokens(2, seed=3))
    later = optimizer_state(trainer)
    assert same_tensors(held, later), 'a mid-run rejection is not undone exactly'
    assert later[1][0] == held[1][0] + 1
    if recipe != 'boostcut':
        return

    o = proxy_options(dict(preset='2m', batch=2, step_control='adapt', accept_up=1.25, accept_down=0.5, warmdown=0))
    trainer = Backprop(o, reference, cpu, 0, 10)
    multipliers = []
    for step, judge in zip(range(1, 11), (accept, accept, reject, accept) * 3):
        trainer.batch_loss = judge
        _, fields = trainer.update(step, rows)
        multipliers.append(fields['lr_multiplier'])
        assert fields['lr']['anvil'] == reference.settings.anvil_lr * fields['lr_multiplier']  # constant schedule
    assert multipliers[:5] == [1.0, 1.25, 1.5625, 0.78125, 0.9765625]


def variants_control_pr40(scratch):
    """The same exact undo with PR #40's batched ANVIL II, whose per-matrix state is views into stacked tensors."""
    variants_control(scratch, 'pr40')


def variants_shaping(scratch):
    """E10 against a brute-force per-token computation: credits read with retain_grad on each nn.Linear output, every
    token's contribution formed and counted explicitly; and agree at threshold 0.5 keeps the plain gradient exactly."""
    from torch import nn
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    from loud.variants import TokenShaping
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    cpu, rows = torch.device('cpu'), tokens(2)
    plain = Backprop(proxy_options(dict(preset='2m', batch=2)), reference, cpu, 0, 10)
    plain.gradient(rows)
    plain_grads = {n: p.grad.clone() for n, p in plain.model.named_parameters()}
    model = plain.model
    linears = {n + '.weight': m for n, m in model.named_modules() if isinstance(m, nn.Linear)}

    credits, inputs = {n: [] for n in linears}, {n: [] for n in linears}
    for i in range(2):  # brute force: each sequence's credits at every linear output
        outputs, hooks = {}, []
        for n, m in linears.items():
            def hook(module, args, out, n=n):
                out.retain_grad()
                outputs[n] = out
                inputs[n].append(args[0].to(out.dtype).reshape(-1, args[0].shape[-1]).float())
            hooks.append(m.register_forward_hook(hook))
        model.requires_grad_(True)
        plain.loss_of(rows[i:i + 1]).backward()
        model.requires_grad_(False)
        model.zero_grad(set_to_none=True)
        for h in hooks:
            h.remove()
        for n, out in outputs.items():
            credits[n].append(out.grad.reshape(-1, out.shape[-1]).float())
    credits = {n: torch.cat(c) for n, c in credits.items()}
    inputs = {n: torch.cat(c) for n, c in inputs.items()}

    def shaped(**options):
        trainer = Backprop(proxy_options(dict(preset='2m', batch=2, **options)), reference, cpu, 0, 10)
        trainer.gradient(rows)
        return {n: p.grad for n, p in trainer.model.named_parameters()}

    def close(a, b):
        return (a - b).norm() <= 1e-2 * b.norm() + 1e-12

    got = shaped(token_shaping='cap', cap_quantile=0.9)
    for n in linears:
        norms = credits[n].norm(dim=-1)
        scale = (torch.quantile(norms, 0.9) / norms.clamp_min(1e-30)).clamp(max=1)
        assert (scale < 1).any() and (scale == 1).any()
        expected = torch.einsum('ti,tj->ij', credits[n] * scale[:, None], inputs[n]) / 2  # mean over 2 sequences
        assert close(got[n], expected), f'cap differs at {n}'
    for n in plain_grads:
        if n not in linears:
            assert torch.equal(got[n], plain_grads[n]), f'cap changed {n}, which is not a linear weight'

    for threshold, over in ((0.6, 'active'), (0.8, 'active'), (0.6, 'all')):
        got = shaped(token_shaping='agree', agree_threshold=threshold, agree_over=over)
        dropped = kept = 0
        for n in linears:
            positive = negative = torch.zeros(credits[n].shape[1], inputs[n].shape[1])
            for g, x in zip(credits[n], inputs[n]):  # one token at a time
                product = torch.outer(g, x)
                positive, negative = positive + (product > 0), negative + (product < 0)
            voters = positive + negative if over == 'active' else len(credits[n])  # 'all': every token votes
            mask = torch.maximum(positive, negative) >= threshold * voters
            assert torch.equal(got[n], plain_grads[n] * mask) or close(got[n], plain_grads[n] * mask), (n, over)
            dropped, kept = dropped + int((~mask).sum()), kept + int(mask.sum())
        assert dropped and kept, (threshold, over, dropped, kept)

    o = dict(proxy_options(dict(preset='2m', batch=2, token_shaping='agree')), agree_threshold=0.5)
    trainer = Backprop(o, reference, cpu, 0, 10)
    assert isinstance(trainer.shaping, TokenShaping)
    trainer.gradient(rows)
    for n, p in trainer.model.named_parameters():  # the hook's matmul is autograd's own, bit for bit
        assert torch.equal(p.grad, plain_grads[n]), f'threshold 0.5 changed {n}'


def variants_run(scratch):
    """The whole loop with each variant: the experiment series land in metrics.jsonl; the default trainer builds none
    of them."""
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference, run
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    patch_sequences()
    default = Backprop(proxy_options(dict(preset='2m', batch=2)), reference, torch.device('cpu'), 0, 10)
    assert default.blur is None and default.control is None and default.shaping is None
    for variant, args, fields in (('adapt', dict(step_control='adapt', warmdown=0), {'accepted', 'lr_multiplier'}),
                                  ('blur', dict(smooth_tau=3.0), {'blurred_loss'}),
                                  ('cap', dict(token_shaping='cap'), set()),
                                  ('agree', dict(token_shaping='agree'), set())):
        spec = dict(experiment='cpu', variant=variant, seed=0, task='proxy', args=dict(
            preset='2m', sequences=24, heldout=4, batch=4, official_eval=False, **args))
        out = Path(scratch) / variant
        result = run(spec, out)
        metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        assert result['complete'] and len(metrics) == 5 and all(fields <= set(m) for m in metrics), variant
        assert all(torch.isfinite(torch.tensor(m['train_loss'])) for m in metrics), variant


def proxy_trainer(**args):
    from loud.options import proxy_options
    from loud.proxy import Backprop, Reference
    reference = Reference('boostcut', 1.0, '2m')
    patch_attention()
    o = proxy_options(dict(dict(preset='2m', batch=4), **args))
    return Backprop(o, reference, torch.device('cpu'), 0, 10), reference.module


def batches(count, vocab, seed=6):
    return torch.randint(0, vocab, (count, 4, SEQ), generator=torch.Generator().manual_seed(seed))


def anvil_hook(scratch):
    """ShapedAnvil whose shaper returns polar(pending) steps exactly as boost-cut's Anvil, and leaves Anvil in place."""
    from loud.anvil import shaped_optimizer
    trainer, m = proxy_trainer()
    original = m.Anvil
    twin = copy.deepcopy(trainer.model)
    polar = type('Polar', (), dict(direction=lambda self, p, pending: m.polar(pending)))()
    shaped = shaped_optimizer(m, twin, trainer.reference.settings, 10, polar)
    assert m.Anvil is original and type(trainer.optimizer.optimizers[0]) is original
    for rows in batches(3, trainer.model.config.vocab_size):
        for model, optimizer in ((trainer.model, trainer.optimizer), (twin, shaped)):
            model.requires_grad_(True)
            model(rows[:, :-1].contiguous(), rows[:, 1:].contiguous()).backward()
            model.requires_grad_(False)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    for (name, a), b in zip(trainer.model.named_parameters(), twin.parameters()):
        assert torch.equal(a, b), f'{name}: ShapedAnvil does not reproduce Anvil'


def curvature_damping(scratch):
    """E8: the tracked directions are the pending update's top singular directions; h matches an exact Hessian-vector
    product; c is the across-sequence variance of the projections; the factors follow the formula; 'step' damps after
    polar() and 'pending' before it, where polar() undoes most of it."""
    from loud.curvature import CURVATURE_ROWS, damp
    trainer, m = proxy_trainer(per_sequence_grads=True, curvature_damping=3.0, damp_directions=2, damp_every=2)
    shaper, model = trainer.shaper, trainer.model
    rows = batches(3, model.config.vocab_size)
    trainer.update(1, rows[0])  # a refresh: at update 1, fast and slow are multiples of g, so pending is too
    params = dict(model.named_parameters())
    for i, name in enumerate(shaper.names):
        a, b = shaper.left[i], shaper.right[i]
        assert torch.allclose(a.T @ a, torch.eye(2), atol=1e-5) and torch.allclose(b.T @ b, torch.eye(2), atol=1e-5)
        u, _, vh = torch.linalg.svd(trainer.accumulators[list(params).index(name)], full_matrices=False)
        assert (u[:, :2] * a).sum(0).abs().min() > 0.999 and (vh[:2].T * b).sum(0).abs().min() > 0.999, name
        g = trainer.pergrad.stack(name).float()
        projections = torch.einsum('brc,rk,ck->bk', g, a, b)
        assert torch.allclose(shaper.noise[i], projections.var(0), rtol=1e-4)

    i = shaper.names.index('transformer.h.1.mlp.c_fc.weight')
    d = torch.outer(shaper.left[i][:, 0], shaper.right[i][:, 0])
    exact_model = copy.deepcopy(model).float().requires_grad_(True)
    weight = dict(exact_model.named_parameters())[shaper.names[i]]
    curvature_rows = rows[0][:CURVATURE_ROWS]
    with torch.enable_grad():
        loss = exact_model(curvature_rows[:, :-1].contiguous(), curvature_rows[:, 1:].contiguous())
        g = torch.autograd.grad(loss, weight, create_graph=True)[0]
        hvp = torch.autograd.grad((g * d).sum(), weight)[0]
    exact_h = (hvp * d).sum().item()
    assert abs(shaper.h[i][0].item() - exact_h) < 2e-2 * abs(exact_h) + 1e-6, (shaper.h[i][0].item(), exact_h)

    trainer.diagnose(1)
    trainer.update(2, rows[1])  # not a refresh: one more noise sample, factors from the formula
    lr = trainer.reference.settings.anvil_lr
    for i, name in enumerate(shaper.names):
        p = params[name]
        scale = 3.0 * lr ** 2 * max(1, p.size(0) / p.size(1))
        expected = 1 / (1 + (shaper.noise[i] / 2) / (4 * shaper.h[i].square() * scale))
        assert shaper.samples == 2 and torch.allclose(shaper.factors[i], expected, rtol=1e-5)
    trainer.diagnose(2)
    trainer.update(3, rows[2])  # a refresh again: new directions, and a summary with their overlap
    report = trainer.diagnose(3)['damping']
    assert {'h', 'c', 'h2_over_c', 'factor', 'overlap'} <= set(report['mlp.c_fc'])

    p = params[shaper.names[i]]
    pending = torch.randn(p.shape, generator=torch.Generator().manual_seed(7))
    a, b, f = shaper.left[i], shaper.right[i], shaper.factors[i]
    shaper.refreshing = False
    assert torch.equal(shaper.direction(p, pending), damp(m.polar(pending), a, b, f))
    shaper.where = 'pending'
    assert torch.equal(shaper.direction(p, pending), m.polar(damp(pending, a, b, f)))
    x = torch.randn(p.shape, generator=torch.Generator().manual_seed(8))
    coefficient = lambda y: (a.T @ y * b.T).sum(1)
    assert torch.allclose(coefficient(damp(x, a, b, f)), f * coefficient(x), atol=1e-5)
    rest = lambda y: y - a @ torch.diag(coefficient(y)) @ b.T
    assert torch.allclose(rest(damp(x, a, b, f)), rest(x), atol=1e-5)  # other directions are left alone

    # why 'step' is the default: halving a gradient-like matrix's top singular directions before polar() leaves them
    # at about 1 after it; halving them after polar() halves them
    U, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(9)))
    V, _ = torch.linalg.qr(torch.randn(128, 128, generator=torch.Generator().manual_seed(10)))
    G = U @ torch.diag(torch.logspace(0, -3, 128)) @ V.T
    a, b, half = U[:, :4], V[:, :4], torch.full((4,), 0.5)
    top = lambda y: (a.T @ y * b.T).sum(1)
    assert top(m.polar(damp(G, a, b, half))).min() > 0.9
    assert top(damp(m.polar(G), a, b, half)).max() < 0.6


def fisher_factor(scratch):
    """E9: the sums of d d^T come from the sampled-label gradient at each block matrix's output (checked against an
    independent forward, token by token), leave the training gradient untouched, and the update is the damped inverse
    root of S times the pending update at the polar factor's norm."""
    from loud.fisher import SAMPLE_SEED, inverse_root
    trainer, m = proxy_trainer(fisher_damping=0.1)
    plain, _ = proxy_trainer()
    shaper, model = trainer.shaper, trainer.model
    rows = batches(1, model.config.vocab_size)[0]
    trainer.gradient(rows)
    plain.gradient(rows)
    for (name, p), q in zip(model.named_parameters(), plain.model.parameters()):
        assert torch.equal(p.grad, q.grad), f'{name}: the sampled-label backward leaked into the training gradient'

    reference = copy.deepcopy(plain.model).requires_grad_(True)
    names = [n for n, p in reference.named_parameters() if n.startswith('transformer.h.') and p.ndim == 2
             and 've_gate' not in n]
    modules = dict(reference.named_modules())
    generator = torch.Generator().manual_seed(0 + SAMPLE_SEED)  # the job seed (0) + the label generator's offset
    expected = [0] * len(names)
    for micro in rows.split(1):
        outputs = {}
        hooks = [modules[n.removesuffix('.weight')].register_forward_hook(
            lambda mod, args, out, n=n: outputs.__setitem__(n, out)) for n in names]
        with torch.autocast('cpu', dtype=torch.bfloat16):
            logits = reference(micro[:, :-1].contiguous())
        for h in hooks:
            h.remove()
        sampled = torch.multinomial(logits.detach().softmax(-1).flatten(0, 1), 1, generator=generator)
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), sampled.flatten(), reduction='sum')
        deltas = torch.autograd.grad(loss, [outputs[n] for n in names])
        for j, d in enumerate(deltas):
            d = d[0].float()
            expected[j] = expected[j] + sum(torch.outer(d[t], d[t]) for t in range(d.shape[0]))  # token by token
    for j, name in enumerate(names):
        assert torch.allclose(shaper.batch_sum[j], expected[j], rtol=1e-3, atol=1e-6), name
    tokens = 4 * (SEQ - 1)
    assert shaper.tokens == tokens

    shaper.before_step(1)
    S, R = shaper.S[0], shaper.R[0]
    assert torch.allclose(S, expected[0] / tokens, rtol=1e-3, atol=1e-9)
    damped = S + 0.1 * torch.linalg.eigvalsh(S).clamp_min(0).mean() * torch.eye(len(S))
    assert torch.allclose(R @ damped @ R, torch.eye(len(S)), atol=1e-3)
    assert torch.allclose(inverse_root(S, 0.1), R)
    p = dict(model.named_parameters())[names[0]]
    pending = torch.randn(p.shape, generator=torch.Generator().manual_seed(11))
    update = shaper.direction(p, pending)
    assert abs(update.norm().item() - m.polar(pending).norm().item()) < 1e-3  # the step's size stays ANVIL II's
    assert torch.allclose(update / update.norm(), R @ pending / (R @ pending).norm(), atol=1e-6)


def population_run(scratch):
    """E3 end to end on a tiny model, once per rule: every remaining sequence feeds one copy's update at most once,
    the copies' rates and ANVIL II's clock follow the single run's update numbers, and the rule acts as described."""
    from loud import population
    m = refs.trainer_module('boostcut')
    patch_attention()
    patch_sequences(population)
    for rule in ('average', 'select', 'ensemble'):
        spec = dict(experiment='cpu', variant=rule, seed=0, task='population', args=dict(
            preset='2m', sequences=72, heldout=4, rule=rule, every=2, official_eval=False,
            checkpoints=rule == 'average'))
        out = Path(scratch) / rule
        result = population.run(spec, out)
        config = json.loads((out / 'config.json').read_text())
        metrics = [json.loads(line) for line in (out / 'metrics.jsonl').read_text().splitlines()]
        events = [json.loads(line) for line in (out / 'population.jsonl').read_text().splitlines()]
        # 68 update sequences: 17 updates of 4; 3 before the branch, then 56 left, 14 per copy: 3 rounds of 4 each
        assert config['branch_step'] == 3 and config['rounds'] == 3 and config['unused_sequences'] == 8
        assert [r['step'] for r in metrics] == list(range(1, 16)) and result['updates'] == 15
        turns = [(r['round'], r['copy']) for r in metrics if r['phase'] == 'population']
        assert turns == [(r, c) for r in (1, 2, 3) for c in range(4)]
        anvil = m.Anvil([torch.zeros(2, 2)], 1.0, 0.0, 17)
        for r in metrics:
            anvil.t = r['step'] - 1
            assert r['anvil_beta'] == anvil.rail_beta(), 'ANVIL II does not follow the single run'
        base = metrics[0]['lr']['anvil']
        assert all(abs(r['lr']['anvil'] - base * min(1.0, (17 - r['step'] + 1) / 17 / 0.3)) < 1e-12 for r in metrics)
        assert [e['round'] for e in events] == [2, 3] and all(len(e['midpoints']) == 6 for e in events)
        assert all(abs(e['barriers'][p] - (e['midpoints'][p] - max(e['copies'][int(p[0])], e['copies'][int(p[2])])))
                   < 1e-12 for e in events for p in e['midpoints'])
        if rule == 'average':
            assert all(e['action'] == 'average' for e in events) and result['final'] == 'mean'
            # H4: the trunk's fractions up to the branch, then every copy's 50% and its end (100%)
            saved = {p.name for p in out.glob('ckpt_*.pt')}
            assert saved == {'ckpt_00.pt', 'ckpt_10.pt', 'ckpt_20.pt'} | {
                f'ckpt_{f}.c{c}.pt' for f in (50, 100) for c in range(4)}, saved
        if rule == 'select':
            assert all(set(e['action']) == {'select', 'replace'} for e in events)
            last = events[-1]
            assert result['final'] == f'copy {last["action"]["select"]}'
            assert result['heldout_loss'] == min(result['final_copy_scores'])
        if rule == 'ensemble':
            mean, best = result['final_mean_score'], min(result['final_copy_scores'])
            assert result['final'] == ('mean' if mean < best else f'copy {result["final_copy_scores"].index(best)}')
            assert result['heldout_loss'] == min(mean, best)


def population_rules(scratch):
    """The rules on live copies: averaging leaves every copy at the mean with its own optimizer state; selection gives
    the worst copy the best's weights and an unaliased copy of its optimizer state."""
    from loud.heldout import HeldoutScorer
    from loud.common import JsonlLog
    from loud.population import Clock, Population
    trainer, m = proxy_trainer()
    rows = batches(3, trainer.model.config.vocab_size)
    for step in (1, 2):
        trainer.update(step, rows[step - 1])
    heldout = HeldoutScorer(rows[2], Path(scratch) / 'heldout.jsonl', torch.device('cpu'))
    events = JsonlLog(Path(scratch) / 'population.jsonl')
    o = dict(trainer.o, copies=3, rule='average')
    population = Population(o, trainer, trainer.reference, torch.device('cpu'), 0, heldout, events,
                            Clock(torch.device('cpu')))
    for c, member in enumerate(population.copies):  # make them differ
        member.optimizer.optimizers[0].t = 2 + c
        member.update(3 + c, rows[c])
    states = [copy.deepcopy(member.optimizer.optimizers[0].state_dict()) for member in population.copies]
    population.checkpoint(1, 5)
    weights = [list(member.model.parameters()) for member in population.copies]
    for ps in zip(*weights):
        assert all(torch.equal(ps[0], q) for q in ps[1:]), 'averaging left the copies apart'
    for member, state in zip(population.copies, states):
        assert all(torch.equal(member.optimizer.optimizers[0].state[p]['slow'], state['state'][i]['slow'])
                   for i, p in enumerate(member.optimizer.optimizers[0].param_groups[0]['params']))

    population.o = dict(o, rule='select')
    for c, member in enumerate(population.copies):
        member.update(6 + c, rows[(c + 1) % 3])
    scores = population.checkpoint(2, 8)
    best, worst = scores.index(min(scores)), scores.index(max(scores))
    for p, q in zip(population.copies[worst].model.parameters(), population.copies[best].model.parameters()):
        assert torch.equal(p, q)
    best_anvil = population.copies[best].optimizer.optimizers[0]
    kept = {id(p): best_anvil.state[p]['slow'].clone() for p in best_anvil.param_groups[0]['params']}
    population.copies[worst].update(9, rows[0])  # its state moves; the best copy's must not
    assert all(torch.equal(best_anvil.state[p]['slow'], kept[id(p)]) for p in best_anvil.param_groups[0]['params'])








def cut_compile_plan(scratch):
    """Every Cut config's plan: its chain's shapes, which the warmup compiles, fit the record's own Dynamo limit."""
    cut_module()
    from loud.cut import CutChain
    from loud.options import RANGES, cut_options
    from loud.run import specs_from_config
    configs = [p for p in sorted((Path(__file__).parents[1] / 'configs').rglob('*.yaml'))
               if 'task: cut' in p.read_text()]
    assert len(configs) >= 3, configs
    for path in configs:
        o = cut_options(specs_from_config(path)[0][0]['args'])
        n = RANGES.get(o['sequences'], o['sequences'])
        plan = CutChain(o, torch.device('cpu'), 0, (n - o['heldout']) // o['batch'],
                        torch.zeros(o['heldout'], CUT_SEQ, dtype=torch.long))
        assert 0 < len(plan.graphs) <= 17 and plan.recompile_limit == 64, (path, len(plan.graphs))


if __name__ == '__main__':
    torch.set_num_threads(2 if sys.argv[1] == 'cut_rank' else 8)  # every core would oversubscribe a shared machine
    torch.manual_seed(0)
    globals()[sys.argv[1]](*sys.argv[2:])
    print('ok')

"""E3's data split and schedule at the benchmark's size, and the option checks of E3, E8 and E9 (CPU, no training)."""
import pytest
import torch

from loud.options import branch_rows, population_options, proxy_options
from loud.population import global_step, shares


def test_population_split_uses_each_sequence_once():
    prefix, rest = branch_rows(population_options(dict(rule='average')))
    assert (prefix, rest) == (242, 3880)  # 10M less 32 held out: 1,212 updates of 4; the copies branch after 242
    used, unused, rounds = shares(rest, 4, 4)
    assert rounds == 242 and len(unused) == 8  # 970 sequences per copy: 242 batches of 4, 2 left
    every = torch.cat(used + [unused])
    assert len(every) == rest and len(set(every.tolist())) == rest
    assert all(torch.equal(u, torch.arange(c, rest, 4)[:968]) for c, u in enumerate(used))


def test_population_steps_follow_the_single_run():
    steps = [global_step(242, 4, r, c) for r in range(1, 243) for c in range(4)]
    assert steps == list(range(243, 1211))  # one update number each, in turn order, ending 2 short of the run's 1,212


@pytest.mark.parametrize('args, error', [
    (dict(curvature_damping=1.0), 'per_sequence_grads'),
    (dict(curvature_damping=1.0, per_sequence_grads=True, fisher_damping=0.1), 'one change per variant'),
    (dict(fisher_damping=0.1, recipe='pr40'), 'recipe: boostcut'),
    (dict(curvature_damping=1.0, per_sequence_grads=True, damp_where='before'), 'damp_where'),
    (dict(fisher_damping=-1.0), '>= 0'),
])
def test_proxy_variant_options(args, error):
    with pytest.raises(ValueError, match=error):
        proxy_options(args)


@pytest.mark.parametrize('args, error', [
    (dict(), 'rule must be'),
    (dict(rule='vote'), 'rule must be'),
    (dict(rule='average', checkpoints=True), 'at their defaults'),
    (dict(rule='average', fisher_damping=0.1), 'at their defaults'),
    (dict(rule='average', copies=1), 'at least two'),
    (dict(rule='average', heldout=0), 'held-out'),
    (dict(rule='average', branch=1.0), 'fraction'),
])
def test_population_options(args, error):
    with pytest.raises(ValueError, match=error):
        population_options(args)

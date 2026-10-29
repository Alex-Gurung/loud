"""E3's branch, the option checks of E2, E3, E4, E8 and E9, and H3's subsets (CPU, no training)."""
import pytest
import torch

from loud.heldout import HeldoutScorer
from loud.options import branch_step, population_options, proxy_options


def test_population_branches_at_the_plans_update():
    assert branch_step(population_options(dict(rule='average'))) == (121, 606)  # v2: update 121 of 606, batch 8


@pytest.mark.parametrize('args, error', [
    (dict(snr_scaling='snr'), 'per_sequence_grads'),
    (dict(snr_scaling='snr', per_sequence_grads=True, fisher=True), 'one change per variant'),
    (dict(fisher=True, recipe='pr40'), 'recipe: boostcut'),
    (dict(snr_scaling='svd'), 'snr_scaling one of'),
    (dict(blur=-0.1), '>= 0'),
    (dict(blur=0.1, step_control='reject'), 'one change per variant'),
    (dict(accept_up=1.2), 'acts only with step_control: adapt'),
    (dict(step_control='reject', heldout=2), 'heldout must be >= 4'),
])
def test_proxy_variant_options(args, error):
    with pytest.raises(ValueError, match=error):
        proxy_options(args)


@pytest.mark.parametrize('args, error', [
    (dict(), 'rule must be'),
    (dict(rule='vote'), 'rule must be'),
    (dict(rule='average', means=True), 'at their defaults'),
    (dict(rule='average', fisher=True), 'at their defaults'),
    (dict(rule='average', copies=1), 'at least two'),
    (dict(rule='average', copies=3), 'batch / copies'),
    (dict(rule='average', heldout=0), 'held-out'),
    (dict(rule='average', branch=1.0), 'fraction'),
])
def test_population_options(args, error):
    with pytest.raises(ValueError, match=error):
        population_options(args)


def test_heldout_subset_scores_and_logs_its_rows(tmp_path):
    class Mean(torch.nn.Module):  # a 'model' whose summed loss is the sum of its inputs' first tokens
        def forward(self, x, y, loss_reduction='mean'):
            return x[:, 0].float().sum() * x.shape[1] if loss_reduction == 'sum' else x[:, 0].float().mean()
    rows = torch.arange(8).repeat_interleave(3).view(8, 3)  # row i is all i
    scorer = HeldoutScorer(rows, tmp_path / 'heldout.jsonl', torch.device('cpu'))
    assert scorer.score(Mean(), 1, 'test', rows=[1, 3]) == 2.0 and scorer.score(Mean(), 1, 'test') == 3.5
    lines = (tmp_path / 'heldout.jsonl').read_text().splitlines()
    assert '"rows": [1, 3]' in lines[0] and '"rows"' not in lines[1]

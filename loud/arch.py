"""E16's second architecture: track-3 record #5's design (Noise PR #48, train/arch/model.py, pinned), a GPT whose
value embeddings, on every block, add a per-block projection of hashed bigram, trigram and skip-bigram tables. At the
'track3' preset it is the record's 12 x 1,024 shape. Built with PR #40's init on top of the model's own (the c_proj
writers uniform at the record's projection std, the head at its head std), as PR #40's make_model builds the fixed
model, so both train under one recipe in H8 and the proxy.
"""
import importlib.util
import math
import sys

import torch
from torch import nn

from . import refs

MODULE = 'noise_arch_model'


def module():
    """train/arch/model.py (self-contained: it imports only torch), loaded once by its pinned path."""
    if MODULE not in sys.modules:
        spec = importlib.util.spec_from_file_location(MODULE, model_file())
        sys.modules[MODULE] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules[MODULE])
    return sys.modules[MODULE]


def model_file():
    """The pinned model file, which the official scorer loads for track 3 (utils/eval.py --model)."""
    return refs.verified('noise', 'train/arch/model.py')


def config(n_layer, width, ngram_rows):
    return module().GPTConfig(n_layer=n_layer, n_head=width // 64, n_kv_head=width // 64, n_embd=width,
                              ngram_rows=ngram_rows)


def make_model(device, seed, config, projection_std, head_std):
    """PR #40's make_model with the track-3 GPT."""
    torch.manual_seed(seed)
    with torch.device('meta'):
        model = module().GPT(config)
    model.to_empty(device=device)
    model.init_weights()
    with torch.no_grad():
        for name, mod in model.named_modules():
            if 'c_proj' in name and isinstance(mod, nn.Linear):
                scale = projection_std * math.sqrt(3 / mod.weight.shape[1])
                nn.init.uniform_(mod.weight, -scale, scale)
        if head_std != 0.001:
            nn.init.normal_(model.lm_head.weight, std=head_std)
    return model.requires_grad_(False)


def ngram_features(model, idx):
    """The summed hashed n-gram embeddings the track-3 GPT projects into every block's value embedding (as its
    forward), or None for a model without them (the fixed one)."""
    rows = getattr(model.config, 'ngram_rows', 0)
    if not rows:
        return None
    pad = model.transformer.wte.num_embeddings
    return sum(table(module().ngram_index(idx, n, rows, pad)) for n, table in model.ngram_embeds.items())


def value_embedding(model, idx, i, ngram):
    """Block i's value embedding as the model's forward forms it (None where the block has none)."""
    if str(i) not in model.value_embeds:
        return None
    ve = model.value_embeds[str(i)](idx)
    return ve if ngram is None else ve + model.ngram_proj[str(i)](ngram)

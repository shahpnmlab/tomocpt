"""G3-C1 (v1 lock against the ``v2-gate-0-base`` golden) and the v1 part of G3-C9.

These tests must pass on HEAD before Gate 3 and keep passing after it.
"""

import json

import numpy as np
import pytest
import torch

from gate3 import g3_helpers as H

RTOL = 1e-6


@pytest.fixture(scope="module")
def golden():
    return json.loads((H.GOLDEN_DIR / "gate3_goldens.json").read_text())


def test_G3_C1_v1_training_step_loss_matches_base_golden(tiny_global, golden):
    """Tiny unconditioned single-class model, pipeline v1, dropout 0: the loss that
    ``training_step`` returns on a fixed batch equals the ``v2-gate-0-base`` golden."""
    from tomocpt.networks.pickingModel import BasePickingModel

    g3 = H.load_common()
    batch = g3.v1_batch()
    # the inputs are rebuilt, not stored: check they are the ones the golden was made from
    assert float(batch["input_data"].astype(np.float64).sum()) == pytest.approx(golden["G3_C1_input_sum"], rel=1e-9)
    assert float(batch["target_data"].astype(np.float64).sum()) == pytest.approx(golden["G3_C1_target_sum"], rel=1e-9)

    torch.manual_seed(0)
    model = BasePickingModel(lr=1e-3, train_continue=None)
    assert not [n for n, _ in model.named_parameters() if ".cond." in f".{n}"], \
        "conditioning modules exist although scale_conditioning is off (invariant 5)"
    g3.reinit_(model.model)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(batch["input_data"]))
    assert tuple(logits.shape) == (2, 1, 64, 64, 64)
    assert float(logits.double().sum()) == pytest.approx(golden["G3_C1_logits_sum"], rel=1e-5)
    model.train()

    loss = g3.first_training_step_loss(model, batch)
    assert loss == pytest.approx(golden["G3_C1_training_step_loss"], rel=RTOL, abs=0)


def test_G3_C1_v1_with_several_single_class_datasets_keeps_one_channel(tiny_global):
    """Pipeline v1 merges every dataset into one heatmap: a legacy config with several
    single-class datasets (which gives a multi-row class table) still builds one channel."""
    from tomocpt.networks.pickingModel import BasePickingModel

    tiny_global.prepData.particle_length_ang = "120,300"
    tiny_global.prepData.class_id = "1,2"
    model = BasePickingModel(lr=1e-3, train_continue=None).eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 1, 64, 64, 64))
    assert tuple(logits.shape) == (1, 1, 64, 64, 64)


def _names(model):
    return {id(p): n for n, p in model.named_parameters()}


def test_G3_C9_v1_parameter_groups_unchanged(tiny_global):
    """v1 grouping (CLAUDE.md 1.6): from scratch one group with every trainable parameter;
    fine-tuning: swinViT x0.01, encoder* x0.1, decoder* x1.0, then the rest (``out``) x1.0."""
    from tomocpt.networks.pickingModel import BasePickingModel

    scratch = BasePickingModel(lr=1e-3, train_continue=None)
    groups = H.param_groups(scratch)
    assert len(groups) == 1
    trainable = [p for p in scratch.parameters() if p.requires_grad]
    got = [id(p) for p in groups[0][1]]
    assert len(got) == len(set(got)) == len(trainable)
    assert set(got) == {id(p) for p in trainable}

    ft = BasePickingModel(lr=1e-3, train_continue="some/previous.ckpt")
    names = _names(ft)
    groups = H.param_groups(ft)
    assert [m for m, _ in groups] == [0.01, 0.1, 1.0, 1.0]
    seen = []
    for mult, params in groups:
        for p in params:
            seen.append(id(p))
            n = names[id(p)][len("model."):]
            if n.startswith("swinViT."):
                want = 0.01
            elif "encoder" in n:
                want = 0.1
            else:
                want = 1.0
            assert mult == want, (n, mult)
    assert len(seen) == len(set(seen))
    assert set(seen) == {id(p) for p in ft.parameters() if p.requires_grad}
    rest = [names[id(p)] for p in groups[3][1]]
    assert rest and all(n.startswith("model.out.") for n in rest), rest

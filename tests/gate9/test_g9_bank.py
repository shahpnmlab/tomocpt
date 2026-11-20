"""G9-C1 (bank part), G9-C4: memory-bank age limit (``train.discriminative.bank_max_age``) and
checkpoint compatibility of the bank (invariant 2, strict loading)."""

import torch

from gate3 import g3_helpers as G3
from utils.tiny import applied_main_config

DIM = 4


def _vec(k, label_scale=1.0):
    return torch.full((k, DIM), float(label_scale))


def _bank(max_age=0, n_classes=2, size=8):
    from tomocpt.training.discriminative import ContrastiveMemoryBank

    return ContrastiveMemoryBank(n_classes, size, DIM, max_age=max_age) if max_age else \
        ContrastiveMemoryBank(n_classes, size, DIM)


def _fill(bank, steps):
    """One entry of class 0 per step, value = step; one negative at the first step."""
    for s in steps:
        bank.push(_vec(1, s), torch.tensor([0]), step=s)
    bank.push(_vec(1, -1), torch.tensor([-1]), step=steps[0])


def test_G9_C1_default_bank_contents_unchanged():
    """With max_age = 0 the step arguments change nothing: get(step) == get() == legacy order."""
    a, b = _bank(), _bank()
    torch.manual_seed(0)
    e = torch.randn(20, DIM)
    lab = torch.tensor([0, 1, -1, 0] * 5)
    a.push(e, lab)
    b.push(e, lab, step=3)
    ea, la = a.get()
    eb, lb = b.get(step=10_000)
    assert torch.equal(ea, eb) and torch.equal(la, lb)
    for k in ("queue", "ptr", "count"):
        assert torch.equal(a.state_dict()[k], b.state_dict()[k])


def test_G9_C4_age_exclusion():
    bank = _bank(max_age=5)
    _fill(bank, [0, 2, 4, 6])
    e, lab = bank.get(step=6)  # ages 6, 4, 2, 0 (class 0) and 6 (negative)
    vals = sorted(float(v) for v in e[lab == 0, 0])
    assert vals == [2.0, 4.0, 6.0]
    assert int((lab == -1).sum()) == 0
    e, lab = bank.get(step=5)  # ages 5, 3, 1 (and entry 6 from the 'future' counts as age <= 5)
    assert sorted(float(v) for v in e[lab == 0, 0]) == [0.0, 2.0, 4.0, 6.0]
    assert int((lab == -1).sum()) == 1
    # without a step, everything
    e, lab = bank.get()
    assert len(e) == 5
    e, lab = bank.get(step=100)
    assert len(e) == 0 and e.shape == (0, DIM)


def test_G9_C4_state_dict_keys_unchanged():
    for bank in (_bank(), _bank(max_age=3)):
        assert set(bank.state_dict()) == {"queue", "ptr", "count"}


def test_G9_C4_old_bank_state_loads_strictly_and_is_stamped_at_first_use():
    old = _bank()
    old.push(_vec(3, 1.0), torch.tensor([0, 1, -1]))
    sd = {k: v.clone() for k, v in old.state_dict().items()}  # what a Gate 7 checkpoint holds
    new = _bank(max_age=5)
    new.load_state_dict(sd, strict=True)
    # restored entries have unknown age: stamped with the first step they are used at
    e, lab = new.get(step=100)
    assert len(e) == 3
    e, lab = new.get(step=105)
    assert len(e) == 3
    e, lab = new.get(step=106)
    assert len(e) == 0
    # and a new bank's state loads into an old-style (max_age 0) bank
    _bank().load_state_dict(new.state_dict(), strict=True)


def test_G9_C4_restored_entries_stamped_by_push_too():
    old = _bank()
    old.push(_vec(2, 1.0), torch.tensor([0, 0]))
    new = _bank(max_age=5)
    new.load_state_dict(old.state_dict(), strict=True)
    new.push(_vec(1, 2.0), torch.tensor([1]), step=50)
    e, lab = new.get(step=56)
    assert len(e) == 0
    e, lab = new.get(step=55)
    assert len(e) == 3


CLASSES = ((1, "A", 150.0), (2, "B", 300.0))


def _model(tmp_path, **disc):
    from tomocpt.networks.pickingModel import BasePickingModel

    cfg = G3.v2_config(tmp_path, CLASSES, cond=False)
    cfg.train.discriminative.contrastive_weight = 0.5
    for k, v in disc.items():
        setattr(cfg.train.discriminative, k, v)
    with applied_main_config(cfg):
        torch.manual_seed(0)
        return BasePickingModel(lr=4e-4)


def test_G9_C4_model_state_dict_keys_and_strict_load(tmp_path):
    off = _model(tmp_path / "a")
    on = _model(tmp_path / "b", bank_max_age=10, contrastive_warmup_steps=5)
    assert set(off.state_dict()) == set(on.state_dict())
    assert not any("stamp" in k for k in on.state_dict())
    on.load_state_dict(off.state_dict(), strict=True)
    off.load_state_dict(on.state_dict(), strict=True)

"""Volume snapshots for scripts/train_dashboard.py: chunk choice and the opt-in switch."""
from tomocpt.defaultConfigs.train_config import TrainConfig
from tomocpt.training.snapshots import pick_chunks


def test_snapshots_off_by_default():
    assert TrainConfig().snapshot_chunks == 0


def test_pick_chunks_per_class_spread_and_unique():
    sets = [{0}, {0, 1}, {1}, set(), {0}, {1}, {0}]
    chosen = pick_chunks(sets, 2, 2)
    assert chosen == [(0, 0), (6, 0), (1, 1), (5, 1)]
    assert len({i for i, _ in chosen}) == len(chosen)
    assert all(c in sets[i] for i, c in chosen)


def test_pick_chunks_missing_class_and_few_chunks():
    assert pick_chunks([{0}], 2, 3) == [(0, 0)]
    assert pick_chunks([], 2, 1) == []

"""Gate 5b config (spec section "Config"): new ``SSLConfig`` fields and changed v2 defaults.

The spec gives the defaults as plain values; the test reads them on a fresh ``SSLConfig`` and on
its resolved form (``g5_helpers.resolved``: ``resolved()`` / ``resolve()`` if present).
"""

import pytest

from gate5 import g5_helpers as H

NEW_FIELDS = dict(view_lowpass_min=0.35, view_lowpass_max=0.6, view_noise_snr_min=0.5,
                  view_noise_snr_max=1.0, collapse_reg="vicreg", variance_weight=25.0,
                  covariance_weight=1.0, queue_size=1024)
CHANGED = dict(dense_grad_scale=1.0, recon_weight=1.0)


@pytest.mark.parametrize("version", [None, 2])
def test_G5b_CFG_new_fields_and_changed_defaults(version):
    kw = {} if version is None else dict(version=version)
    for cfg in (H.ssl_cfg(**kw), H.resolved(H.ssl_cfg(**kw))):
        for name, want in {**NEW_FIELDS, **CHANGED}.items():
            got = getattr(cfg, name)
            if isinstance(want, str):
                assert str(getattr(got, "value", got)) == want, (name, got)
            else:
                assert got == pytest.approx(want), (name, got)
                assert type(got) is type(want), (name, type(got))


def test_G5b_CFG_unchanged_gate5_fields():
    """Gate 5 settings that Gate 5b keeps (``recon_visible_weight`` 0.1 for v2, version 1 default)."""
    from tomocpt.mainConfig import MainConfig

    assert MainConfig().train.ssl.version == 1
    v2 = H.resolved(H.ssl_cfg(version=2))
    assert v2.recon_visible_weight == pytest.approx(0.1)
    assert v2.use_rotation is False


def test_G5b_CFG_collapse_reg_none_accepted():
    cfg = H.resolved(H.ssl_cfg(version=2, collapse_reg="none"))
    assert str(getattr(cfg.collapse_reg, "value", cfg.collapse_reg)) == "none"

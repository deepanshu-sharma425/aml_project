import numpy as np
import pandas as pd
import pytest

from traffic import cleaning as C
from traffic.data import DATA_DIR, load_locations


def test_detect_stuck_keeps_first_reading_and_cap():
    s = np.full((40, 2), 70.0, dtype=np.float32)            # sensor 1: 70 mph cap all along -> fine
    s[:, 0] = np.r_[np.arange(10) + 50, [46.0] * 20, np.arange(10) + 50]
    stuck = C.detect_stuck(s, min_steps=12)
    assert stuck[:, 0].sum() == 19 and not stuck[10, 0] and stuck[11:30, 0].all()
    assert not stuck[:, 1].any()


def test_detect_spikes_flags_isolated_drop_not_congestion_onset():
    s = np.full((60, 2), 65.0)
    s[30, 0] = 11.0                                          # one-step glitch
    s[30:, 1] = 15.0                                         # real, persistent slowdown
    spikes = C.detect_spikes(s + np.random.default_rng(0).normal(0, 1, s.shape))
    assert spikes[30, 0] and spikes[:, 0].sum() == 1
    assert not spikes[:, 1].any()


def test_gap_lengths_and_short_interpolation():
    x = np.array([[1.0], [np.nan], [np.nan], [4.0], [np.nan], [np.nan], [np.nan], [np.nan], [9.0], [np.nan]])
    assert C.gap_lengths(np.isnan(x))[:, 0].tolist() == [0, 2, 2, 0, 4, 4, 4, 4, 0, 1]
    y = C.interpolate_short(x, max_steps=3)
    assert np.allclose(y[:4, 0], [1, 2, 3, 4])
    assert np.isnan(y[4:8, 0]).all()                         # too long: left for the profile
    assert np.isnan(y[9, 0])                                 # trailing gap: no right-hand anchor


def test_clean_synthetic_codes_and_fills():
    idx = pd.date_range("2012-03-05", periods=2 * 2016, freq="5min")
    rng = np.random.default_rng(1)
    s = (60 + rng.normal(0, 2, (len(idx), 3))).astype(np.float32)
    s[100:103] = 0                                           # network outage, 3 steps
    s[500:505, 0] = 0                                        # short gap -> linear
    s[700:800, 1] = 0                                        # long gap -> profile
    s[900, 2] = 5.0                                          # spike
    raw = s.copy()
    r = C.clean(s, idx, train_end=2016)
    assert np.array_equal(s, raw)                            # input untouched
    assert (r.issue[100:103] == C.OUTAGE).all()
    assert (r.issue[500:505, 0] == C.MISSING).all() and (r.filled[500:505, 0] == C.LINEAR).all()
    assert (r.filled[700:800, 1] == C.PROFILE).all()
    assert r.issue[900, 2] == C.SPIKE and abs(r.speed[900, 2] - 60) < 10
    assert not np.isnan(r.speed).any()


def test_profile_fill_uses_training_period_only():
    idx = pd.date_range("2012-03-05", periods=2 * 2016, freq="5min")
    s = np.full((len(idx), 1), 60.0, dtype=np.float32)
    s[2016:] = 30.0                                          # post-training regime change
    s[3000:3100] = 0                                         # long gap after training
    r = C.clean(s, idx, train_end=2016)
    assert np.allclose(r.speed[3000:3100], 60.0)             # filled from training weeks, not 30


@pytest.fixture(scope="module")
def cleaned(data):
    return C.clean(data.speed, data.index, data.train_end)


def test_real_data_validation_passes(data):
    df = pd.DataFrame(data.speed, index=data.index, columns=data.sensor_ids)
    dist = pd.read_csv(DATA_DIR / "distances_la_2012.csv", dtype={"from": str, "to": str})
    failed = [c for c in C.validate(df, data.sensor_ids, load_locations(), dist) if not c["passed"]]
    assert not failed, failed


def test_real_data_cleaning_invariants(data, cleaned):
    r = cleaned
    assert r.speed.shape == data.speed.shape and not np.isnan(r.speed).any()
    assert r.speed.min() > 0 and r.speed.max() <= 70
    ok = r.issue == C.OK
    assert np.array_equal(r.speed[ok], data.speed[ok])       # valid readings never change
    assert ((data.speed == 0) <= (r.issue != C.OK)).all()   # every zero is flagged
    assert ((r.filled == C.RAW) == ok).all()
    assert (r.issue[C.detect_outages(data.speed)] == C.OUTAGE).all()
    assert int(C.detect_outages(data.speed).sum()) == 2148
    assert 8.1 < 100 * (~ok).mean() < 8.5                    # 8.1 % zeros + a few stuck/spikes


def test_models_still_see_raw_data(data):
    """Cleaning is analysis-only: the benchmark protocol keeps raw speeds with zeros masked."""
    assert (data.speed == 0).mean() == pytest.approx(0.0811, abs=1e-3)


def test_save_and_load_roundtrip(tmp_path, data, cleaned):
    path = C.save_clean(cleaned, data.index, data.sensor_ids, tmp_path / "clean.h5")
    df, issue, filled = C.load_clean(path)
    assert list(df.columns) == data.sensor_ids and df.index.equals(data.index)
    assert np.array_equal(df.to_numpy(), cleaned.speed)
    assert np.array_equal(issue, cleaned.issue) and np.array_equal(filled, cleaned.filled)

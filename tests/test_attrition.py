import numpy as np, pandas as pd, pytest
from datamoat.attrition import (AttritionConfig, build_snapshots, chronological_split,
    run_attrition, choose_threshold, calibration_table, evaluate)
from tests.synth import make_data

@pytest.fixture(scope="module")
def data(): return make_data()

def test_labels_and_censoring(data):
    cfg = AttritionConfig()
    snaps, end = build_snapshots(data, cfg)
    stu = data["students"].set_index("student_id")
    h = pd.Timedelta(days=cfg.horizon_days)
    for r in snaps.sample(300, random_state=1).itertuples():
        drop = pd.to_datetime(stu.loc[r.student_id, "dropout_date"])
        if pd.notna(drop):
            assert r.snapshot < drop                      # no snapshot after leaving
            assert r.label == (1.0 if drop <= r.snapshot + h else 0.0)
        elif r.snapshot + h > end:
            assert np.isnan(r.label)                      # unknown outcome is censored, not 0
        else:
            assert r.label == 0.0

def test_features_ignore_future_data(data):
    """Dropping all rows after a snapshot must not change that snapshot's features."""
    cfg = AttritionConfig()
    snaps, end = build_snapshots(data, cfg)
    cut = snaps["snapshot"].quantile(0.5)
    trunc = {k: v.copy() for k, v in data.items()}
    for name, col in (("attendance","date"),("engagement","date")):
        trunc[name] = trunc[name][pd.to_datetime(trunc[name][col]) <= cut]
    trunc["fees"] = trunc["fees"][pd.to_datetime(trunc["fees"]["due_date"]) <= cut]
    trunc["tests"] = trunc["tests"][pd.to_datetime(trunc["tests"]["test_date"]) <= cut]
    trunc["test_responses"] = trunc["test_responses"][trunc["test_responses"]["test_id"].isin(trunc["tests"]["test_id"])]
    s2, _ = build_snapshots(trunc, cfg)
    a = snaps[snaps.snapshot <= cut].set_index(["student_id","snapshot"]).drop(columns="label")
    b = s2[s2.snapshot <= cut].set_index(["student_id","snapshot"]).drop(columns="label")
    common = a.index.intersection(b.index)
    # fee paid_date after cut is unknown at cut either way; compare non-fee features
    cols = [c for c in a.columns if not c.startswith("fee_")]
    pd.testing.assert_frame_equal(a.loc[common, cols].sort_index(), b.loc[common, cols].sort_index(), check_exact=False)

def test_split_is_chronological_and_purged(data):
    cfg = AttritionConfig()
    snaps, _ = build_snapshots(data, cfg); snaps = snaps[snaps.label.notna()]
    tr, va, te, (t1, t2) = chronological_split(snaps, cfg)
    gap = pd.Timedelta(days=cfg.horizon_days)
    assert tr.snapshot.max() + gap <= t1 + pd.Timedelta(days=1) and tr.snapshot.max() < va.snapshot.min()
    assert va.snapshot.max() < te.snapshot.min()
    assert va.snapshot.max() + gap <= t2

def test_pipeline_runs_and_reports_metrics(data):
    r = run_attrition(data)
    assert r.status == "ok" and r.model_version
    for k in ("pr_auc","roc_auc","brier","precision","recall","confusion"): assert k in r.metrics
    assert r.baseline_metrics["pr_auc"] is not None
    assert r.predictions["probability"].between(0, 1).all()
    assert not set(r.predictions.student_id) & set(data["students"].dropna(subset=["dropout_date"]).student_id)

def test_model_beats_base_rate_on_synthetic(data):
    r = run_attrition(data)
    assert r.metrics["roc_auc"] > 0.6   # pipeline sanity only; says nothing about real data

def test_insufficient_without_departures(data):
    d = {k: v.copy() for k, v in data.items()}; d["students"]["dropout_date"] = None
    r = run_attrition(d); assert r.status == "insufficient" and not r.validated

def test_empty_dataset():
    r = run_attrition({"students": pd.DataFrame(columns=["student_id"])})
    assert r.status == "insufficient"

def test_small_data_not_validated():
    r = run_attrition(make_data(n=60, seed=3))
    assert r.status == "insufficient" or not r.validated

def test_missing_fees_are_nan_not_risk():
    d = make_data(n=120, with_fees=False); snaps, _ = build_snapshots(d, AttritionConfig())
    assert snaps["fee_overdue_days"].isna().all()

def test_threshold_respects_capacity():
    rng = np.random.default_rng(0); p = rng.random(500); y = (rng.random(500) < p*0.5).astype(int)
    t = choose_threshold(y, p, 0.2); assert (p >= t).mean() <= 0.21

def test_calibration_table_and_metrics():
    y = np.array([0,0,1,1,0,1,0,0,1,0]); p = np.linspace(.05,.95,10)
    assert len(calibration_table(y, p, 5)) == 5
    assert evaluate(y, p, .5)["confusion"]["tp"] + evaluate(y, p, .5)["confusion"]["fn"] == y.sum()

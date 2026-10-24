"""Student-departure model: P(student leaves within H days | data known today).

Pure pandas / scikit-learn, no database access. Pipeline:

1. Label   : dropout_date inside (snapshot, snapshot + H]  -> 1
             dropout_date later than snapshot + H, or none and the window is
             fully observed                                 -> 0
             no dropout recorded but window not fully observed -> censored
             (excluded from training, never treated as "stayed").
2. Snapshot: one row per student every `step_days`, features use data <= snapshot.
3. Split   : chronological train / validation / test with a purge gap of H days
             so label windows never overlap the next period.
4. Model   : regularised logistic regression; HistGradientBoosting is evaluated
             only as a comparison. Platt calibration is fitted on validation.
5. Gate    : the model is flagged "validated" only with enough real outcomes.

IMPORTANT: a student with no recorded dropout_date is assumed retained once the
window is observed. If the institute does not record withdrawals reliably the
labels are wrong, and so is the model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES_BASE = [
    "att_7", "att_14", "att_30", "att_trend", "consec_absent",
    "score_overall", "score_recent3", "score_slope", "tests_90d",
    "hw_30", "days_since_activity", "logins_30", "logins_change", "doubts_30",
    "tenure_days",
]
FEATURES_FEES = ["fee_overdue_days", "fee_overdue_amount"]

FEATURE_LABELS = {
    "att_7": "Attendance, last 7 days",
    "att_14": "Attendance, last 14 days",
    "att_30": "Attendance, last 30 days",
    "att_trend": "Attendance change vs previous 30 days",
    "consec_absent": "Consecutive missed sessions",
    "score_overall": "Overall test average",
    "score_recent3": "Last 3 tests average",
    "score_slope": "Test score trend",
    "tests_90d": "Tests taken, last 90 days",
    "hw_30": "Homework completion, last 30 days",
    "days_since_activity": "Days since last recorded activity",
    "logins_30": "App logins, last 30 days",
    "logins_change": "Login change vs previous 30 days",
    "doubts_30": "Doubts asked, last 30 days",
    "tenure_days": "Days enrolled",
    "fee_overdue_days": "Days a fee instalment has been overdue",
    "fee_overdue_amount": "Overdue fee amount",
}

MIN_TRAIN_POSITIVES = 20
MIN_VAL_POSITIVES = 5
MIN_TEST_POSITIVES = 10


@dataclass
class AttritionConfig:
    horizon_days: int = 60
    step_days: int = 14
    min_tenure_days: int = 30
    use_fees: bool = True
    max_alert_rate: float = 0.20  # share of students the institute can follow up
    train_frac: float = 0.60
    val_frac: float = 0.20
    c_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


@dataclass
class AttritionResult:
    status: str  # "ok" | "insufficient"
    message: str
    validated: bool = False
    model_version: str | None = None
    horizon_days: int = 60
    as_of: date | None = None
    threshold: float | None = None
    features: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    gb_metrics: dict[str, Any] = field(default_factory=dict)
    calibration_bins: pd.DataFrame = field(default_factory=pd.DataFrame)
    coefficients: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    snapshot_summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    feature_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    watchlist: pd.DataFrame = field(default_factory=pd.DataFrame)


# ----------------------------------------------------------------------------
# Feature engineering (only data <= snapshot date)
# ----------------------------------------------------------------------------
def _dates(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce")


def _as_bool(series: pd.Series) -> pd.Series:
    return series.map(
        lambda v: bool(v) if isinstance(v, (bool, np.bool_))
        else str(v).strip().casefold() in {"1", "true", "yes", "present"}
    ).astype(float)


class _StudentHistory:
    """Pre-sorted per-student arrays so each snapshot is a cheap slice."""

    def __init__(self, att, tests, eng, fees):
        self.att_d = att["_d"].values.astype("datetime64[D]") if len(att) else np.array([], dtype="datetime64[D]")
        self.att_p = att["_p"].values.astype(float) if len(att) else np.array([])
        self.tst_d = tests["_d"].values.astype("datetime64[D]") if len(tests) else np.array([], dtype="datetime64[D]")
        self.tst_s = tests["_s"].values.astype(float) if len(tests) else np.array([])
        self.eng_d = eng["_d"].values.astype("datetime64[D]") if len(eng) else np.array([], dtype="datetime64[D]")
        self.eng_hw = pd.to_numeric(eng["homework_submitted"], errors="coerce").values.astype(float) if len(eng) else np.array([])
        self.eng_logins = pd.to_numeric(eng["app_logins"], errors="coerce").values.astype(float) if len(eng) else np.array([])
        self.eng_doubts = pd.to_numeric(eng["doubts_asked"], errors="coerce").values.astype(float) if len(eng) else np.array([])
        self.has_fees = len(fees) > 0
        self.fee_due = fees["_due"].values.astype("datetime64[D]") if len(fees) else np.array([], dtype="datetime64[D]")
        self.fee_paid = fees["_paid"].values.astype("datetime64[D]") if len(fees) else np.array([], dtype="datetime64[D]")
        self.fee_amt = pd.to_numeric(fees["amount"], errors="coerce").values.astype(float) if len(fees) else np.array([])


def _window_mean(dates, values, start, end):
    """Mean of values with start < date <= end; NaN if none."""
    if len(dates) == 0:
        return np.nan
    mask = (dates > start) & (dates <= end)
    vals = values[mask]
    vals = vals[~np.isnan(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def _features_at(h: _StudentHistory, snap: np.datetime64, join: np.datetime64,
                 use_fees: bool) -> dict[str, float]:
    day = np.timedelta64(1, "D")
    s = snap
    f: dict[str, float] = {}
    for w in (7, 14, 30):
        f[f"att_{w}"] = _window_mean(h.att_d, h.att_p, s - w * day, s)
    prior = _window_mean(h.att_d, h.att_p, s - 60 * day, s - 30 * day)
    f["att_trend"] = f["att_30"] - prior if not (np.isnan(f["att_30"]) or np.isnan(prior)) else np.nan

    if len(h.att_d):
        known = h.att_d <= s
        order = np.argsort(h.att_d[known], kind="stable")
        recent = h.att_p[known][order]
        run = 0
        for value in recent[::-1]:
            if value == 0:
                run += 1
            else:
                break
        f["consec_absent"] = float(run) if len(recent) else np.nan
    else:
        f["consec_absent"] = np.nan

    known_t = h.tst_d <= s if len(h.tst_d) else np.array([], dtype=bool)
    if known_t.any():
        order = np.argsort(h.tst_d[known_t], kind="stable")
        scores = h.tst_s[known_t][order]
        dts = h.tst_d[known_t][order]
        f["score_overall"] = float(scores.mean())
        f["score_recent3"] = float(scores[-3:].mean())
        last = scores[-5:]
        if len(last) >= 3:
            f["score_slope"] = float(np.polyfit(np.arange(len(last)), last, 1)[0])
        else:
            f["score_slope"] = np.nan
        f["tests_90d"] = float(((dts > s - 90 * day)).sum())
    else:
        f.update(score_overall=np.nan, score_recent3=np.nan, score_slope=np.nan, tests_90d=np.nan)

    f["hw_30"] = _window_mean(h.eng_d, h.eng_hw, s - 30 * day, s)
    f["logins_30"] = _window_mean(h.eng_d, h.eng_logins, s - 30 * day, s)
    prior_logins = _window_mean(h.eng_d, h.eng_logins, s - 60 * day, s - 30 * day)
    f["logins_change"] = (
        f["logins_30"] - prior_logins
        if not (np.isnan(f["logins_30"]) or np.isnan(prior_logins)) else np.nan
    )
    f["doubts_30"] = _window_mean(h.eng_d, h.eng_doubts, s - 30 * day, s)

    activity_days = []
    if len(h.eng_d):
        active = ((np.nan_to_num(h.eng_logins) > 0) | (np.nan_to_num(h.eng_doubts) > 0)) & (h.eng_d <= s)
        if active.any():
            activity_days.append(h.eng_d[active].max())
    if len(h.att_d):
        present = (h.att_p == 1) & (h.att_d <= s)
        if present.any():
            activity_days.append(h.att_d[present].max())
    f["days_since_activity"] = float((s - max(activity_days)) / day) if activity_days else np.nan
    f["tenure_days"] = float((s - join) / day)

    f["fee_overdue_days"] = np.nan  # unknown unless fee records exist; never treated as risk
    f["fee_overdue_amount"] = np.nan
    if use_fees:
        if h.has_fees:
            due_mask = h.fee_due <= s
            unpaid = np.isnat(h.fee_paid) | (h.fee_paid > s)
            overdue = due_mask & unpaid
            if overdue.any():
                f["fee_overdue_days"] = float(((s - h.fee_due[overdue]) / day).max())
                f["fee_overdue_amount"] = float(np.nansum(h.fee_amt[overdue]))
            else:
                f["fee_overdue_days"] = 0.0
                f["fee_overdue_amount"] = 0.0
    return f


def rule_score(row: pd.Series) -> float:
    """The existing rule-based support score, reproduced for baseline comparison."""
    points = 0.0
    seen = False
    score, recent, att, hw = (row.get("score_overall"), row.get("score_recent3"),
                              row.get("att_30"), row.get("hw_30"))
    if pd.notna(score):
        seen = True
        if score < 0.45:
            points += 28
        if pd.notna(recent) and score - recent >= 0.10:
            points += 22
    if pd.notna(att):
        seen = True
        if att < 0.70:
            points += 35
        elif att < 0.80:
            points += 18
    if pd.notna(hw):
        seen = True
        if hw < 0.60:
            points += 15
    return points if seen else 0.0


# ----------------------------------------------------------------------------
# Snapshots and labels
# ----------------------------------------------------------------------------
def _prepare(data: dict[str, pd.DataFrame]):
    students = data["students"].copy()
    students["student_id"] = students["student_id"].astype(str)
    students["_join"] = _dates(students, "join_date")
    students["_drop"] = _dates(students, "dropout_date") if "dropout_date" in students else pd.NaT

    tests = data.get("tests", pd.DataFrame()).copy()
    resp = data.get("test_responses", pd.DataFrame()).copy()
    scores = pd.DataFrame(columns=["student_id", "_d", "_s"])
    if not tests.empty and not resp.empty:
        tests["_d"] = _dates(tests, "test_date")
        resp["student_id"] = resp["student_id"].astype(str)
        resp["_c"] = _as_bool(resp["is_correct"])
        merged = resp.merge(tests[["test_id", "_d"]], on="test_id", how="inner")
        scores = (merged.groupby(["student_id", "test_id", "_d"], as_index=False)
                  .agg(_s=("_c", "mean")))

    att = data.get("attendance", pd.DataFrame()).copy()
    if not att.empty:
        att["student_id"] = att["student_id"].astype(str)
        att["_d"] = _dates(att, "date")
        att["_p"] = _as_bool(att["present"])
        att = att.dropna(subset=["_d"])
    eng = data.get("engagement", pd.DataFrame()).copy()
    if not eng.empty:
        eng["student_id"] = eng["student_id"].astype(str)
        eng["_d"] = _dates(eng, "date")
        eng = eng.dropna(subset=["_d"])
    fees = data.get("fees", pd.DataFrame()).copy()
    if not fees.empty:
        fees["student_id"] = fees["student_id"].astype(str)
        fees["_due"] = _dates(fees, "due_date")
        fees["_paid"] = _dates(fees, "paid_date")
        fees = fees.dropna(subset=["_due"])

    candidates = [students["_join"].max()]
    for frame, col in ((att, "_d"), (scores, "_d"), (eng, "_d")):
        if len(frame):
            candidates.append(frame[col].max())
    data_end = max(c for c in candidates if pd.notna(c))
    return students, scores, att, eng, fees, data_end


def build_snapshots(
    data: dict[str, pd.DataFrame], config: AttritionConfig
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Return (snapshot frame, data_end). Column `label` is 0/1, or NaN if censored."""
    students, scores, att, eng, fees, data_end = _prepare(data)
    use_fees = config.use_fees and not fees.empty
    horizon = pd.Timedelta(days=config.horizon_days)
    rows: list[dict[str, Any]] = []

    def by_student(frame):
        return {k: v for k, v in frame.groupby("student_id")} if len(frame) else {}

    att_g, sc_g, eng_g, fee_g = by_student(att), by_student(scores), by_student(eng), by_student(fees)
    empty_att = pd.DataFrame(columns=["_d", "_p"])
    empty_sc = pd.DataFrame(columns=["_d", "_s"])
    empty_eng = pd.DataFrame(columns=["_d", "homework_submitted", "app_logins", "doubts_asked"])
    empty_fee = pd.DataFrame(columns=["_due", "_paid", "amount"])

    for _, st_row in students.iterrows():
        sid = st_row["student_id"]
        join = st_row["_join"]
        if pd.isna(join):
            continue
        drop = st_row["_drop"]
        if pd.notna(drop) and drop > data_end:
            drop = pd.NaT  # future-dated withdrawal is not an observed outcome
        hist = _StudentHistory(
            att_g.get(sid, empty_att), sc_g.get(sid, empty_sc),
            eng_g.get(sid, empty_eng), fee_g.get(sid, empty_fee),
        )
        last = (drop - pd.Timedelta(days=1)) if pd.notna(drop) else data_end
        snap = join + pd.Timedelta(days=config.min_tenure_days)
        while snap <= last:
            if pd.notna(drop):
                label = 1.0 if drop <= snap + horizon else 0.0
            else:
                label = 0.0 if snap + horizon <= data_end else np.nan
            feats = _features_at(hist, np.datetime64(snap.date()), np.datetime64(join.date()), use_fees)
            feats.update(student_id=sid, snapshot=snap, label=label)
            rows.append(feats)
            snap += pd.Timedelta(days=config.step_days)
    frame = pd.DataFrame(rows)
    return frame, data_end


# ----------------------------------------------------------------------------
# Modelling
# ----------------------------------------------------------------------------
def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _make_lr(c: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=2000)),
    ])


def _safe_auc(kind: str, y, p) -> float | None:
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, p) if kind == "roc" else average_precision_score(y, p))


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(y).astype(int)
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean()) if len(y) else None,
        "pr_auc": _safe_auc("pr", y, p), "roc_auc": _safe_auc("roc", y, p),
        "brier": float(brier_score_loss(y, p)) if len(y) else None,
        "threshold": float(threshold),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "alert_rate": float(pred.mean()) if len(y) else None,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def calibration_table(y, p, bins: int = 5) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y).astype(int), "p": p})
    if frame.empty or frame["p"].nunique() < 2:
        return pd.DataFrame(columns=["bin", "mean_predicted", "observed_rate", "n"])
    frame["bin"] = pd.qcut(frame["p"], q=min(bins, frame["p"].nunique()), duplicates="drop")
    out = (frame.groupby("bin", observed=True)
           .agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), n=("y", "size"))
           .reset_index())
    out["bin"] = np.arange(1, len(out) + 1)
    return out


def choose_threshold(y, p, max_alert_rate: float, beta: float = 2.0) -> float:
    """Best F-beta threshold on validation data, limited by follow-up capacity."""
    y = np.asarray(y).astype(int)
    best_t, best_f = None, -1.0
    for t in np.unique(p):
        pred = p >= t
        if pred.mean() > max_alert_rate:
            continue
        tp = int((pred & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        fn = int((~pred & (y == 1)).sum())
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f = (1 + beta**2) * prec * rec / (beta**2 * prec + rec)
        if f > best_f:
            best_f, best_t = f, float(t)
    if best_t is None:  # no useful threshold: fall back to the capacity quantile
        best_t = float(np.quantile(p, 1 - max_alert_rate))
    return best_t


def chronological_split(snaps: pd.DataFrame, config: AttritionConfig):
    """Train < validation < test by snapshot date, with a purge gap of H days."""
    dates = snaps["snapshot"]
    lo, hi = dates.min(), dates.max()
    span = hi - lo
    t1 = lo + span * config.train_frac
    t2 = lo + span * (config.train_frac + config.val_frac)
    gap = pd.Timedelta(days=config.horizon_days)
    train = snaps[dates <= t1 - gap]
    val = snaps[(dates > t1) & (dates <= t2 - gap)]
    test = snaps[dates > t2]
    return train, val, test, (t1, t2)


def _model_version(config: AttritionConfig, data_end, n_rows: int, feats: list[str]) -> str:
    key = f"{config.horizon_days}|{config.step_days}|{data_end}|{n_rows}|{','.join(feats)}"
    return f"lr-h{config.horizon_days}-{pd.Timestamp(data_end):%Y%m%d}-{hashlib.sha1(key.encode()).hexdigest()[:6]}"


def run_attrition(
    data: dict[str, pd.DataFrame], config: AttritionConfig | None = None
) -> AttritionResult:
    config = config or AttritionConfig()
    result = AttritionResult(status="insufficient", message="", horizon_days=config.horizon_days)
    if data["students"].empty or "dropout_date" not in data["students"]:
        result.message = "No student roster with a dropout_date column is available."
        return result

    snaps, data_end = build_snapshots(data, config)
    result.as_of = data_end.date()
    if snaps.empty:
        result.message = "No prediction snapshots could be built from the available dates."
        return result

    feats = FEATURES_BASE + (FEATURES_FEES if config.use_fees and snaps["fee_overdue_days"].notna().any() else [])
    result.features = feats
    labelled = snaps[snaps["label"].notna()].copy()
    n_students = data["students"]["student_id"].nunique()
    n_drop = int(pd.to_datetime(data["students"]["dropout_date"], errors="coerce").notna().sum())
    result.snapshot_summary = {
        "students": int(n_students), "recorded_departures": n_drop,
        "snapshots": int(len(snaps)), "labelled_snapshots": int(len(labelled)),
        "censored_snapshots": int(snaps["label"].isna().sum()),
        "positive_snapshots": int((labelled["label"] == 1).sum()),
        "data_end": str(data_end.date()),
    }
    result.warnings.append(
        "Students with no recorded dropout_date are assumed to have stayed once the "
        f"{config.horizon_days}-day window is fully observed. Check that withdrawals are recorded reliably."
    )

    # Latest-state features for scoring (always computed, even without a model)
    latest = snaps.sort_values("snapshot").groupby("student_id").tail(1).copy()
    active_ids = set(
        data["students"].loc[pd.to_datetime(data["students"]["dropout_date"], errors="coerce").isna(), "student_id"].astype(str)
    )

    # Rule-based watchlist: always available, even when no model can be trained.
    wl = latest[latest["student_id"].isin(active_ids)].copy()
    if not wl.empty:
        names = data["students"].assign(student_id=data["students"]["student_id"].astype(str)).set_index("student_id")
        wl["name"] = wl["student_id"].map(names["name"])
        wl["batch_id"] = wl["student_id"].map(names["batch_id"])
        wl["rule_score"] = wl.apply(rule_score, axis=1)
        wl["band"] = np.select([wl["rule_score"] >= 55, wl["rule_score"] >= 30],
                               ["High support priority", "Needs a check-in"], "Steady")
        result.watchlist = (wl[["student_id", "name", "batch_id", "rule_score", "band", "att_30",
                                "score_overall", "hw_30", "consec_absent", "days_since_activity"]]
                            .sort_values("rule_score", ascending=False).reset_index(drop=True))

    if len(labelled) == 0 or labelled["label"].sum() == 0:
        result.message = ("No recorded departures inside the observed period, so a departure model "
                          "cannot be trained. Record withdrawal dates in the Students file.")
        return result

    train, val, test, (t1, t2) = chronological_split(labelled, config)
    result.snapshot_summary.update(
        train_rows=len(train), val_rows=len(val), test_rows=len(test),
        train_positives=int(train["label"].sum()), val_positives=int(val["label"].sum()),
        test_positives=int(test["label"].sum()),
        train_end=str(t1.date()), val_end=str(t2.date()),
    )
    if len(train) < 50 or train["label"].nunique() < 2:
        result.message = ("Not enough labelled history to train: need at least 50 training snapshots "
                          "with both outcomes. Collect more historical withdrawals.")
        return result
    if len(val) == 0 or len(test) == 0 or val["label"].nunique() < 2:
        result.message = ("Not enough later history for validation and testing. "
                          "The model was not trained.")
        return result

    Xtr, ytr = train[feats], train["label"].astype(int)
    Xva, yva = val[feats], val["label"].astype(int).values
    Xte, yte = test[feats], test["label"].astype(int).values

    # Model selection on validation (PR-AUC)
    best_c, best_score = config.c_grid[0], -1.0
    for c in config.c_grid:
        model = _make_lr(c).fit(Xtr, ytr)
        score = _safe_auc("pr", yva, model.predict_proba(Xva)[:, 1]) or 0.0
        if score > best_score:
            best_c, best_score = c, score
    lr = _make_lr(best_c).fit(Xtr, ytr)

    # Platt calibration on validation only
    raw_va = lr.predict_proba(Xva)[:, 1]
    calibrator = None
    if yva.sum() >= MIN_VAL_POSITIVES and (len(yva) - yva.sum()) >= MIN_VAL_POSITIVES:
        calibrator = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_va)[:, None], yva)

    def predict(frame: pd.DataFrame) -> np.ndarray:
        raw = lr.predict_proba(frame[feats])[:, 1]
        return calibrator.predict_proba(_logit(raw)[:, None])[:, 1] if calibrator else raw

    p_va, p_te = predict(val), predict(test)
    threshold = choose_threshold(yva, p_va, config.max_alert_rate)
    result.threshold = threshold
    result.metrics = evaluate(yte, p_te, threshold)
    result.metrics["calibrated"] = calibrator is not None
    result.metrics["brier_uncalibrated"] = float(brier_score_loss(yte, lr.predict_proba(Xte)[:, 1]))
    result.metrics["best_c"] = best_c
    result.calibration_bins = calibration_table(yte, p_te)

    # Rule-based baseline on the same test rows
    rule_te = test.apply(rule_score, axis=1).values / 100.0
    rule_va = val.apply(rule_score, axis=1).values / 100.0
    rule_thr = 0.30
    result.baseline_metrics = evaluate(yte, rule_te, rule_thr)
    result.baseline_metrics["note"] = "Existing points score / 100; threshold 0.30 = 'Needs a check-in'."
    result.baseline_metrics["val_pr_auc"] = _safe_auc("pr", yva, rule_va)

    # Gradient boosting comparison only
    try:
        gb = HistGradientBoostingClassifier(max_depth=3, max_iter=150, learning_rate=0.05,
                                            l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
        p_gb = gb.predict_proba(Xte)[:, 1]
        result.gb_metrics = evaluate(yte, p_gb, threshold)
        result.gb_metrics["val_pr_auc"] = _safe_auc("pr", yva, gb.predict_proba(Xva)[:, 1])
    except Exception:  # comparison is optional
        result.gb_metrics = {}

    # Reliability gate
    reasons = []
    if train["label"].sum() < MIN_TRAIN_POSITIVES:
        reasons.append(f"only {int(train['label'].sum())} departures in training (need {MIN_TRAIN_POSITIVES})")
    if val["label"].sum() < MIN_VAL_POSITIVES:
        reasons.append(f"only {int(val['label'].sum())} departures in validation (need {MIN_VAL_POSITIVES})")
    if test["label"].sum() < MIN_TEST_POSITIVES:
        reasons.append(f"only {int(test['label'].sum())} departures in the test period (need {MIN_TEST_POSITIVES})")
    if calibrator is None:
        reasons.append("validation data too small to calibrate probabilities")
    pr_auc = result.metrics.get("pr_auc")
    if pr_auc is not None and result.metrics["base_rate"] is not None and pr_auc <= result.metrics["base_rate"]:
        reasons.append("test PR-AUC is no better than the base departure rate")
    result.validated = not reasons
    if reasons:
        result.warnings.append("Not validated as a predictor: " + "; ".join(reasons) + ".")

    # Coefficients
    clf = lr.named_steps["clf"]
    result.coefficients = (
        pd.DataFrame({"feature": feats, "label": [FEATURE_LABELS[f] for f in feats],
                      "coefficient": clf.coef_[0]})
        .sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)
    )

    # Score current active students
    current = latest[latest["student_id"].isin(active_ids)].copy()
    if not current.empty:
        raw = lr.predict_proba(current[feats])[:, 1]
        current["probability"] = calibrator.predict_proba(_logit(raw)[:, None])[:, 1] if calibrator else raw
        Z = lr.named_steps["scale"].transform(lr.named_steps["impute"].transform(current[feats]))
        contrib = Z * clf.coef_[0]
        names = data["students"].set_index(data["students"]["student_id"].astype(str))
        top = []
        for row in contrib:
            idx = np.argsort(row)[::-1][:3]
            zrow = Z[len(top)]
            top.append("; ".join(
                f"{FEATURE_LABELS[feats[i]]} ({'lower' if zrow[i] < 0 else 'higher'} than typical)"
                for i in idx if row[i] > 0.05) or "No feature stands out")
        current["top_factors"] = top
        current["name"] = current["student_id"].map(names["name"])
        current["batch_id"] = current["student_id"].map(names["batch_id"])
        current["rule_score"] = current.apply(rule_score, axis=1)
        current["flag"] = np.where(current["probability"] >= threshold, "Review recommended", "No flag")
        current["horizon_days"] = config.horizon_days
        current["feature_cutoff"] = data_end.date()
        result.predictions = current.sort_values("probability", ascending=False).reset_index(drop=True)
        result.feature_matrix = current[["student_id"] + feats].reset_index(drop=True)

    result.model_version = _model_version(config, data_end, len(labelled), feats)
    result.status = "ok"
    result.message = ("Model trained and evaluated on held-out later cohorts." if result.validated
                      else "Model trained, but it is not validated for decision support.")
    return result

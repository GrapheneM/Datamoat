from datetime import date
import pandas as pd
from datamoat.analytics import economic_summary, compute_analytics
from tests.synth import make_data

def _fees():
    return pd.DataFrame([
        dict(student_id="A", instalment_no=1, due_date="2026-01-10", paid_date="2026-01-09", amount=1000),
        dict(student_id="A", instalment_no=2, due_date="2026-02-10", paid_date=None, amount=500),
        dict(student_id="B", instalment_no=1, due_date="2026-01-15", paid_date="2026-03-01", amount=700),  # paid after cutoff
        dict(student_id="B", instalment_no=2, due_date="2026-06-01", paid_date=None, amount=900),         # not yet due
    ])
STU = pd.DataFrame({"student_id": ["A", "B"], "name": ["Ann", "Bob"], "batch_id": ["X", "X"]})

def test_economic_totals_and_units():
    s, overdue = economic_summary(_fees(), STU, date(2026, 2, 28))
    assert s["scheduled"] == 1000 + 500 + 700          # only instalments due by cutoff
    assert s["collected"] == 1000                      # payment after cutoff is not collected yet
    assert s["overdue"] == 500 + 700
    assert s["overdue_count"] == 2
    assert s["collected"] + s["overdue"] == s["scheduled"]
    assert set(overdue["student_id"]) == {"A", "B"}

def test_economics_empty():
    s, o = economic_summary(pd.DataFrame(), STU, date(2026, 1, 1))
    assert s["scheduled"] == 0 and o.empty

def test_analytics_runs_and_keeps_rule_baseline():
    d = make_data(n=60, seed=2)
    a = compute_analytics(d, "2025", date(2026, 2, 1))
    assert not a["risk"].empty
    assert {"support_score", "band"} <= set(a["risk"].columns)
    assert set(a["risk"]["band"]) <= {"High support priority", "Needs a check-in", "Steady", "Insufficient data"}

def test_missing_data_is_not_risk():
    d = make_data(n=10, seed=4)
    for k in ("attendance", "engagement", "test_responses"): d[k] = d[k].iloc[0:0]
    a = compute_analytics(d, "2025", date(2026, 2, 1))
    assert set(a["risk"]["band"]) == {"Insufficient data"}

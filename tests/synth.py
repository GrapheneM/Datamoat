"""Synthetic fixture. Test success here is NOT validation on real data."""
import numpy as np, pandas as pd

def make_data(n=400, seed=0, days=420, with_fees=True):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-01")
    end = start + pd.Timedelta(days=days)
    students, att, eng, fees, resp = [], [], [], [], []
    tests = pd.DataFrame({"test_id": [f"T{i}" for i in range(20)],
        "test_date": [(start + pd.Timedelta(days=20*i + 10)).date() for i in range(20)], "max_marks": 100})
    questions = pd.DataFrame({"question_id": [f"Q{i}" for i in range(5)], "subject": "Math",
        "topic": [f"Topic{i%3}" for i in range(5)], "sub_topic": ""})
    for i in range(n):
        sid = f"S{i:04d}"
        join = start + pd.Timedelta(days=int(rng.integers(0, 150)))
        engaged = rng.random() > 0.35
        # disengaged students tend to drop; outcome driven by attendance
        p_att = 0.9 if engaged else 0.55
        drop = None
        if not engaged and rng.random() < 0.8:
            drop = join + pd.Timedelta(days=int(rng.integers(60, 300)))
            if drop > end: drop = None
        students.append(dict(student_id=sid, name=f"N{i}", batch_id=f"B{i%4}", join_date=join.date(),
                             cohort="2025", dropout_date=drop.date() if drop is not None else None))
        last = drop if drop is not None else end
        for d in pd.date_range(join, last, freq="3D"):
            att.append(dict(student_id=sid, date=d.date(), present=bool(rng.random() < p_att), session_id="s"))
            eng.append(dict(student_id=sid, date=d.date(), doubts_asked=int(rng.poisson(1 if engaged else .2)),
                            homework_submitted=float(rng.random() < p_att), app_logins=int(rng.poisson(3 if engaged else 1))))
        for t in tests.itertuples():
            if join.date() <= t.test_date <= last.date():
                for q in questions.question_id:
                    resp.append(dict(test_id=t.test_id, student_id=sid, question_id=q, is_correct=bool(rng.random() < (.7 if engaged else .45)), time_taken_sec=30))
        if with_fees:
            for k in range(1, 6):
                due = join + pd.Timedelta(days=60*k)
                if due <= last:
                    fees.append(dict(student_id=sid, instalment_no=k, due_date=due.date(), paid_date=(due+pd.Timedelta(days=int(rng.integers(0,20)))).date(), amount=1000.0))
    return dict(students=pd.DataFrame(students), questions=questions, tests=tests,
                test_responses=pd.DataFrame(resp), attendance=pd.DataFrame(att),
                fees=pd.DataFrame(fees), engagement=pd.DataFrame(eng))

import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "test.db")

from fastapi.testclient import TestClient  # noqa: E402

from backend.main import app  # noqa: E402
from db.models import BacktestResult, SessionLocal, TrainingRun, init_db  # noqa: E402

init_db()
client = TestClient(app)


def test_train_endpoint_creates_run():
    r = client.post("/train", json={"timesteps": 10, "n_paths_backtest": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "queued"
    assert "run_id" in body


def test_status_endpoint():
    r = client.post("/train", json={"timesteps": 10})
    run_id = r.json()["run_id"]
    s = client.get(f"/train/{run_id}/status")
    assert s.status_code == 200
    assert s.json()["run_id"] == run_id
    assert s.json()["status"] in ("queued", "training", "backtesting",
                                 "completed", "failed")


def test_status_404():
    assert client.get("/train/99999/status").status_code == 404


def test_backtest_results_schema():
    init_db()
    db = SessionLocal()
    run = TrainingRun(hyperparams={}, status="completed")
    db.add(run)
    db.commit()
    db.add(BacktestResult(
        run_id=run.id, regime="gbm", agent="rl", pnl_mean=0.1,
        pnl_variance=0.5, total_cost=1.0, p_value=0.05,
        pnl_samples=[0.1, 0.2],
    ))
    db.commit()
    run_id = run.id
    db.close()

    r = client.get(f"/backtest/{run_id}")
    assert r.status_code == 200
    body = r.json()
    assert body["regimes"]["gbm"]["rl"]["pnl_mean"] == 0.1
    assert body["regimes"]["gbm"]["rl"]["pnl_samples"] == [0.1, 0.2]


def test_backtest_not_ready():
    init_db()
    db = SessionLocal()
    run = TrainingRun(hyperparams={}, status="training")
    db.add(run)
    db.commit()
    run_id = run.id
    db.close()
    assert client.get(f"/backtest/{run_id}").status_code == 409

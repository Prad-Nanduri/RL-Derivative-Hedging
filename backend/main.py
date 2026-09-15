"""FastAPI backend: launch training runs, poll status, fetch backtest results."""
import os
import threading

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from db.models import BacktestResult, SessionLocal, TrainingRun, init_db

app = FastAPI(title="Deep Hedging RL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATUS = {}  # run_id -> {"status":..., "progress":...}


class TrainRequest(BaseModel):
    regime: str = "gbm"
    days_to_expiry: int = 30
    timesteps: int = 50_000
    n_paths_backtest: int = 200
    seed: int = 7


def _set(run_id, status, progress=None):
    _STATUS[run_id] = {"status": status, "progress": progress if progress is not None else _STATUS.get(run_id, {}).get("progress", 0.0)}
    db = SessionLocal()
    try:
        run = db.get(TrainingRun, run_id)
        if run:
            run.status = status
            if progress is not None:
                run.progress = progress
            db.commit()
    finally:
        db.close()


def _run_pipeline(run_id, req: TrainRequest):
    from evaluation.backtest import run_backtest
    from training.train_ppo import train

    try:
        _set(run_id, "training", 0.1)
        model_path = train(
            {"regime": req.regime, "days_to_expiry": req.days_to_expiry,
             "seed": req.seed},
            total_timesteps=req.timesteps,
            model_dir=os.path.join("models", f"run_{run_id}"),
        )
        db = SessionLocal()
        run = db.get(TrainingRun, run_id)
        run.model_path = model_path
        db.commit()
        db.close()

        _set(run_id, "backtesting", 0.7)
        results = run_backtest(
            model_path, n_paths=req.n_paths_backtest,
            env_overrides={"regime": req.regime, "days_to_expiry": req.days_to_expiry},
        )
        db = SessionLocal()
        for regime, r in results.items():
            for agent in ("rl", "baseline"):
                db.add(BacktestResult(
                    run_id=run_id, regime=regime, agent=agent,
                    pnl_mean=r[agent]["pnl_mean"],
                    pnl_variance=r[agent]["pnl_variance"],
                    total_cost=r[agent]["total_cost"],
                    p_value=r["wilcoxon_pvalue"],
                    pnl_samples=r[agent]["pnl_samples"],
                ))
        db.commit()
        db.close()
        _set(run_id, "completed", 1.0)
    except Exception as exc:  # noqa: BLE001
        db = SessionLocal()
        run = db.get(TrainingRun, run_id)
        if run:
            run.status = "failed"
            run.error = str(exc)
            db.commit()
        db.close()
        _STATUS[run_id] = {"status": "failed", "progress": 0.0}


@app.on_event("startup")
def startup():
    init_db()


@app.post("/train")
def start_training(req: TrainRequest, background: BackgroundTasks):
    db = SessionLocal()
    run = TrainingRun(hyperparams=req.model_dump(), status="queued")
    db.add(run)
    db.commit()
    db.refresh(run)
    run_id = run.id
    db.close()
    _STATUS[run_id] = {"status": "queued", "progress": 0.0}
    background.add_task(_run_pipeline, run_id, req)
    return {"run_id": run_id, "status": "queued"}


@app.get("/train/{run_id}/status")
def train_status(run_id: int):
    if run_id in _STATUS:
        return {"run_id": run_id, **_STATUS[run_id]}
    db = SessionLocal()
    run = db.get(TrainingRun, run_id)
    db.close()
    if not run:
        raise HTTPException(404, "run not found")
    return {"run_id": run_id, "status": run.status, "progress": run.progress}


@app.get("/backtest/{run_id}")
def backtest_results(run_id: int):
    db = SessionLocal()
    rows = db.query(BacktestResult).filter(BacktestResult.run_id == run_id).all()
    run = db.get(TrainingRun, run_id)
    db.close()
    if run is None:
        raise HTTPException(404, "run not found")
    if not rows:
        raise HTTPException(409, f"backtest not ready (status={run.status})")
    out = {"run_id": run_id, "status": run.status, "regimes": {}}
    for row in rows:
        out["regimes"].setdefault(row.regime, {})[row.agent] = {
            "pnl_mean": row.pnl_mean,
            "pnl_variance": row.pnl_variance,
            "total_cost": row.total_cost,
            "p_value": row.p_value,
            "pnl_samples": row.pnl_samples,
        }
    return out

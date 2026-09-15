"""SQLAlchemy models and session setup (SQLite)."""
import os
from datetime import datetime

from sqlalchemy import JSON, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

DB_PATH = os.environ.get("DB_PATH", os.path.join(os.path.dirname(__file__), "runs.db"))
engine = create_engine(f"sqlite:///{DB_PATH}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    hyperparams: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(32), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    model_path: Mapped[str] = mapped_column(String(512), default="")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class BacktestResult(Base):
    __tablename__ = "backtest_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(Integer, index=True)
    regime: Mapped[str] = mapped_column(String(16))
    agent: Mapped[str] = mapped_column(String(16))
    pnl_mean: Mapped[float] = mapped_column(Float)
    pnl_variance: Mapped[float] = mapped_column(Float)
    total_cost: Mapped[float] = mapped_column(Float)
    p_value: Mapped[float] = mapped_column(Float)
    pnl_samples: Mapped[list] = mapped_column(JSON, default=list)


def init_db():
    Base.metadata.create_all(engine)

"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { getStatus, startTraining } from "@/lib/api";

const STATUS_DOT: Record<string, string> = {
  queued: "status-dot status-warn",
  training: "status-dot status-live",
  backtesting: "status-dot status-live",
  completed: "status-dot status-live",
  failed: "status-dot status-bad",
};

export default function Home() {
  const [regime, setRegime] = useState("gbm");
  const [timesteps, setTimesteps] = useState(50000);
  const [nPaths, setNPaths] = useState(200);
  const [runId, setRunId] = useState<number | null>(null);
  const [status, setStatus] = useState<string>("");
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (runId === null) return;
    timer.current = setInterval(async () => {
      try {
        const s = await getStatus(runId);
        setStatus(s.status);
        setProgress(s.progress ?? 0);
        if (s.status === "completed" || s.status === "failed") {
          if (timer.current) clearInterval(timer.current);
        }
      } catch {
        /* keep polling */
      }
    }, 3000);
    return () => {
      if (timer.current) clearInterval(timer.current);
    };
  }, [runId]);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const r = await startTraining({
        regime,
        days_to_expiry: 30,
        timesteps,
        n_paths_backtest: nPaths,
      });
      setRunId(r.run_id);
      setStatus("queued");
      setProgress(0);
    } catch (err) {
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl px-5 py-14 sm:py-20">
      <header className="reveal" style={{ "--i": 0 } as React.CSSProperties}>
        <p className="label-mono mb-4">// option hedging · reinforcement learning</p>
        <h1 className="font-display text-4xl sm:text-5xl font-semibold leading-tight tracking-tight">
          Deep Hedging <span className="foil">RL</span>
        </h1>
        <p className="mt-3 text-sm" style={{ color: "var(--ink-2)" }}>
          PPO agent vs Black-Scholes delta hedge · Buehler et al. 2019 · Kolm
          &amp; Ritter 2019
        </p>
      </header>

      <form
        onSubmit={submit}
        className="panel mt-10 reveal"
        style={{ "--i": 1 } as React.CSSProperties}
      >
        <div className="panel-head">run_control — new training session</div>
        <div className="panel-body space-y-6">
          <div>
            <p className="label-mono mb-2">price-path regime</p>
            <div className="flex gap-2">
              {(["gbm", "garch"] as const).map((r) => (
                <button
                  key={r}
                  type="button"
                  onClick={() => setRegime(r)}
                  data-active={regime === r}
                  className="seg"
                >
                  {r}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-5">
            <label className="block">
              <span className="label-mono">timesteps</span>
              <input
                type="number"
                min={1000}
                step={1000}
                value={timesteps}
                onChange={(e) => setTimesteps(Number(e.target.value))}
                className="mt-1.5"
              />
            </label>
            <label className="block">
              <span className="label-mono">backtest paths</span>
              <input
                type="number"
                min={10}
                value={nPaths}
                onChange={(e) => setNPaths(Number(e.target.value))}
                className="mt-1.5"
              />
            </label>
          </div>
          <button
            type="submit"
            disabled={submitting}
            className="btn-accent w-full sm:w-auto px-6 py-2.5 text-sm font-medium"
          >
            {submitting ? "spawning…" : "▸ start training run"}
          </button>
        </div>
      </form>

      {error && (
        <p className="reveal mt-4 font-mono text-xs" style={{ color: "var(--danger)" }}>
          [err] {error}
        </p>
      )}

      {runId !== null && (
        <section
          className="panel mt-6 reveal"
          style={{ "--i": 2 } as React.CSSProperties}
        >
          <div className="panel-head">run_status</div>
          <div className="panel-body">
            <p className="text-sm">
              <span className="font-mono" style={{ color: "var(--accent)" }}>
                run #{runId}
              </span>
            </p>
            <p className="mt-2 font-mono text-xs" style={{ color: "var(--ink-2)" }}>
              <span className={STATUS_DOT[status] ?? "status-dot status-warn"} />
              {status || "queued"}
            </p>
            {(status === "training" ||
              status === "backtesting" ||
              status === "queued") && (
              <div className="mt-4">
                <div className="progress-track">
                  <div
                    className="progress-fill"
                    style={{ width: `${Math.max(4, progress * 100)}%` }}
                  />
                </div>
                <p className="mt-2 font-mono text-[0.66rem]" style={{ color: "var(--muted)" }}>
                  polling every 3s — training runs in a background task on the
                  api server
                </p>
              </div>
            )}
            {status === "completed" && (
              <Link
                href={`/backtest/${runId}`}
                className="btn-accent mt-4 inline-block px-5 py-2 text-sm"
              >
                view backtest results →
              </Link>
            )}
          </div>
        </section>
      )}

      <p className="reveal mt-10 font-mono text-xs" style={{ "--i": 3, color: "var(--muted)" } as React.CSSProperties}>
        already have a run?{" "}
        <Link href="/backtest/1" className="underline" style={{ color: "var(--accent-2)" }}>
          open the backtest dashboard
        </Link>
      </p>
    </main>
  );
}

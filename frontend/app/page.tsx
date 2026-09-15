"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { getStatus, startTraining } from "@/lib/api";

export default function Home() {
  const [regime, setRegime] = useState("gbm");
  const [timesteps, setTimesteps] = useState(50000);
  const [nPaths, setNPaths] = useState(200);
  const [runId, setRunId] = useState<number | null>(null);
  const [status, setStatus] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [submitting, setSubmitting] = useState(false);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (runId === null) return;
    timer.current = setInterval(async () => {
      try {
        const s = await getStatus(runId);
        setStatus(s.status);
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
    } catch (err) {
      setError(err instanceof Error ? err.message : "request failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="mx-auto max-w-3xl p-6">
      <h1 className="text-2xl font-bold mb-1">Deep Hedging RL</h1>
      <p className="text-sm text-gray-600 mb-6">
        PPO agent vs Black-Scholes delta hedge (Buehler et al. 2019; Kolm &amp;
        Ritter 2019)
      </p>

      <form
        onSubmit={submit}
        className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm space-y-4"
      >
        <div>
          <label className="block text-sm font-medium mb-1">
            Price-path regime
          </label>
          <div className="flex gap-2">
            {(["gbm", "garch"] as const).map((r) => (
              <button
                key={r}
                type="button"
                onClick={() => setRegime(r)}
                className={`px-4 py-2 rounded border text-sm uppercase ${
                  regime === r
                    ? "bg-blue-600 text-white border-blue-600"
                    : "bg-white text-gray-700 border-gray-300"
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <label className="block text-sm">
            <span className="font-medium">Timesteps</span>
            <input
              type="number"
              min={1000}
              step={1000}
              value={timesteps}
              onChange={(e) => setTimesteps(Number(e.target.value))}
              className="mt-1 w-full rounded border border-gray-300 px-3 py-2"
            />
          </label>
          <label className="block text-sm">
            <span className="font-medium">Backtest paths</span>
            <input
              type="number"
              min={10}
              value={nPaths}
              onChange={(e) => setNPaths(Number(e.target.value))}
              className="mt-1 w-full rounded border border-gray-300 px-3 py-2"
            />
          </label>
        </div>
        <button
          type="submit"
          disabled={submitting}
          className="w-full sm:w-auto rounded bg-blue-600 px-5 py-2 text-white disabled:opacity-50"
        >
          {submitting ? "Starting…" : "Start training run"}
        </button>
      </form>

      {error && (
        <p className="mt-4 text-sm text-red-600">Error: {error}</p>
      )}

      {runId !== null && (
        <div className="mt-6 rounded-lg border border-gray-200 bg-white p-5 shadow-sm">
          <p className="text-sm">
            Run <span className="font-mono font-semibold">#{runId}</span> —{" "}
            status:{" "}
            <span
              className={`font-semibold ${
                status === "completed"
                  ? "text-green-600"
                  : status === "failed"
                    ? "text-red-600"
                    : "text-amber-600"
              }`}
            >
              {status || "queued"}
            </span>
          </p>
          {(status === "training" || status === "backtesting" || status === "queued") && (
            <p className="mt-2 text-xs text-gray-500">
              Polling every 3s… training runs in a background task on the API
              server.
            </p>
          )}
          {status === "completed" && (
            <Link
              href={`/backtest/${runId}`}
              className="mt-3 inline-block rounded bg-green-600 px-4 py-2 text-white text-sm"
            >
              View backtest results →
            </Link>
          )}
        </div>
      )}

      <p className="mt-8 text-xs text-gray-500">
        Already have a run?{" "}
        <Link href="/backtest/1" className="text-blue-600 underline">
          Open the backtest dashboard
        </Link>
      </p>
    </main>
  );
}

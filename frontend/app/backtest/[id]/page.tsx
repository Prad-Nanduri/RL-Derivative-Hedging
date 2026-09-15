"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BacktestResponse, getBacktest } from "@/lib/api";

const RL_COLOR = "oklch(0.86 0.20 150)";
const BS_COLOR = "oklch(0.76 0.16 230)";

function histogram(
  rl: number[],
  bs: number[],
  bins = 20,
): { bin: string; rl: number; baseline: number }[] {
  const all = [...rl, ...bs];
  if (!all.length) return [];
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  const w = (hi - lo) / bins || 1;
  const counts = Array.from({ length: bins }, (_, i) => ({
    bin: (lo + (i + 0.5) * w).toFixed(1),
    rl: 0,
    baseline: 0,
  }));
  for (const v of rl) counts[Math.min(bins - 1, Math.floor((v - lo) / w))].rl++;
  for (const v of bs)
    counts[Math.min(bins - 1, Math.floor((v - lo) / w))].baseline++;
  return counts;
}

export default function BacktestPage({
  params,
}: {
  params: { id: string };
}) {
  const [data, setData] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getBacktest(Number(params.id))
      .then(setData)
      .catch((e) => setError(e.message));
  }, [params.id]);

  return (
    <main className="mx-auto max-w-5xl px-5 py-14 sm:py-20">
      <header className="reveal" style={{ "--i": 0 } as React.CSSProperties}>
        <Link href="/" className="label-mono underline" style={{ color: "var(--accent-2)" }}>
          ← new training run
        </Link>
        <h1 className="font-display mt-4 text-4xl sm:text-5xl font-semibold tracking-tight">
          Backtest <span className="foil">#{params.id}</span>
        </h1>
      </header>
      {error && (
        <p className="reveal mt-6 font-mono text-xs" style={{ color: "var(--danger)" }}>
          [err] {error}
        </p>
      )}
      {!data && !error && (
        <p className="mt-6 font-mono text-xs" style={{ color: "var(--muted)" }}>
          fetching run from api…
        </p>
      )}
      {data &&
        Object.entries(data.regimes).map(([regime, agents], i) => {
          const rl = agents.rl;
          const bs = agents.baseline;
          if (!rl || !bs) return null;
          const hist = histogram(rl.pnl_samples, bs.pnl_samples);
          return (
            <section
              key={regime}
              className="panel mt-8 reveal"
              style={{ "--i": i + 1 } as React.CSSProperties}
            >
              <div className="panel-head">pnl_distribution — {regime}</div>
              <div className="panel-body">
                <div className="h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={hist}>
                      <CartesianGrid
                        strokeDasharray="2 6"
                        stroke="oklch(0.26 0.014 220)"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="bin"
                        fontSize={11}
                        stroke="oklch(0.45 0.012 220)"
                        tick={{ fill: "oklch(0.58 0.012 220)", fontFamily: "var(--font-mono)" }}
                        tickLine={false}
                      />
                      <YAxis
                        allowDecimals={false}
                        fontSize={11}
                        stroke="oklch(0.45 0.012 220)"
                        tick={{ fill: "oklch(0.58 0.012 220)", fontFamily: "var(--font-mono)" }}
                        tickLine={false}
                        axisLine={false}
                      />
                      <Tooltip
                        cursor={{ fill: "oklch(0.22 0.014 220)" }}
                        contentStyle={{
                          background: "oklch(0.17 0.014 220)",
                          border: "1px solid oklch(0.28 0.014 220)",
                          borderRadius: 2,
                          fontFamily: "var(--font-mono)",
                          fontSize: 12,
                          color: "oklch(0.93 0.01 200)",
                        }}
                      />
                      <Legend
                        wrapperStyle={{
                          fontFamily: "var(--font-mono)",
                          fontSize: 11,
                          textTransform: "uppercase",
                          letterSpacing: "0.1em",
                        }}
                      />
                      <Bar dataKey="rl" name="RL (PPO)" fill={RL_COLOR} />
                      <Bar dataKey="baseline" name="BS delta hedge" fill={BS_COLOR} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-6 overflow-x-auto">
                  <table className="data">
                    <thead>
                      <tr>
                        <th>agent</th>
                        <th>p&l mean</th>
                        <th>p&l variance</th>
                        <th>total cost</th>
                        <th>wilcoxon p</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(
                        [
                          ["RL (PPO)", rl],
                          ["BS delta hedge", bs],
                        ] as const
                      ).map(([name, a]) => (
                        <tr key={name}>
                          <td className="agent-name" style={{ color: "var(--ink)" }}>
                            {name}
                          </td>
                          <td>{a.pnl_mean.toFixed(4)}</td>
                          <td>{a.pnl_variance.toFixed(4)}</td>
                          <td>{a.total_cost.toFixed(2)}</td>
                          <td>{a.p_value.toExponential(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </section>
          );
        })}
    </main>
  );
}

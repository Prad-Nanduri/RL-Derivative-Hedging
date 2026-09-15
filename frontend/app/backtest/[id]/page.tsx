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
    <main className="mx-auto max-w-5xl p-6">
      <Link href="/" className="text-sm text-blue-600 underline">
        ← New training run
      </Link>
      <h1 className="text-2xl font-bold mt-2 mb-4">
        Backtest — run #{params.id}
      </h1>
      {error && <p className="text-red-600 text-sm">{error}</p>}
      {!data && !error && <p className="text-sm text-gray-500">Loading…</p>}
      {data &&
        Object.entries(data.regimes).map(([regime, agents]) => {
          const rl = agents.rl;
          const bs = agents.baseline;
          if (!rl || !bs) return null;
          const hist = histogram(rl.pnl_samples, bs.pnl_samples);
          return (
            <section
              key={regime}
              className="mb-10 rounded-lg border border-gray-200 bg-white p-5 shadow-sm"
            >
              <h2 className="text-lg font-semibold uppercase mb-4">{regime}</h2>
              <div className="h-72 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={hist}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="bin" fontSize={11} />
                    <YAxis allowDecimals={false} fontSize={11} />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="rl" name="RL (PPO)" fill="#2563eb" />
                    <Bar dataKey="baseline" name="BS delta hedge" fill="#f59e0b" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <table className="mt-4 w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b">
                    <th className="py-2">Agent</th>
                    <th>P&L mean</th>
                    <th>P&L variance</th>
                    <th>Total cost</th>
                    <th>Wilcoxon p</th>
                  </tr>
                </thead>
                <tbody>
                  {(
                    [
                      ["RL (PPO)", rl],
                      ["BS delta hedge", bs],
                    ] as const
                  ).map(([name, a]) => (
                    <tr key={name} className="border-b last:border-0">
                      <td className="py-2 font-medium">{name}</td>
                      <td>{a.pnl_mean.toFixed(4)}</td>
                      <td>{a.pnl_variance.toFixed(4)}</td>
                      <td>{a.total_cost.toFixed(2)}</td>
                      <td>{a.p_value.toExponential(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          );
        })}
    </main>
  );
}

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export interface TrainResponse {
  run_id: number;
  status: string;
}

export interface StatusResponse {
  run_id: number;
  status: string;
  progress: number;
}

export interface AgentStats {
  pnl_mean: number;
  pnl_variance: number;
  total_cost: number;
  p_value: number;
  pnl_samples: number[];
}

export interface BacktestResponse {
  run_id: number;
  status: string;
  regimes: Record<string, { rl?: AgentStats; baseline?: AgentStats }>;
}

export async function startTraining(body: {
  regime: string;
  days_to_expiry: number;
  timesteps: number;
  n_paths_backtest: number;
}): Promise<TrainResponse> {
  const r = await fetch(`${API_URL}/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`POST /train failed: ${r.status}`);
  return r.json();
}

export async function getStatus(runId: number): Promise<StatusResponse> {
  const r = await fetch(`${API_URL}/train/${runId}/status`, { cache: "no-store" });
  if (!r.ok) throw new Error(`status failed: ${r.status}`);
  return r.json();
}

export async function getBacktest(runId: number): Promise<BacktestResponse> {
  const r = await fetch(`${API_URL}/backtest/${runId}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`backtest failed: ${r.status}`);
  return r.json();
}

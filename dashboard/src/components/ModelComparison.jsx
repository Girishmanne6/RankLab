import React, { useEffect, useState } from "react";
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
import { api, MODEL_COLORS, MODEL_LABELS } from "../api.js";

const QUALITY_KEYS = ["recall@10", "ndcg@10", "mrr"];

export default function ModelComparison() {
  const [metrics, setMetrics] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="error">Failed to load metrics: {error}</div>;
  if (!metrics) return <div className="loading">Loading metrics…</div>;

  const models = Object.keys(metrics);
  if (models.length === 0)
    return (
      <div className="card">
        <h2>No metrics yet</h2>
        <p className="hint">
          Run <code>python -m api.train</code> (or let docker compose bootstrap)
          to train models and populate this view.
        </p>
      </div>
    );

  const chartData = QUALITY_KEYS.map((key) => {
    const row = { metric: key.toUpperCase() };
    models.forEach((m) => (row[m] = metrics[m][key] ?? 0));
    return row;
  });

  const bestModel = models.reduce((best, m) =>
    (metrics[m]["ndcg@10"] ?? 0) > (metrics[best]["ndcg@10"] ?? 0) ? m : best
  );

  const allKeys = [
    "recall@10",
    "ndcg@10",
    "mrr",
    "p95_latency_ms",
    "diversity",
    "freshness_hours",
    "long_tail_exposure",
    "popularity_bias",
  ];

  return (
    <>
      <div className="card">
        <h2>Ranking quality (test split)</h2>
        <p className="hint">
          Recall@10, NDCG@10 and MRR per model — higher is better. Best model by
          NDCG@10:{" "}
          <span className="badge accent">{MODEL_LABELS[bestModel] || bestModel}</span>
        </p>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#273253" />
            <XAxis dataKey="metric" stroke="#93a0c2" />
            <YAxis stroke="#93a0c2" />
            <Tooltip
              contentStyle={{ background: "#1a2340", border: "1px solid #273253" }}
              formatter={(v) => v.toFixed(4)}
            />
            <Legend />
            {models.map((m) => (
              <Bar
                key={m}
                dataKey={m}
                name={MODEL_LABELS[m] || m}
                fill={MODEL_COLORS[m] || "#888"}
                radius={[4, 4, 0, 0]}
              />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="card">
        <h2>All metrics</h2>
        <p className="hint">Best model row (by NDCG@10) is highlighted.</p>
        <div style={{ overflowX: "auto" }}>
          <table>
            <thead>
              <tr>
                <th>Model</th>
                {allKeys.map((k) => (
                  <th key={k}>{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m} className={m === bestModel ? "best" : ""}>
                  <td>
                    <strong>{MODEL_LABELS[m] || m}</strong>
                    {m === bestModel && (
                      <span className="badge accent" style={{ marginLeft: 8 }}>
                        best
                      </span>
                    )}
                  </td>
                  {allKeys.map((k) => (
                    <td key={k} className="num">
                      {metrics[m][k] !== undefined ? metrics[m][k].toFixed(3) : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

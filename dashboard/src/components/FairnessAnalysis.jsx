import React, { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, MODEL_COLORS, MODEL_LABELS } from "../api.js";

const FAIRNESS_KEYS = ["diversity", "long_tail_exposure", "popularity_bias"];

export default function FairnessAnalysis() {
  const [metrics, setMetrics] = useState(null);
  const [scatterA, setScatterA] = useState(null);
  const [scatterB, setScatterB] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch((e) => setError(e.message));
    api.scatter("lightgcn").then(setScatterA).catch(() => {});
    api.scatter("xsimgcl").then(setScatterB).catch(() => {});
  }, []);

  if (error) return <div className="error">Failed to load: {error}</div>;
  if (!metrics) return <div className="loading">Loading fairness metrics…</div>;

  const models = Object.keys(metrics);

  const chartData = FAIRNESS_KEYS.map((key) => {
    const row = { metric: key.replace(/_/g, " ") };
    models.forEach((m) => (row[m] = metrics[m][key] ?? 0));
    return row;
  });

  const freshnessData = models.map((m) => ({
    model: MODEL_LABELS[m] || m,
    hours: metrics[m]["freshness_hours"] ?? 0,
    fill: MODEL_COLORS[m] || "#888",
  }));

  const biasReduction =
    metrics.lightgcn && metrics.xsimgcl
      ? metrics.lightgcn.popularity_bias - metrics.xsimgcl.popularity_bias
      : null;
  const tailGain =
    metrics.lightgcn && metrics.xsimgcl
      ? metrics.xsimgcl.long_tail_exposure - metrics.lightgcn.long_tail_exposure
      : null;

  return (
    <>
      {biasReduction !== null && (
        <div className="stat-row">
          <div className="stat">
            <div className="label">Popularity bias: XSimGCL vs LightGCN</div>
            <div className="value" style={{ color: biasReduction > 0 ? "#10b981" : "#ef4444" }}>
              {biasReduction > 0 ? "−" : "+"}
              {Math.abs(biasReduction).toFixed(3)}
            </div>
          </div>
          <div className="stat">
            <div className="label">Long-tail exposure gain (XSimGCL)</div>
            <div className="value" style={{ color: tailGain > 0 ? "#10b981" : "#ef4444" }}>
              {tailGain > 0 ? "+" : ""}
              {(tailGain * 100).toFixed(1)}%
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <h2>Fairness metrics per model</h2>
        <p className="hint">
          Diversity and long-tail exposure: higher is better. Popularity bias
          (correlation of popularity vs exposure): lower is better.
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

      <div className="grid-2">
        <div className="card">
          <h2>Freshness (avg article age, hours)</h2>
          <p className="hint">Lower = fresher recommendations.</p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={freshnessData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#273253" />
              <XAxis dataKey="model" stroke="#93a0c2" />
              <YAxis stroke="#93a0c2" />
              <Tooltip
                contentStyle={{ background: "#1a2340", border: "1px solid #273253" }}
                formatter={(v) => `${v.toFixed(1)}h`}
              />
              <Bar dataKey="hours" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card">
          <h2>Popularity bias: LightGCN vs XSimGCL</h2>
          <p className="hint">
            Each point is an article: x = training popularity, y = how often it
            gets recommended. A tight upward line = strong bias; XSimGCL's
            contrastive regularisation spreads exposure across the tail.
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#273253" />
              <XAxis
                type="number"
                dataKey="popularity"
                name="popularity"
                stroke="#93a0c2"
                label={{ value: "popularity", position: "bottom", fill: "#93a0c2", fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="rec_frequency"
                name="rec freq"
                stroke="#93a0c2"
              />
              <Tooltip
                contentStyle={{ background: "#1a2340", border: "1px solid #273253" }}
                cursor={{ strokeDasharray: "3 3" }}
              />
              <Legend />
              {scatterA && (
                <Scatter
                  name="LightGCN"
                  data={scatterA.points}
                  fill={MODEL_COLORS.lightgcn}
                  fillOpacity={0.55}
                />
              )}
              {scatterB && (
                <Scatter
                  name="XSimGCL"
                  data={scatterB.points}
                  fill={MODEL_COLORS.xsimgcl}
                  fillOpacity={0.55}
                />
              )}
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}

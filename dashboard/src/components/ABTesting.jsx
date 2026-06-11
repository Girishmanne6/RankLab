import React, { useCallback, useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api, MODEL_LABELS } from "../api.js";

export default function ABTesting() {
  const [results, setResults] = useState(null);
  const [noTest, setNoTest] = useState(false);
  const [error, setError] = useState(null);
  const [starting, setStarting] = useState(false);
  const [variantA, setVariantA] = useState("lightgcn");
  const [variantB, setVariantB] = useState("xsimgcl");

  const refresh = useCallback(() => {
    api
      .abResults()
      .then((r) => {
        setResults(r);
        setNoTest(false);
      })
      .catch((e) => {
        if (e.message.includes("No active")) setNoTest(true);
        else setError(e.message);
      });
  }, []);

  useEffect(() => {
    refresh();
    const interval = setInterval(refresh, 10000);
    return () => clearInterval(interval);
  }, [refresh]);

  const startTest = async () => {
    setStarting(true);
    setError(null);
    try {
      await api.abStart({
        name: `${variantA} vs ${variantB}`,
        variant_a: variantA,
        variant_b: variantB,
        simulate_users: 400,
      });
      refresh();
    } catch (e) {
      setError(e.message);
    } finally {
      setStarting(false);
    }
  };

  const modelOptions = Object.keys(MODEL_LABELS);

  return (
    <>
      <div className="card">
        <h2>Start a new A/B test</h2>
        <p className="hint">
          Held-out test users are hashed 50/50 into the two arms; each user is
          served by their arm's model and NDCG@10 is logged as an observation.
        </p>
        <div className="controls">
          <select value={variantA} onChange={(e) => setVariantA(e.target.value)}>
            {modelOptions.map((m) => (
              <option key={m} value={m}>
                A: {MODEL_LABELS[m]}
              </option>
            ))}
          </select>
          <select value={variantB} onChange={(e) => setVariantB(e.target.value)}>
            {modelOptions.map((m) => (
              <option key={m} value={m}>
                B: {MODEL_LABELS[m]}
              </option>
            ))}
          </select>
          <button className="primary" onClick={startTest} disabled={starting}>
            {starting ? "Starting…" : "Start test"}
          </button>
        </div>
        {error && <div className="error">{error}</div>}
      </div>

      {noTest && (
        <div className="card">
          <h2>No active test</h2>
          <p className="hint">Start one above to see live results here.</p>
        </div>
      )}

      {results && <TestResults results={results} />}
    </>
  );
}

function TestResults({ results }) {
  const { variant_a: a, variant_b: b } = results;
  const chartData = [
    { arm: `A · ${MODEL_LABELS[a.model] || a.model}`, mean: a.mean, p95: a.p95_latency_ms },
    { arm: `B · ${MODEL_LABELS[b.model] || b.model}`, mean: b.mean, p95: b.p95_latency_ms },
  ];

  return (
    <>
      <div className="stat-row">
        <div className="stat">
          <div className="label">Test</div>
          <div className="value" style={{ fontSize: 16 }}>{results.name}</div>
        </div>
        <div className="stat">
          <div className="label">Status</div>
          <div className="value" style={{ fontSize: 16 }}>
            <span className={`badge ${results.status === "running" ? "good" : "neutral"}`}>
              {results.status}
            </span>
          </div>
        </div>
        <div className="stat">
          <div className="label">Observations (A / B)</div>
          <div className="value" style={{ fontSize: 16 }}>
            {a.n} / {b.n}
          </div>
        </div>
        <div className="stat">
          <div className="label">p-value (Welch's t-test)</div>
          <div className="value" style={{ fontSize: 16 }}>
            {results.p_value.toExponential(2)}{" "}
            <span className={`badge ${results.significant ? "good" : "neutral"}`}>
              {results.significant ? "significant" : "not significant"}
            </span>
          </div>
        </div>
        {results.winner && (
          <div className="stat">
            <div className="label">Winner</div>
            <div className="value" style={{ fontSize: 16, color: "#10b981" }}>
              {MODEL_LABELS[results.winner] || results.winner}
            </div>
          </div>
        )}
      </div>

      <div className="grid-2">
        <div className="card">
          <h2>Mean {results.primary_metric} per arm</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#273253" />
              <XAxis dataKey="arm" stroke="#93a0c2" />
              <YAxis stroke="#93a0c2" />
              <Tooltip
                contentStyle={{ background: "#1a2340", border: "1px solid #273253" }}
                formatter={(v) => v.toFixed(4)}
              />
              <Bar dataKey="mean" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <h2>p95 latency per arm (ms)</h2>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#273253" />
              <XAxis dataKey="arm" stroke="#93a0c2" />
              <YAxis stroke="#93a0c2" />
              <Tooltip
                contentStyle={{ background: "#1a2340", border: "1px solid #273253" }}
                formatter={(v) => `${v.toFixed(2)} ms`}
              />
              <Bar dataKey="p95" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}

import React, { useEffect, useState } from "react";
import { api, MODEL_LABELS } from "../api.js";

export default function Recommendations() {
  const [userId, setUserId] = useState("");
  const [model, setModel] = useState("xsimgcl");
  const [sampleUsers, setSampleUsers] = useState([]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.sampleUsers(12).then(setSampleUsers).catch(() => {});
  }, []);

  const fetchRecs = async (uid = userId) => {
    if (uid === "" || uid === null) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.recommend(Number(uid), model, 10);
      setResult(res);
    } catch (e) {
      setError(e.message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const formatAge = (ts) => {
    if (!ts) return "";
    const date = new Date(ts * 1000);
    return date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  return (
    <>
      <div className="card">
        <h2>Get recommendations</h2>
        <p className="hint">
          Enter a user index and pick a model. Each recommendation includes an
          explanation of why it was ranked.
        </p>
        <div className="controls">
          <input
            type="number"
            placeholder="User ID (e.g. 42)"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && fetchRecs()}
            style={{ width: 160 }}
          />
          <select value={model} onChange={(e) => setModel(e.target.value)}>
            {Object.keys(MODEL_LABELS).map((m) => (
              <option key={m} value={m}>
                {MODEL_LABELS[m]}
              </option>
            ))}
          </select>
          <button className="primary" onClick={() => fetchRecs()} disabled={loading}>
            {loading ? "Loading…" : "Recommend"}
          </button>
        </div>
        {sampleUsers.length > 0 && (
          <p className="hint">
            Try a user:{" "}
            {sampleUsers.map((u) => (
              <button
                key={u}
                className="tab"
                style={{ padding: "2px 10px", marginRight: 6, fontSize: 12 }}
                onClick={() => {
                  setUserId(String(u));
                  fetchRecs(u);
                }}
              >
                {u}
              </button>
            ))}
          </p>
        )}
        {error && <div className="error">{error}</div>}
      </div>

      {result && (
        <div className="card">
          <h2>
            Top {result.recommendations.length} for user {result.user_id}{" "}
            <span className="badge accent">{MODEL_LABELS[result.model_name]}</span>{" "}
            {result.cold_start && <span className="badge bad">cold start</span>}{" "}
            <span className="badge neutral">{result.latency_ms.toFixed(1)} ms</span>
          </h2>
          <div>
            {result.recommendations.map((rec) => (
              <div className="rec-item" key={rec.item_idx}>
                <div className="rec-rank">{rec.rank}</div>
                <div style={{ flex: 1 }}>
                  <div className="rec-title">{rec.title}</div>
                  <div className="rec-meta">
                    <span className="badge neutral">{rec.category}</span>
                    {"  "}published {formatAge(rec.published_ts)} · score{" "}
                    {rec.score.toFixed(4)} · item #{rec.item_idx}
                  </div>
                  <div className="rec-why">{rec.explanation}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

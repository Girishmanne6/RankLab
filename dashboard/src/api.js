const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

async function request(path, options = {}) {
  const resp = await fetch(`${API_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}));
    throw new Error(body.detail || `${resp.status} ${resp.statusText}`);
  }
  return resp.json();
}

export const api = {
  health: () => request("/health"),
  models: () => request("/models"),
  metrics: () => request("/metrics"),
  scatter: (model) => request(`/scatter/${model}`),
  sampleUsers: (n = 20) => request(`/users/sample?n=${n}`),
  recommend: (userId, modelName, k = 10) =>
    request("/recommend", {
      method: "POST",
      body: JSON.stringify({ user_id: userId, model_name: modelName, k }),
    }),
  abResults: () => request("/ab-test/results"),
  abStart: (payload) =>
    request("/ab-test/start", { method: "POST", body: JSON.stringify(payload) }),
};

export const MODEL_COLORS = {
  popularity: "#f59e0b",
  recency: "#10b981",
  item_cf: "#3b82f6",
  lightgcn: "#8b5cf6",
  xsimgcl: "#ec4899",
};

export const MODEL_LABELS = {
  popularity: "Popularity",
  recency: "Recency",
  item_cf: "Item-CF",
  lightgcn: "LightGCN",
  xsimgcl: "XSimGCL",
};

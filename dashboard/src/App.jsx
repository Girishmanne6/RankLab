import React, { useState } from "react";
import ModelComparison from "./components/ModelComparison.jsx";
import FairnessAnalysis from "./components/FairnessAnalysis.jsx";
import ABTesting from "./components/ABTesting.jsx";
import Recommendations from "./components/Recommendations.jsx";

const TABS = [
  { id: "models", label: "Model Comparison", component: ModelComparison },
  { id: "fairness", label: "Fairness Analysis", component: FairnessAnalysis },
  { id: "abtest", label: "A/B Testing", component: ABTesting },
  { id: "recs", label: "Recommendations", component: Recommendations },
];

export default function App() {
  const [active, setActive] = useState("models");
  const ActiveComponent = TABS.find((t) => t.id === active).component;

  return (
    <div className="app">
      <div className="header">
        <h1>
          <span className="logo">Rank</span>Lab
        </h1>
      </div>
      <p className="subtitle">
        Responsible news recommendation on EB-NeRD — accuracy, fairness, and
        popularity-bias analysis across five models.
      </p>

      <div className="tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            className={`tab ${active === tab.id ? "active" : ""}`}
            onClick={() => setActive(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      <ActiveComponent />
    </div>
  );
}

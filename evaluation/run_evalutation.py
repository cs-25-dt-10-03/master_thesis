# Entry point for evaluation
# Works with both FlexOffers and DFOs

from evaluation.scenarios.default import generate_test_scenario
from evaluation.evaluators import run_all_evaluators
from evaluation.metrics.core import summarize_metrics
from evaluation.utils.plots import plot_results
import json
import os

RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_evaluation():
    scenarios = generate_test_scenario()
    all_results = []

    for obj in scenarios:
        results = run_all_evaluators(obj)
        all_results.append(results)

    summary = summarize_metrics(all_results)

    with open(f"{RESULTS_DIR}/latest_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(all_results)
    print("Evaluation complete. Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_evaluation()

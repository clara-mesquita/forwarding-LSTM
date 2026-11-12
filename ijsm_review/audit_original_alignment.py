"""Check the archived evaluator without importing or executing its training cells."""
import json
import re
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def audit():
    notebook = json.loads((ROOT / "results_sbrc_26/results_sbrc_26.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
    functions = {match.group(1): match.group(0) for match in
                 re.finditer(r"(?ms)^def (\w+)\(.*?(?=^def |\Z)", source)}
    evaluator = functions["infer_recursive_proposal"]
    rollout = functions["walk_forward_validation_hybrid_loaded"]
    findings = {}
    markers = {
        "rollout_starts_at_first_test_target": "state = train_scaled[-look_back:]",
        "rollout_returns_original_units": "predictions.append(float(pred_orig))",
        "evaluator_shifts_targets_three_steps": "y_true_scaled = test_scaled[look_back : look_back + min_len]",
        "evaluator_inverse_transforms_predictions_again": "y_pred = scaler.inverse_transform",
        "legacy_agreement_is_pearson": "def correncapa_pearson",
    }
    for name, marker in markers.items():
        scope = evaluator if name.startswith("evaluator_") else rollout if name.startswith("rollout_") else source
        findings[name] = marker in scope
    # Prediction i is generated before test[i] is ingested. Its target is test[i].
    pairs = [{"prediction_index": i, "correct_target_index": i,
              "archived_target_index": i + 3, "offset_hours": 18}
             for i in range(4)]
    assert all(pair["archived_target_index"] - pair["correct_target_index"] == 3 for pair in pairs)
    # An inverse affine scaling applied twice changes already-original values.
    lower, scale, scaled_prediction = 500.0, 400.0, 0.5
    original = lower + scale * scaled_prediction
    double_inverse = lower + scale * original
    assert original == 700.0 and double_inverse == 280500.0
    links = ("ac-to", "es-ce", "es-pr", "go-sp", "ms-to", "sc-to")
    archived = {}
    for model in ("proposal", "sarima", "gru", "lstm"):
        path = ROOT / "results_sbrc_26/metrics" / f"metrics_{model}.csv"
        archived[model] = {r["link"]: r for r in csv.DictReader(path.open())
                           if r["protocol"] == "bbr" and r["link"] in links}
        assert len(archived[model]) == 6
    ranks = {link: {metric: sorted(archived, key=lambda model: float(archived[model][link][metric]))
                    for metric in ("rmse", "mae", "smape")} for link in links}
    assert all(order[0] == "proposal" for metrics in ranks.values() for order in metrics.values())
    assert findings["evaluator_shifts_targets_three_steps"]
    assert not findings["evaluator_inverse_transforms_predictions_again"]
    return {"archived_rankings": ranks, "source_checks": findings, "first_four_target_pairs": pairs,
            "scaling_example_mbps": {"correct": original, "double_inverse": double_inverse},
            "status": "Archived scores require aligned replay before extension."}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))

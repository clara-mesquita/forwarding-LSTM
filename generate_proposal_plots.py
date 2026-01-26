"""
Generate plots highlighting the advantages of the Proposal model.

This script reads the aggregated results from 'final_comparison_results.csv'
and outputs SVG files that do not rely on external plotting libraries:
1) proposal_advantage_win_counts.svg
   - Number of links where each model is the best per metric.
2) proposal_advantage_rank_heatmap.svg
   - Average rank per model per metric (lower is better).
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DATA_FILE = Path("final_comparison_results.csv")
LOWER_IS_BETTER = ["RMSE", "MAE", "NRMSE", "Execution_Time"]
HIGHER_IS_BETTER = ["Fuzzy_Accuracy"]
MODEL_ORDER = ["Proposal", "LSTM", "GRU", "SARIMA"]
METRIC_ORDER = LOWER_IS_BETTER + HIGHER_IS_BETTER
MODEL_COLORS = {
    "Proposal": "#2a9d8f",
    "LSTM": "#264653",
    "GRU": "#f4a261",
    "SARIMA": "#e76f51",
}


def load_rows() -> List[Dict[str, str]]:
    if not DATA_FILE.exists():
        raise SystemExit(f"Error: The data file '{DATA_FILE}' was not found.")
    with DATA_FILE.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compute_win_counts(rows: Iterable[Dict[str, str]]) -> Dict[Tuple[str, str], int]:
    rows_by_link: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_link[row["Link"]].append(row)

    win_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    for metric in METRIC_ORDER:
        for entries in rows_by_link.values():
            if metric in LOWER_IS_BETTER:
                winner = min(entries, key=lambda r: float(r[metric]))
            else:
                winner = max(entries, key=lambda r: float(r[metric]))
            win_counts[(metric, winner["Model"])] += 1
    return win_counts


def compute_average_ranks(
    rows: Iterable[Dict[str, str]],
) -> Dict[Tuple[str, str], float]:
    rows_by_link: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_link[row["Link"]].append(row)

    rank_sums: Dict[Tuple[str, str], float] = defaultdict(float)
    rank_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    for metric in METRIC_ORDER:
        for entries in rows_by_link.values():
            sorted_entries = sorted(
                entries,
                key=lambda r: float(r[metric]),
                reverse=metric in HIGHER_IS_BETTER,
            )
            for rank, entry in enumerate(sorted_entries, start=1):
                key = (metric, entry["Model"])
                rank_sums[key] += rank
                rank_counts[key] += 1

    return {key: rank_sums[key] / rank_counts[key] for key in rank_sums}


def svg_header(width: int, height: int) -> List[str]:
    return [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>",
        "<style>",
        "text { font-family: Arial, sans-serif; font-size: 12px; }",
        "</style>",
    ]


def svg_footer() -> List[str]:
    return ["</svg>"]


def generate_win_count_svg(win_counts: Dict[Tuple[str, str], int]) -> None:
    width, height = 900, 520
    margin = {"left": 80, "right": 40, "top": 60, "bottom": 80}
    chart_width = width - margin["left"] - margin["right"]
    chart_height = height - margin["top"] - margin["bottom"]
    max_wins = max(win_counts.values(), default=1)

    output: List[str] = svg_header(width, height)
    output.append(
        f"<text x='{width / 2}' y='30' text-anchor='middle' font-size='16'>"
        "Links Where Each Model Is Best (Per Metric)</text>"
    )

    bar_group_width = chart_width / len(METRIC_ORDER)
    bar_width = bar_group_width / (len(MODEL_ORDER) + 1)

    for i, metric in enumerate(METRIC_ORDER):
        group_x = margin["left"] + i * bar_group_width
        output.append(
            f"<text x='{group_x + bar_group_width / 2}' y='{height - 40}' "
            "text-anchor='middle'>"
            f"{metric}</text>"
        )
        for j, model in enumerate(MODEL_ORDER):
            wins = win_counts.get((metric, model), 0)
            bar_height = (wins / max_wins) * chart_height
            x = group_x + j * bar_width + bar_width * 0.5
            y = margin["top"] + chart_height - bar_height
            color = MODEL_COLORS.get(model, "#999999")
            output.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{bar_width * 0.8:.2f}' "
                f"height='{bar_height:.2f}' fill='{color}'/>"
            )
            output.append(
                f"<text x='{x + bar_width * 0.4:.2f}' y='{y - 4:.2f}' "
                "text-anchor='middle'>"
                f"{wins}</text>"
            )

    # Axes
    output.append(
        f"<line x1='{margin['left']}' y1='{margin['top']}' "
        f"x2='{margin['left']}' y2='{margin['top'] + chart_height}' "
        "stroke='black'/>"
    )
    output.append(
        f"<line x1='{margin['left']}' y1='{margin['top'] + chart_height}' "
        f"x2='{margin['left'] + chart_width}' y2='{margin['top'] + chart_height}' "
        "stroke='black'/>"
    )

    output.append(
        f"<text x='20' y='{margin['top'] + chart_height / 2}' "
        "transform='rotate(-90 20,"
        f"{margin['top'] + chart_height / 2})' text-anchor='middle'>"
        "Number of Links</text>"
    )

    legend_x = width - margin["right"] - 160
    legend_y = margin["top"]
    output.append(
        f"<text x='{legend_x}' y='{legend_y - 10}' font-weight='bold'>Model</text>"
    )
    for idx, model in enumerate(MODEL_ORDER):
        y = legend_y + idx * 18
        color = MODEL_COLORS.get(model, "#999999")
        output.append(
            f"<rect x='{legend_x}' y='{y}' width='12' height='12' fill='{color}'/>"
        )
        output.append(f"<text x='{legend_x + 18}' y='{y + 10}'>{model}</text>")

    output.extend(svg_footer())
    Path("proposal_advantage_win_counts.svg").write_text(
        "\n".join(output), encoding="utf-8"
    )
    print("Generated proposal_advantage_win_counts.svg")


def generate_rank_heatmap_svg(avg_ranks: Dict[Tuple[str, str], float]) -> None:
    width, height = 700, 420
    margin = {"left": 120, "right": 40, "top": 60, "bottom": 60}
    cell_width = (width - margin["left"] - margin["right"]) / len(MODEL_ORDER)
    cell_height = (height - margin["top"] - margin["bottom"]) / len(METRIC_ORDER)

    all_values = list(avg_ranks.values())
    min_val = min(all_values)
    max_val = max(all_values)
    span = max_val - min_val or 1

    def color_for(value: float) -> str:
        # Map lower ranks to darker green, higher ranks to lighter.
        normalized = (value - min_val) / span
        shade = int(255 - normalized * 160)
        return f"rgb(42,{shade},143)"

    output: List[str] = svg_header(width, height)
    output.append(
        f"<text x='{width / 2}' y='30' text-anchor='middle' font-size='16'>"
        "Average Rank per Model and Metric</text>"
    )

    for row_idx, metric in enumerate(METRIC_ORDER):
        y = margin["top"] + row_idx * cell_height
        output.append(
            f"<text x='{margin['left'] - 10}' y='{y + cell_height / 2 + 4}' "
            "text-anchor='end'>"
            f"{metric}</text>"
        )
        for col_idx, model in enumerate(MODEL_ORDER):
            x = margin["left"] + col_idx * cell_width
            value = avg_ranks.get((metric, model), 0.0)
            output.append(
                f"<rect x='{x:.2f}' y='{y:.2f}' width='{cell_width:.2f}' "
                f"height='{cell_height:.2f}' fill='{color_for(value)}'/>"
            )
            output.append(
                f"<text x='{x + cell_width / 2:.2f}' y='{y + cell_height / 2 + 4:.2f}' "
                "text-anchor='middle' fill='white'>"
                f"{value:.2f}</text>"
            )

    for col_idx, model in enumerate(MODEL_ORDER):
        x = margin["left"] + col_idx * cell_width + cell_width / 2
        output.append(
            f"<text x='{x:.2f}' y='{height - 20}' text-anchor='middle'>{model}</text>"
        )

    output.extend(svg_footer())
    Path("proposal_advantage_rank_heatmap.svg").write_text(
        "\n".join(output), encoding="utf-8"
    )
    print("Generated proposal_advantage_rank_heatmap.svg")


def main() -> None:
    rows = load_rows()
    win_counts = compute_win_counts(rows)
    avg_ranks = compute_average_ranks(rows)
    generate_win_count_svg(win_counts)
    generate_rank_heatmap_svg(avg_ranks)


if __name__ == "__main__":
    main()

"""Build comparison tables and HTML reports from evaluation results."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

VARIANT_LABELS = {
    "without_sentiment": "baseline",
    "with_sentiment": "+sentiment",
    "with_sarcasm": "+sarcasm",
    "with_sentiment_and_sarcasm": "+sentiment+sarcasm",
    "baseline": "baseline",
    "sentiment_baseline": "sentiment BEZ detekcije sarkazma",
    "sentiment_with_sarcasm_detection": "sentiment SA detekcijom sarkazma",
}


def friendly_variant(variant: Optional[str]) -> str:
    if not variant:
        return "-"
    return VARIANT_LABELS.get(variant, variant)


def flatten_evaluation_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert a single evaluation_*.json report into flat comparison rows."""
    rows: List[Dict[str, Any]] = []
    base = {
        "run_id": report.get("run_id"),
        "model": report.get("model"),
        "technique": report.get("prompting_technique"),
        "split": report.get("split"),
        "limit": report.get("limit"),
    }

    for task_result in report.get("tasks", []):
        task = task_result.get("task")
        row = {
            **base,
            "task": task,
            "source": task_result.get("source"),
            "variant": friendly_variant(task_result.get("variant")),
        }

        if task in ("sarcasm", "sentiment"):
            metrics = task_result.get("metrics", {})
            row.update(
                {
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "f1": metrics.get("f1"),
                    "rouge1_f": None,
                    "rouge2_f": None,
                    "rougeL_f": None,
                }
            )
        elif task == "summarization":
            rouge = task_result.get("metrics", {}).get("rouge", {})
            row.update(
                {
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                    "f1": None,
                    "rouge1_f": rouge.get("rouge1_f"),
                    "rouge2_f": rouge.get("rouge2_f"),
                    "rougeL_f": rouge.get("rougeL_f"),
                }
            )
        else:
            continue

        rows.append(row)

    return rows


def flatten_comparison_report(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert comparison_*.json (grid run) into flat rows."""
    rows: List[Dict[str, Any]] = []
    dataset = report.get("dataset", {})

    for experiment in report.get("experiments", []):
        task = experiment.get("task")
        row = {
            "run_id": report.get("run_id"),
            "model": experiment.get("model"),
            "technique": experiment.get("technique"),
            "split": dataset.get("split"),
            "limit": dataset.get("limit"),
            "task": task,
            "source": experiment.get("source"),
            "variant": friendly_variant(experiment.get("variant")),
            "accuracy": experiment.get("accuracy"),
            "precision": experiment.get("precision"),
            "recall": experiment.get("recall"),
            "f1": experiment.get("f1"),
            "rouge1_f": experiment.get("rouge1_f"),
            "rouge2_f": experiment.get("rouge2_f"),
            "rougeL_f": experiment.get("rougeL_f"),
        }
        rows.append(row)

    return rows


def load_rows_from_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        report = json.load(handle)

    if "experiments" in report:
        return flatten_comparison_report(report)
    return flatten_evaluation_report(report)


def load_rows_from_directory(directory: Path, pattern: str = "*.json") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(directory.glob(pattern)):
        if path.name.startswith("comparison_"):
            continue
        try:
            rows.extend(load_rows_from_json(path))
        except (json.JSONDecodeError, KeyError):
            continue
    return rows


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    fieldnames = [
        "run_id",
        "model",
        "technique",
        "task",
        "source",
        "variant",
        "split",
        "limit",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "rouge1_f",
        "rouge2_f",
        "rougeL_f",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _format_metric(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def _build_html_table(rows: List[Dict[str, Any]]) -> str:
    headers = [
        "Model",
        "Technique",
        "Task",
        "Source",
        "Variant",
        "Accuracy",
        "F1",
        "ROUGE-1",
        "ROUGE-L",
    ]

    body_rows = []
    for row in rows:
        body_rows.append(
            "<tr>"
            f"<td>{row.get('model', '-')}</td>"
            f"<td>{row.get('technique', '-')}</td>"
            f"<td>{row.get('task', '-')}</td>"
            f"<td>{row.get('source') or '-'}</td>"
            f"<td>{row.get('variant', '-')}</td>"
            f"<td>{_format_metric(row.get('accuracy'))}</td>"
            f"<td>{_format_metric(row.get('f1'))}</td>"
            f"<td>{_format_metric(row.get('rouge1_f'))}</td>"
            f"<td>{_format_metric(row.get('rougeL_f'))}</td>"
            "</tr>"
        )

    header_html = "".join(f"<th>{header}</th>" for header in headers)
    return (
        '<table class="comparison-table">'
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _build_plotly_charts(rows: List[Dict[str, Any]]) -> str:
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        return "<p><em>Plotly nije instaliran — prikazana je samo tabela.</em></p>"

    charts: List[str] = []
    classification_rows = [row for row in rows if row.get("task") in ("sarcasm", "sentiment")]
    summarization_rows = [row for row in rows if row.get("task") == "summarization"]

    if classification_rows:
        fig = px.bar(
            classification_rows,
            x="model",
            y="f1",
            color="technique",
            facet_col="task",
            barmode="group",
            title="F1 po modelu i prompting tehnici (klasifikacija)",
            labels={"f1": "F1", "model": "Model", "technique": "Technique"},
        )
        fig.update_layout(height=450, margin=dict(l=40, r=40, t=60, b=40))
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    if summarization_rows:
        summarization_rows = sorted(
            summarization_rows,
            key=lambda row: (row.get("model", ""), row.get("variant", "")),
        )
        fig = px.bar(
            summarization_rows,
            x="variant",
            y="rougeL_f",
            color="model",
            facet_col="technique",
            barmode="group",
            title="ROUGE-L po pipeline varijanti i modelu (sumarizacija)",
            labels={"rougeL_f": "ROUGE-L", "variant": "Pipeline varijanta", "model": "Model"},
        )
        fig.update_layout(height=450, margin=dict(l=40, r=40, t=60, b=40))
        charts.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

        heatmap_rows = summarization_rows
        if heatmap_rows:
            import pandas as pd

            frame = pd.DataFrame(heatmap_rows)
            if not frame.empty and "rougeL_f" in frame.columns:
                pivot = frame.pivot_table(
                    index="model",
                    columns="variant",
                    values="rougeL_f",
                    aggfunc="mean",
                )
                fig_heat = px.imshow(
                    pivot,
                    text_auto=".3f",
                    aspect="auto",
                    title="Heatmap: ROUGE-L (model × pipeline varijanta)",
                    labels=dict(color="ROUGE-L"),
                )
                fig_heat.update_layout(height=400, margin=dict(l=40, r=40, t=60, b=40))
                charts.append(pio.to_html(fig_heat, full_html=False, include_plotlyjs=False))

    if not charts:
        return ""

    return (
        '<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>'
        + '<div class="charts">'
        + "".join(f'<div class="chart">{chart}</div>' for chart in charts)
        + "</div>"
    )


def write_html_report(
    rows: List[Dict[str, Any]],
    output_path: Path,
    title: str = "Uporedna analiza modela i pipeline varijanti",
) -> None:
    table_html = _build_html_table(rows)
    charts_html = _build_plotly_charts(rows)

    html = f"""<!DOCTYPE html>
<html lang="sr">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #222;
      background: #fafafa;
    }}
    h1, h2 {{
      color: #1a1a2e;
    }}
    .comparison-table {{
      width: 100%;
      border-collapse: collapse;
      margin: 16px 0 32px;
      background: #fff;
    }}
    .comparison-table th, .comparison-table td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: left;
      font-size: 14px;
    }}
    .comparison-table th {{
      background: #1a1a2e;
      color: #fff;
    }}
    .comparison-table tr:nth-child(even) {{
      background: #f5f5f5;
    }}
    .chart {{
      background: #fff;
      padding: 12px;
      margin-bottom: 24px;
      border: 1px solid #e0e0e0;
      border-radius: 8px;
    }}
    .meta {{
      color: #555;
      margin-bottom: 20px;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p class="meta">Broj eksperimenata u tabeli: {len(rows)}</p>
  <h2>Tabela rezultata</h2>
  {table_html}
  <h2>Grafici</h2>
  {charts_html or "<p>Nema dovoljno podataka za grafik.</p>"}
</body>
</html>
"""

    output_path.write_text(html, encoding="utf-8")

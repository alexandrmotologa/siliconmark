"""JSON exporter for SiliconMark benchmark run results."""

from __future__ import annotations

from pathlib import Path
from siliconmark.core.config import SiliconMarkRunResult


def export_to_json(result: SiliconMarkRunResult, output_path: str | Path) -> str:
    """Export benchmark result to JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = result.model_dump_json(indent=2)
    path.write_text(content, encoding="utf-8")
    return str(path)

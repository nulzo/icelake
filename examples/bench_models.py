"""Benchmark matrix: run the e2e sim against several chat models, aggregate results.

Each model runs in an isolated subprocess of e2e_simulation.py (fresh db per
model; a crash in one run doesn't kill the matrix). Runs execute concurrently
(--jobs to throttle, e.g. if your provider rate-limits); each run's full output
lands in <out>/<model>.log. Writes <out>/report.json (full detail) and
<out>/report.md (comparison table).

    uv run python examples/bench_models.py
    uv run python examples/bench_models.py --models google/gemini-3.7-flash,openai/gpt-4o-mini
    uv run python examples/bench_models.py --jobs 2 --out bench_runs/aug27
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# Defaults all have entries in the meter's price table, so cost estimates are
# meaningful. Any OpenRouter chat model id works; unknown models report cost 0.
DEFAULT_MODELS = ("google/gemini-3.7-flash",)

COLUMNS = (
    "model",
    "hard (A+B)",
    "expectations (A+B)",
    "llm calls",
    "prompt tok",
    "est. cost",
    "duration",
)


def _slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "_")


def _counts(raw: dict[str, Any]) -> tuple[str, str, str]:
    suites = raw.get("suites", {})
    a, b = suites.get("A", {}), suites.get("B", {})
    hard = (
        f"{a.get('checks', 0) - a.get('failures', 0)}/{a.get('checks', 0)} + "
        f"{b.get('checks', 0) - b.get('failures', 0)}/{b.get('checks', 0)}"
    )
    exp = (
        f"{a.get('expectations', 0) - a.get('weak', 0)}/{a.get('expectations', 0)} + "
        f"{b.get('expectations', 0) - b.get('weak', 0)}/{b.get('expectations', 0)}"
    )
    llm = raw.get("llm", {})
    usage = f"{llm.get('calls', 0)}|{llm.get('prompt_tokens', 0)}|${llm.get('cost_usd', 0.0):.6f}"
    return hard, exp, usage


def _report_markdown(raws: list[dict[str, Any]]) -> str:
    lines = [
        "# Model benchmark — e2e simulation",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}. Each row is one full run of",
        "`examples/e2e_simulation.py` (suite A drain-mode + suite B worker-mode).",
        "",
        "| " + " | ".join(COLUMNS) + " |",
        "|" + "---|" * len(COLUMNS),
    ]
    for raw in raws:
        if "error" in raw:
            lines.append(f"| {raw['model']} | crashed | — | — | — | — | — |")
            continue
        hard, exp, usage = _counts(raw)
        calls, tokens, cost = usage.split("|")
        duration = f"{raw.get('duration_seconds', 0):.0f}s"
        lines.append(
            f"| {raw['model']} | {hard} | {exp} | {calls} | {tokens} | {cost} | {duration} |"
        )
    for key, title in (("failed", "Failed hard checks"), ("weak_checks", "Weak expectations")):
        lines += ["", f"## {title} by model", ""]
        for raw in raws:
            if "error" in raw:
                lines.append(f"- **{raw['model']}**: {raw['error']}")
                continue
            names = [
                name for suite in raw.get("suites", {}).values() for name in suite.get(key, [])
            ]
            rendered = ", ".join(names) if names else "—"
            lines.append(f"- **{raw['model']}**: {rendered}")
    return "\n".join(lines) + "\n"


def _run_one(model: str, out: Path, sim: Path, extra: list[str]) -> dict[str, Any]:
    """One full sim run for one model, output captured to <out>/<model>.log."""
    name = _slug(model)
    raw_path = out / f"{name}.json"
    print(f"[start] {model}", flush=True)
    with (out / f"{name}.log").open("w") as log:
        proc = subprocess.run(
            [
                sys.executable,
                str(sim),
                "--model",
                model,
                "--db",
                str(out / f"{name}.db"),
                "--report",
                str(raw_path),
                *extra,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())
        raw["exit_code"] = proc.returncode
        return raw
    return {"model": model, "error": f"run crashed, exit {proc.returncode} (see {name}.log)"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models", default=",".join(DEFAULT_MODELS), help="comma-separated model ids"
    )
    parser.add_argument("--out", default=".bench", help="output directory (default .bench)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="max concurrent runs (default: all models at once)",
    )
    parser.add_argument(
        "--reasoning",
        choices=("minimal", "low", "medium", "high"),
        default=None,
        help="forwarded to every run, e.g. --reasoning low for reasoning models",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        help="forwarded to every run; 'none' omits the parameter (reasoning-model endpoints)",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    jobs = args.jobs or len(models)
    extra = ["--reasoning", args.reasoning] if args.reasoning else []
    extra += ["--temperature", args.temperature] if args.temperature else []
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = Path(__file__).resolve().parent / "e2e_simulation.py"

    started = time.monotonic()
    raws: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(_run_one, model, out, sim, extra): model for model in models}
        for future in as_completed(futures):
            raw = future.result()
            raws.append(raw)
            status = raw.get("error") or f"exit {raw['exit_code']}"
            print(f"[done] {raw['model']}: {status}", flush=True)
    order = {model: index for index, model in enumerate(models)}
    raws.sort(key=lambda raw: order.get(raw["model"], len(order)))
    crashed = any("error" in raw for raw in raws)

    (out / "report.json").write_text(json.dumps(raws, indent=2) + "\n")
    (out / "report.md").write_text(_report_markdown(raws))
    elapsed = time.monotonic() - started
    print(f"\nwrote {out / 'report.md'} and {out / 'report.json'} in {elapsed:.0f}s")
    raise SystemExit(1 if crashed else 0)


if __name__ == "__main__":
    main()

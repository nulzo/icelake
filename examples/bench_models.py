"""Benchmark matrix: run the e2e sim against several chat models, aggregate results.

Each model runs in an isolated subprocess of e2e_simulation.py (fresh db per
model; a crash in one run doesn't kill the matrix). Per-model OpenRouter knobs
live in MODELS — add a row there to keep testing it. Runs execute concurrently
(--jobs to throttle); each run's full output lands in <out>/<model>.log.
Writes <out>/report.json (full detail) and <out>/report.md (comparison table).

    uv run python examples/bench_models.py
    uv run python examples/bench_models.py --models z-ai/glm-5.3-flash,openai/gpt-4o-mini
    uv run python examples/bench_models.py --jobs 2 --out bench_runs/aug30
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, TypedDict


class ModelParams(TypedDict, total=False):
    """OpenRouter knobs forwarded to e2e_simulation.py.

    Omit a key to use the library default (temperature=0, no reasoning field,
    structured_outputs=strict). ``temperature="none"`` omits sampling entirely.
    ``reasoning="none"`` disables thinking — rejected when the model is mandatory.
    """

    reasoning: str
    temperature: str
    structured_outputs: str


# Catalog: every id you want to keep re-testing. Empty {} = library defaults.
# Mandatory-thinking models get the cheapest allowed effort (not none).
# GPT-5.x endpoints reject temperature; 404-on-json_schema models use json_object.
MODELS: dict[str, ModelParams] = {
    "z-ai/glm-5.3-flash": {"reasoning": "low"},
    # "z-ai/glm-4.6": {"reasoning": "low"},
    "z-ai/glm-4.5-air": {"reasoning": "low", "structured_outputs": "json_object"},
    "openai/gpt-4o-mini": {},
    "openai/gpt-4.1-mini": {},
    "openai/gpt-4.1-nano": {},
    "openai/gpt-5-mini": {"reasoning": "minimal", "temperature": "none"},
    "openai/gpt-5-nano": {"reasoning": "minimal", "temperature": "none"},
    "openai/gpt-5.6-luna": {"reasoning": "none", "temperature": "none"},
    # "openai/gpt-5.6-sol": {"reasoning": "minimal", "temperature": "none"},
    "openai/gpt-oss-120b": {},
    "google/gemini-3.7-flash": {"reasoning": "low"},
    "google/gemini-2.5-flash": {"reasoning": "low"},
    "google/gemini-3.1-flash-lite": {"reasoning": "low"},
    "google/gemini-3-flash-preview": {"reasoning": "low"},
    # "google/gemma-4-31b-it": {"reasoning": "low", "structured_outputs": "json_object"},
    # "inception/mercury-2": {},
    # "minimax/minimax-m3": {},
    "mistralai/mistral-small-3.2-24b-instruct": {},
    "qwen/qwen3-32b": {"reasoning": "low"},
    "qwen/qwen3.7-flash": {"structured_outputs": "json_object"},
    "qwen/qwen3-30b-a3b-instruct-2507": {},
    "deepseek/deepseek-v4-flash": {"reasoning": "low"},
    "deepseek/deepseek-v4-flash-0731": {"reasoning": "low"},
    # "x-ai/grok-4.3": {"reasoning": "low"},
    # "~anthropic/claude-haiku-latest": {},
    # "moonshotai/kimi-k2.5": {"reasoning": "low"},
    # "tencent/hy4-preview": {"reasoning": "low", "structured_outputs": "json_object"},
}

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


def _cli_flags(params: ModelParams) -> list[str]:
    extra: list[str] = []
    if reasoning := params.get("reasoning"):
        extra += ["--reasoning", reasoning]
    if temperature := params.get("temperature"):
        extra += ["--temperature", temperature]
    if structured := params.get("structured_outputs"):
        extra += ["--structured-outputs", structured]
    return extra


def _params_label(params: ModelParams) -> str:
    if not params:
        return "defaults"
    return " ".join(f"{key}={value}" for key, value in params.items())


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


def _run_one(model: str, params: ModelParams, out: Path, sim: Path) -> dict[str, Any]:
    """One full sim run for one model, output captured to <out>/<model>.log."""
    name = _slug(model)
    raw_path = out / f"{name}.json"
    extra = _cli_flags(params)
    print(f"[start] {model} ({_params_label(params)})", flush=True)
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
        raw["params"] = dict(params)
        return raw
    return {
        "model": model,
        "params": dict(params),
        "error": f"run crashed, exit {proc.returncode} (see {name}.log)",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        default=",".join(MODELS),
        help="comma-separated model ids (default: every key in MODELS)",
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
        choices=("none", "minimal", "low", "medium", "high"),
        default=None,
        help="override MODELS[id].reasoning for every selected run",
    )
    parser.add_argument(
        "--temperature",
        default=None,
        help="override MODELS[id].temperature for every selected run; 'none' omits sampling",
    )
    parser.add_argument(
        "--structured-outputs",
        choices=("strict", "json_object"),
        default=None,
        help="override MODELS[id].structured_outputs for every selected run",
    )
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    jobs = args.jobs or len(models)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sim = Path(__file__).resolve().parent / "e2e_simulation.py"
    overrides: ModelParams = {}
    if args.reasoning:
        overrides["reasoning"] = args.reasoning
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if args.structured_outputs:
        overrides["structured_outputs"] = args.structured_outputs

    started = time.monotonic()
    raws: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {}
        for model in models:
            params: ModelParams = {**MODELS.get(model, {}), **overrides}
            if model not in MODELS:
                label = _params_label(params)
                print(f"[warn] {model} is not in MODELS; running with {label}", flush=True)
            futures[pool.submit(_run_one, model, params, out, sim)] = model
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

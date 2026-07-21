"""EXP-0003 — AVQA with ``compile_enabled`` opt-in vs eager (OPT-0002).
Per ``BENCHMARKS.md`` this captures:
- configuration (warm-up, repetitions, batch / heads / seq / dim)
- raw per-call latency in milliseconds
- median / mean / stdev / min / max
- speedup ratio (sdpa / avqa)
Run:
    PYTHONPATH=src python benchmarks/repro_cpu.py \\
        --bench benchmarks/repro_compile.py \\
        --markdown --out benchmarks/raw/EXP-0003
The companion module ``benchmarks/repro_compile.py`` exercises the
``compile_enabled`` opt-in (OPT-0002).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional

from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    CodebookConfig,
    ExecutionConfig,
    RoutingConfig,
)

DEFAULT_SEQ_LENS: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_BATCH: int = 2
DEFAULT_HEADS: int = 4
DEFAULT_HEAD_DIM: int = 32
DEFAULT_NUM_CODEWORDS: int = 16
DEFAULT_BUDGET: int = 4
WARMUP: int = 5
REPS: int = 10
def make_inputs(
    batch: int, seq_len: int, embed_dim: int, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq_len, embed_dim, generator=gen)
    k = torch.randn(batch, seq_len, embed_dim, generator=gen)
    v = torch.randn(batch, seq_len, embed_dim, generator=gen)
    return q, k, v
def make_module(embed_dim: int, num_heads: int, *, compile_enabled: bool) -> AVQAttention:
    config = AVQConfig(
        attention=AttentionShapeConfig(
            embed_dim=embed_dim, num_heads=num_heads, head_dim=DEFAULT_HEAD_DIM
        ),
        codebook=CodebookConfig(
            num_codewords=DEFAULT_NUM_CODEWORDS,
            children_per_codeword=2,
        ),
        routing=RoutingConfig(refinement_budget=DEFAULT_BUDGET),
        execution=ExecutionConfig(compile_enabled=compile_enabled),
    )
    mod = AVQAttention(config, in_proj=False, out_proj=False)
    mod.eval()
    return mod
def bench(fn: object) -> dict[str, float]:
    for _ in range(WARMUP):
        fn()
    samples: list[float] = []
    for _ in range(REPS):
        torch.manual_seed(0)
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-0003 OPT-0002 compile_enabled benchmark")
    parser.add_argument("--out", type=str, default="benchmarks/raw/EXP-0003")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    env_info = {
        "python": "n/a",
        "torch": torch.__version__,
        "warmup": WARMUP,
        "reps": REPS,
        "batch": DEFAULT_BATCH,
        "heads": DEFAULT_HEADS,
        "head_dim": DEFAULT_HEAD_DIM,
        "num_codewords": DEFAULT_NUM_CODEWORDS,
        "refinement_budget": DEFAULT_BUDGET,
        "sequence_lengths": list(DEFAULT_SEQ_LENS),
        "compile_enabled": True,
    }
    embed_dim = DEFAULT_HEADS * DEFAULT_HEAD_DIM
    eager = make_module(embed_dim, DEFAULT_HEADS, compile_enabled=False)
    compiled = make_module(embed_dim, DEFAULT_HEADS, compile_enabled=True)
    # Sync codebook so the comparison is fair (BENCHMARKS.md §Hardware).
    with torch.no_grad():
        compiled.codebook.parents.copy_(eager.codebook.parents)
        compiled.codebook.children.copy_(eager.codebook.children)
    def sdpa_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return functional.scaled_dot_product_attention(q, k, v)
    rows: list[dict[str, object]] = []
    for seq_len in DEFAULT_SEQ_LENS:
        q, k, v = make_inputs(
            batch=DEFAULT_BATCH,
            seq_len=seq_len,
            embed_dim=embed_dim,
            seed=0,
        )
        sdpa_stats = bench(lambda: sdpa_call(q, k, v))
        eager_stats = bench(lambda: eager(q, k, v, mask=None))
        compiled_stats = bench(lambda: compiled(q, k, v, mask=None))
        rows.append(
            {
                "sequence_length": seq_len,
                "sdpa": sdpa_stats,
                "avqa_eager": eager_stats,
                "avqa_compiled": compiled_stats,
                "speedup_eager": sdpa_stats["median_ms"] / eager_stats["median_ms"],
                "speedup_compiled": sdpa_stats["median_ms"] / compiled_stats["median_ms"],
            }
        )
    payload = {"config": env_info, "results": rows}
    raw_path = out_dir / "raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(env_info, indent=2, sort_keys=True))
    if args.markdown:
        lines = ["# EXP-0003 summary", "", "## Configuration", ""]
        lines.append(f"- torch: ``{env_info['torch']}``")
        lines.append(f"- warmup: ``{env_info['warmup']}``")
        lines.append(f"- reps: ``{env_info['reps']}``")
        lines.append(f"- compile_enabled: ``{env_info['compile_enabled']}``")
        lines.extend(
            [
                "",
                "| seq_len | sdpa | eager | compiled | sdpa/eager | sdpa/compiled |",
                "|--------:|-----:|------:|---------:|-----------:|--------------:|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['sequence_length']} | "
                f"{row['sdpa']['median_ms']:.3f} | "
                f"{row['avqa_eager']['median_ms']:.3f} | "
                f"{row['avqa_compiled']['median_ms']:.3f} | "
                f"{row['speedup_eager']:.2f} | "
                f"{row['speedup_compiled']:.2f} |"
            )
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {raw_path}")
    print(f"wrote {config_path}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())

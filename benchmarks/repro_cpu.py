"""EXP-0001 — CPU baseline reproduction of AVQA vs PyTorch SDPA.

Per ``BENCHMARKS.md`` every benchmark must record hardware, software
versions, configuration, raw metrics, and statistical summaries. This
script runs the v0.1.0 ``TorchBackend`` against
``torch.nn.functional.scaled_dot_product_attention`` on a CPU, sweeping
sequence lengths while keeping batch size, head count, and embedding
dimension fixed.

Run:

    PYTHONPATH=src python benchmarks/repro_cpu.py --markdown

Outputs:

    benchmarks/raw/EXP-0001/raw.json
    benchmarks/raw/EXP-0001/config.json
    benchmarks/raw/EXP-0001/summary.md (if --markdown)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import time

import torch
import torch.nn.functional as F  # noqa: N812

from avqa import AVQAttention, AVQConfig
from avqa.config import AttentionShapeConfig, CodebookConfig, RoutingConfig

DEFAULT_SEQ_LENS: tuple[int, ...] = (128, 256, 512, 1024)
DEFAULT_BATCH: int = 2
DEFAULT_HEADS: int = 4
DEFAULT_HEAD_DIM: int = 32
DEFAULT_NUM_CODEWORDS: int = 16
DEFAULT_BUDGET: int = 4
WARMUP: int = 5
REPS: int = 10


def make_inputs(
    batch: int,
    seq_len: int,
    embed_dim: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate deterministic random Q/K/V tensors of shape ``[B, T, E]``."""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(batch, seq_len, embed_dim, generator=gen)
    k = torch.randn(batch, seq_len, embed_dim, generator=gen)
    v = torch.randn(batch, seq_len, embed_dim, generator=gen)
    return q, k, v


def make_avqa(_heads: int, embed_dim: int) -> AVQAttention:
    """Build a small AVQA module sized for ``embed_dim``."""
    config = AVQConfig(
        attention=AttentionShapeConfig(
            embed_dim=embed_dim,
            num_heads=DEFAULT_HEADS,
            head_dim=DEFAULT_HEAD_DIM,
        ),
        codebook=CodebookConfig(
            num_codewords=DEFAULT_NUM_CODEWORDS,
            children_per_codeword=2,
        ),
        routing=RoutingConfig(refinement_budget=DEFAULT_BUDGET),
    )
    module = AVQAttention(config, in_proj=False, out_proj=False)
    module.eval()
    return module


def _bench(
    fn: object,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> dict[str, float]:
    """Run ``fn`` ``REPS`` times after ``WARMUP`` warm-ups, returning stats."""
    for _ in range(WARMUP):
        fn(q, k, v)
    samples_ms: list[float] = []
    for _ in range(REPS):
        torch.manual_seed(0)
        start = time.perf_counter()
        fn(q, k, v)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        samples_ms.append(elapsed_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "mean_ms": statistics.fmean(samples_ms),
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "samples_ms": samples_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AVQA vs SDPA CPU baseline")
    parser.add_argument("--out", type=str, default="benchmarks/raw/EXP-0001")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env_info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cpu_count": torch.get_num_threads() or 1,
        "seed": 0,
        "warmup": WARMUP,
        "reps": REPS,
        "batch": DEFAULT_BATCH,
        "heads": DEFAULT_HEADS,
        "head_dim": DEFAULT_HEAD_DIM,
        "embed_dim": DEFAULT_HEADS * DEFAULT_HEAD_DIM,
        "num_codewords": DEFAULT_NUM_CODEWORDS,
        "refinement_budget": DEFAULT_BUDGET,
        "sequence_lengths": list(DEFAULT_SEQ_LENS),
    }

    avqa = make_avqa(DEFAULT_HEADS, env_info["embed_dim"])

    def sdpa_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.scaled_dot_product_attention(q, k, v)

    def avqa_call(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return avqa(q, k, v, mask=None)

    rows: list[dict[str, object]] = []
    for seq_len in DEFAULT_SEQ_LENS:
        q, k, v = make_inputs(
            batch=DEFAULT_BATCH,
            seq_len=seq_len,
            embed_dim=env_info["embed_dim"],
            seed=0,
        )
        sdpa_stats = _bench(sdpa_call, q, k, v)
        avqa_stats = _bench(avqa_call, q, k, v)
        rows.append(
            {
                "sequence_length": seq_len,
                "sdpa": sdpa_stats,
                "avqa": avqa_stats,
                "speedup": sdpa_stats["median_ms"] / avqa_stats["median_ms"],
            }
        )

    payload = {
        "config": env_info,
        "results": rows,
    }

    raw_path = out_dir / "raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(env_info, indent=2, sort_keys=True))

    if args.markdown:
        lines = ["# EXP-0001 summary", "", "## Configuration", ""]
        for key in (
            "python",
            "torch",
            "platform",
            "cpu_count",
            "warmup",
            "reps",
        ):
            lines.append(f"- {key}: ``{env_info[key]}``")
        lines.extend(
            [
                "",
                "| seq_len | sdpa median ms | avqa median ms | avqa/sdpa |",
                "|--------:|---------------:|---------------:|-----------|",
            ]
        )
        for row in rows:
            lines.append(
                f"| {row['sequence_length']} | "
                f"{row['sdpa']['median_ms']:.3f} | "
                f"{row['avqa']['median_ms']:.3f} | "
                f"{row['speedup']:.2f} |"
            )
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {raw_path}")
    print(f"wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

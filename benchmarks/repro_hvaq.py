"""EXP-0006 - HVAQ (Hopfield-VQ-Attention) benchmark.

Per ``BENCHMARKS.md`` this captures:

- the paper-exact baseline (fixed-temperature softmax);
- HVAQ-ENT (per-query beta_q from the router's top-P entropy);
- HVAQ-LIN (per-query beta_q linear in the entropy).

The benchmark is CPU-friendly: it uses small synthetic data and
exercises the AVQAttention forward at each adaptive schedule.

Run:

    PYTHONPATH=src python benchmarks/repro_hvaq.py --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F  # noqa: N812

from avqa import AVQAttention, AVQConfig
from avqa.config import (
    AttentionShapeConfig,
    BackendConfig,
    CodebookConfig,
    HopfieldConfig,
    RefinementConfig,
    RoutingConfig,
)

DEFAULT_BATCH: int = 2
DEFAULT_HEADS: int = 4
DEFAULT_HEAD_DIM: int = 16
DEFAULT_NUM_CODEWORDS: int = 16
DEFAULT_CHILDREN_PER: int = 4
DEFAULT_BUDGET: int = 4
DEFAULT_SEQ_LEN: int = 64
WARMUP: int = 3
REPS: int = 10


def _build_attention(*, hopfield_enabled: bool, adaptive: str) -> AVQAttention:
    config = AVQConfig(
        attention=AttentionShapeConfig(
            embed_dim=DEFAULT_HEADS * DEFAULT_HEAD_DIM,
            num_heads=DEFAULT_HEADS,
            head_dim=DEFAULT_HEAD_DIM,
        ),
        codebook=CodebookConfig(
            num_codewords=DEFAULT_NUM_CODEWORDS,
            children_per_codeword=DEFAULT_CHILDREN_PER,
        ),
        routing=RoutingConfig(refinement_budget=DEFAULT_BUDGET),
        refinement=RefinementConfig(enabled=True),
        backend=BackendConfig(hopfield=hopfield_enabled),
        hopfield=HopfieldConfig(
            enabled=hopfield_enabled,
            adaptive=adaptive,
            beta_init=0.0,
        ),
    )
    mod = AVQAttention(config, in_proj=False, out_proj=False)
    mod.eval()
    return mod


def _bench(fn: object) -> dict[str, float]:
    for _ in range(WARMUP):
        fn()
    samples: list[float] = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EXP-0006 HVAQ benchmark")
    parser.add_argument("--out", type=str, default="benchmarks/raw/EXP-0006")
    parser.add_argument("--markdown", action="store_true")
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = {
        "batch": DEFAULT_BATCH,
        "heads": DEFAULT_HEADS,
        "head_dim": DEFAULT_HEAD_DIM,
        "num_codewords": DEFAULT_NUM_CODEWORDS,
        "children_per_codeword": DEFAULT_CHILDREN_PER,
        "refinement_budget": DEFAULT_BUDGET,
        "seq_len": DEFAULT_SEQ_LEN,
        "warmup": WARMUP,
        "reps": REPS,
    }

    torch.manual_seed(0)
    embed_dim = DEFAULT_HEADS * DEFAULT_HEAD_DIM
    q = torch.randn(DEFAULT_BATCH, DEFAULT_SEQ_LEN, embed_dim)
    k = torch.randn(DEFAULT_BATCH, DEFAULT_SEQ_LEN, embed_dim)
    v = torch.randn(DEFAULT_BATCH, DEFAULT_SEQ_LEN, embed_dim)

    def sdpa_call() -> object:
        qh = q.reshape(DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_HEADS, DEFAULT_HEAD_DIM).transpose(
            1, 2
        )
        kh = k.reshape(DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_HEADS, DEFAULT_HEAD_DIM).transpose(
            1, 2
        )
        vh = v.reshape(DEFAULT_BATCH, DEFAULT_SEQ_LEN, DEFAULT_HEADS, DEFAULT_HEAD_DIM).transpose(
            1, 2
        )
        return F.scaled_dot_product_attention(qh, kh, vh)

    sdpa_stats = _bench(sdpa_call)

    paper = _build_attention(hopfield_enabled=False, adaptive="none")
    hvaq_ent = _build_attention(hopfield_enabled=True, adaptive="entropy")
    hvaq_lin = _build_attention(hopfield_enabled=True, adaptive="linear")

    def paper_call() -> object:
        return paper(q, k, v, mask=None)

    def hvaq_ent_call() -> object:
        return hvaq_ent(q, k, v, mask=None)

    def hvaq_lin_call() -> object:
        return hvaq_lin(q, k, v, mask=None)

    paper_stats = _bench(paper_call)
    hvaq_ent_stats = _bench(hvaq_ent_call)
    hvaq_lin_stats = _bench(hvaq_lin_call)

    with torch.no_grad():
        for m in (hvaq_ent, hvaq_lin):
            m.codebook.parents.copy_(paper.codebook.parents)
            m.codebook.children.copy_(paper.codebook.children)
    with torch.no_grad():
        out_paper = paper(q, k, v, mask=None)
        out_ent = hvaq_ent(q, k, v, mask=None)
        out_lin = hvaq_lin(q, k, v, mask=None)
    paper_vs_ent = float((out_paper - out_ent).abs().max().item())
    paper_vs_lin = float((out_paper - out_lin).abs().max().item())

    rows = {
        "sdpa": sdpa_stats,
        "paper_single_pass": paper_stats,
        "hvaq_entropy": hvaq_ent_stats,
        "hvaq_linear": hvaq_lin_stats,
        "paper_vs_hvaq_entropy_max_abs_diff": paper_vs_ent,
        "paper_vs_hvaq_linear_max_abs_diff": paper_vs_lin,
    }

    raw_path = out_dir / "raw.json"
    raw_path.write_text(json.dumps({"config": env, "rows": rows}, indent=2, sort_keys=True))
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(env, indent=2, sort_keys=True))

    if args.markdown:
        lines = [
            "# EXP-0006 summary",
            "",
            "## Configuration",
            "",
        ]
        for k_, v_ in env.items():
            lines.append(f"- {k_}: `{v_}`")
        lines.extend(
            [
                "",
                "| method | median ms | mean ms | stdev ms |",
                "|--------|----------:|--------:|---------:|",
            ]
        )
        for name, stats in (
            ("sdpa", sdpa_stats),
            ("paper single-pass", paper_stats),
            ("hvaq entropy", hvaq_ent_stats),
            ("hvaq linear", hvaq_lin_stats),
        ):
            lines.append(
                f"| {name} | {stats['median_ms']:.3f} | "
                f"{stats['mean_ms']:.3f} | {stats['stdev_ms']:.3f} |"
            )
        lines.append("")
        lines.append("## Attention output (vs paper)")
        lines.append(f"- HVAQ-ENT max abs diff: {paper_vs_ent:.4f}")
        lines.append(f"- HVAQ-LIN max abs diff: {paper_vs_lin:.4f}")
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {raw_path}")
    print(f"wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

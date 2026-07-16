"""EXP-0004 \u2014 BCAR (online codebook adaptation) convergence benchmark.

Per ``BENCHMARKS.md`` this captures:

- configuration (warm-up, repetitions, batch / heads / seq / dim)
- raw per-call latency in milliseconds (informational only)
- VQ reconstruction loss at every checkpoint step

Methods compared:
- ``static``    : the paper's behavior; codebook is frozen after a random init.
- ``bcar``      : BCAR (OPT-0003); online EMA updates every step.
- ``oracle``    : offline 1-pass k-means on the entire stream (cheat ceiling).

Run::

    PYTHONPATH=src python benchmarks/repro_bcar.py --markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch

from avqa.codebook import HierarchicalCodebook
from avqa.online_adaptation import online_codebook_adaptation

DEFAULT_NUM_CODEWORDS: int = 4
DEFAULT_CHILDREN_PER: int = 2
DEFAULT_HEAD_DIM: int = 8
DEFAULT_NUM_HEADS: int = 1
DEFAULT_NUM_STEPS: int = 1024
DEFAULT_BATCH_TOKENS: int = 8


def synthetic_stream(
    centroids: torch.Tensor, num_steps: int, batch_tokens: int, *, sigma: float = 0.1
) -> torch.Tensor:
    """Sample tokens around ``centroids`` cyclically."""
    n_centroids = centroids.shape[0]
    out = torch.empty(num_steps, batch_tokens, centroids.shape[-1])
    for step in range(num_steps):
        out[step] = centroids[step % n_centroids] + sigma * torch.randn(
            batch_tokens, centroids.shape[-1]
        )
    return out


def vq_loss(keys: torch.Tensor, codebook: HierarchicalCodebook) -> float:
    """Mean L2 squared distance from each key to its assigned parent."""
    H, M0, D = codebook.parents.shape
    flat_parents = codebook.parents.reshape(H * M0, D)
    keys_flat = keys.reshape(-1, D).to(flat_parents.dtype)
    dist = torch.cdist(keys_flat, flat_parents)
    assign = dist.argmin(dim=-1)
    assigned = flat_parents[assign].reshape_as(keys)
    return float(((keys - assigned) ** 2).mean().item())


def offline_kmeans_centroid(
    centroids: torch.Tensor, *, seed: int, max_iter: int = 32
) -> torch.Tensor:
    """Single-cluster-per-codeword offline k-means for the oracle baseline.

    Returns an ``[H, M_0, D]`` centroid tensor that approximates the
    stream centroids.
    """
    g = torch.Generator().manual_seed(seed)
    H, M0 = DEFAULT_NUM_HEADS, DEFAULT_NUM_CODEWORDS
    D = centroids.shape[-1]
    parents = centroids[torch.randperm(centroids.shape[0])[:M0]] + 0.05 * torch.randn(
        M0, D, generator=g
    )
    keys_flat = centroids
    for _ in range(max_iter):
        dist = torch.cdist(keys_flat, parents)
        assign = dist.argmin(dim=-1)
        for k in range(M0):
            mask = assign == k
            if mask.any():
                parents[k] = keys_flat[mask].mean(dim=0)
    return parents.unsqueeze(0).expand(H, M0, D).contiguous()


def make_codebook(parents_init: torch.Tensor, children_init: torch.Tensor) -> HierarchicalCodebook:
    cb = HierarchicalCodebook(
        num_heads=parents_init.shape[0],
        num_parents=parents_init.shape[1],
        children_per_parent=children_init.shape[2],
        head_dim=parents_init.shape[-1],
    )
    cb.parents.copy_(parents_init)
    cb.children.copy_(children_init)
    return cb


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description="EXP-0004 BCAR convergence benchmark")
    parser.add_argument("--out", type=str, default="benchmarks/raw/EXP-0004")
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--tokens", type=int, default=DEFAULT_BATCH_TOKENS)
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    H = DEFAULT_NUM_HEADS
    M0 = DEFAULT_NUM_CODEWORDS
    C = DEFAULT_CHILDREN_PER
    D = DEFAULT_HEAD_DIM
    centroids = torch.randn(M0, D) * 4.0
    stream = synthetic_stream(centroids, args.steps, args.tokens)
    # oracle parents
    oracle_parents = offline_kmeans_centroid(centroids, seed=123)
    oracle_children = (oracle_parents.unsqueeze(2) + 0.05 * torch.randn(H, M0, C, D)).contiguous()

    # Static codebook: random init, frozen.
    static_parents = torch.randn(H, M0, D)
    static_children = torch.randn(H, M0, C, D)
    cb_static = make_codebook(static_parents, static_children.clone())

    # BCAR codebook: random init, online updates.
    bcar_parents = torch.randn(H, M0, D)
    bcar_children = torch.randn(H, M0, C, D)
    cb_bcar = make_codebook(bcar_parents, bcar_children.clone())

    checkpoints = sorted({1, 10, 50, 100, 250, 500, 1000, args.steps})

    static_losses: dict[int, float] = {}
    bcar_losses: dict[int, float] = {}

    # ``static`` VQ loss is constant (frozen codebook) \u2014 compute once.
    static_losses[args.steps] = vq_loss(stream[: args.steps], cb_static)

    # Online assignment function: assign each token to its current
    # nearest parent so the (parent, child) cells receive a realistic
    # distribution (mimicking the production VQ). Round-robin would
    # be degenerate because one parent would see all of one
    # centroid, leaving the others untrained.
    def current_assignments(
        cb: HierarchicalCodebook, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat_parents = cb.parents.reshape(H * M0, D)
        dist = torch.cdist(tokens.to(flat_parents.dtype), flat_parents)
        parent_assn = dist.argmin(dim=-1).reshape(1, H, -1)
        flat_children = cb.children.reshape(H * M0, C, D)
        # [H*M_0, C, D] gather by [H*M_0] yields [N, C, D].
        chosen = flat_children[parent_assn.reshape(-1)].to(tokens.dtype)
        # Manual L2 distance per token to each child.
        diff = tokens.unsqueeze(1).to(chosen.dtype) - chosen
        child_assn = (diff * diff).sum(dim=-1).argmin(dim=-1).reshape(1, H, -1)
        return parent_assn, child_assn

    for step in range(args.steps):
        tokens = stream[step]  # [B, D]
        parent_assn, child_assn = current_assignments(cb_bcar, tokens)
        online_codebook_adaptation(
            tokens[None],
            parents=cb_bcar.parents,
            children=cb_bcar.children,
            parent_assignments=parent_assn,
            child_assignments=child_assn,
            decay=0.1,
        )
        if (step + 1) in checkpoints:
            bcar_losses[step + 1] = vq_loss(stream[: step + 1], cb_bcar)
            static_losses[step + 1] = vq_loss(stream[: step + 1], cb_static)

    # Per-step latency (informational).
    timings: dict[str, float] = {}
    import time

    for name, cb in (("static", cb_static), ("bcar", cb_bcar)):
        samples: list[float] = []
        for _ in range(20):
            tokens = stream[0]
            t0 = time.perf_counter()
            if name == "bcar":
                online_codebook_adaptation(
                    tokens[None],
                    parents=cb.parents,
                    children=cb.children,
                    parent_assignments=torch.zeros(1, H, tokens.shape[0], dtype=torch.long),
                    child_assignments=torch.zeros(1, H, tokens.shape[0], dtype=torch.long),
                    decay=0.1,
                )
            t1 = time.perf_counter()
            samples.append((t1 - t0) * 1000)
        timings[name] = statistics.median(samples)

    oracle_loss = vq_loss(stream, make_codebook(oracle_parents, oracle_children))

    payload = {
        "config": {
            "num_heads": H,
            "num_codewords": M0,
            "children_per_codeword": C,
            "head_dim": D,
            "num_steps": args.steps,
            "tokens_per_step": args.tokens,
        },
        "static_loss": static_losses,
        "bcar_loss": bcar_losses,
        "oracle_loss": oracle_loss,
        "latency_ms": timings,
    }
    raw_path = out_dir / "raw.json"
    raw_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    config_path = out_dir / "config.json"
    config_path.write_text(json.dumps(payload["config"], indent=2, sort_keys=True))

    if args.markdown:
        lines = ["# EXP-0004 summary", "", "## Configuration", ""]
        for k, v in payload["config"].items():
            lines.append(f"- {k}: ``{v}``")
        lines.extend(
            [
                "",
                "| steps | static | bcar | oracle | improvement vs static |",
                "------:|-------:|-----:|-------:|----------------------:|",
            ]
        )
        for step in sorted(static_losses.keys()):
            static_v = static_losses[step]
            bcar_v = bcar_losses[step]
            improvement = 1.0 - (bcar_v / static_v)
            lines.append(
                f"| {step} | {static_v:.4f} | {bcar_v:.4f} | "
                f"{oracle_loss:.4f} | {improvement * 100:5.1f}% |"
            )
        lines.append("")
        lines.append("## Latency (CPU, median over 20)")
        for k, v in timings.items():
            lines.append(f"- ``{k}``: {v:.4f} ms / call")
        (out_dir / "summary.md").write_text("\n".join(lines) + "\n")

    print(f"wrote {raw_path}")
    print(f"wrote {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Optional framework integration packages.

This subpackage is intentionally empty in the core AVQA distribution.
Earlier revisions shipped HF, vLLM, FlashAttention, and xFormers
adapters here; those have been removed because the upstream packages
have version-pinned dependencies that conflict with the AVQA core
toolchain.  Re-introduce specific adapters as separate distribution
extras if you need them.
"""

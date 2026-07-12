"""Sovereign host inference agent — Metal Phase 2 (design.md §2.6).

Runs ON the macOS host (launchd-managed), supervises one llama.cpp server
per role, and exposes a single private, token-authenticated port that the
metal-arm64 runtime container reaches via host.docker.internal. The engine
behind the agent is an internal detail: llama.cpp today, host-native
sovereign-vllm later, with no contract change.
"""

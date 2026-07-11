"""Sovereign Runtime appliance layer.

Owns the design.md §24 image contract: configuration, the §3.2 state machine,
health/manifest/errors endpoints, and engine supervision. Implemented in M3;
this package must always import without vLLM present (the engine is lazy).
"""

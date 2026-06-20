"""Shared pytest fixtures/config.

Test tiers are registered as markers in pyproject.toml: ``unit``, ``integration``, ``validation``.
Validation tests compare against analytic results or the FDTDX (JAX) / MEEP reference oracles
(install the cross-check oracle with ``uv sync --extra validation``). See the
``physics-validation`` skill.
"""

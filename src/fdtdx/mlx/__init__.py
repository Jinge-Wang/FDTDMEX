"""Native MLX (Metal) forward FDTD engine for Apple Silicon.

This subpackage holds the pure-MLX translation of the fdtdx forward time loop
(curl -> E/H update -> CPML -> source injection -> detector record -> time loop) plus
the array bridge that converts an :class:`fdtdx.fdtd.container.ArrayContainer` to/from a
plain-MLX state. It is imported lazily by :mod:`fdtdx.backend.dispatch` only when the
MLX path is taken, so ``import mlx.core`` never happens on non-Apple platforms.

Design (see docs/architecture.md): MLX is functional / out-of-place, which makes the
Yee update race-free without ping-pong buffers. The time loop is a plain Python ``for``
loop; the lazy graph is bounded with periodic ``mx.eval``.
"""

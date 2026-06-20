"""WS-C — subpixel smoothing (Kottke/Farjadpour effective-tensor averaging).

Host-side pre-time-stepping step: estimate the interface normal, compute arithmetic/harmonic fill
averages, and project into an effective inverse-permittivity **tensor** per Yee component. Output
feeds the engine's 9-component path (smoothing makes even isotropic geometry locally anisotropic at
tilted interfaces). Algorithm reference: docs/subpixel-smoothing.md; code reference to adapt:
../meep/src/anisotropic_averaging.cpp.

Status: stub.
"""

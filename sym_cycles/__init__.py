"""
Symphonia S02 – sym_cycles package
Cycle/Phase/Backbone/Compass offline stack.

Modules:
- builder_v1_0                          : cycle detectie per sensor
- backbone_v1_1                         : stereo backbone synthese
- phase_universe_v1_0                   : phase_class, run_frac, phase_bins
- phase_tiles_v3_0                      : PhaseTiles v3 (dt_ab_us)
- compass_sign_v3                       : Canonical direction core
- projector_v1_0                        : PureSteps → backbone projectie
- movement_body_v3.py                   : motor awareness
- realtime_compass.py                   : realtime compass
- realtime_states_v1_9_canonical.py     : realtime pipeline
- pipeline_offline                      : end-to-end offline pipeline
- demo_check                            : diagnostische checks
"""

__all__ = [
    "builder_v1_0",
    "backbone_v1_1",
    "phase_universe_v1_0",
    "phase_tiles_v3_0",
    "compass_sign_v3",
    "projector_v1_0",
    "movement_body_v3.py",
    "realtime_compass.py",
    "realtime_states_v1_9_canonical.py",
    "pipeline_offline",
    "demo_check",
]

# versie van het pakket
VERSION = "1.0"

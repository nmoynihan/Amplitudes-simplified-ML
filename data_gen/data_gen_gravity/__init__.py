"""Five-point scalar/graviton data generation.

The public surface intentionally mirrors the scalar-QED generator:

``generate_target -> expand -> scramble -> validate -> tokenise``.
"""

from .core import (
    BENCHMARKS,
    PROCESS_SPECS,
    ProcessSpec,
    expand_expression,
    generate_target,
    validate_expression_pair,
)
from .kinematics import SpinorKinematics, generate_kinematics

__all__ = [
    "BENCHMARKS",
    "PROCESS_SPECS",
    "ProcessSpec",
    "SpinorKinematics",
    "expand_expression",
    "generate_kinematics",
    "generate_target",
    "validate_expression_pair",
]

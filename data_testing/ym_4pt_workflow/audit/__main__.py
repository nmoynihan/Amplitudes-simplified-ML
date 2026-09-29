"""Run with ``python -m data_testing.ym_4pt_workflow.audit``."""

from . import run_all
from ._io import command_line


if __name__ == "__main__":
    command_line(run_all, "Reproduce all exact four-gluon physics audit checks")

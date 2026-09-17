#!/usr/bin/env python3
"""Generate Yang--Mills training pairs with fixed cyclic numerator ordering.

Defaults use the external-order conventions of the reference files
``data/data_ym/gluon4feyn1234.csv.gz`` (1,2,3,4) and
``data/data_ym/gluon5feyn12345.csv.gz`` (1,2,3,4,5). Those files contain expanded
dot products, not compact F-blocks. Their filenames specify the external cycles;
the per-block invariant below uses those same cycles. These defaults are defined
in ``notation.default_cyclic_order`` and recorded in logs and split manifests.

Each trace or open chain contains field-strength labels in a forward cyclic
subsequence of (1, ..., N): labels may be omitted and the sequence may wrap
around N to 1, but it may not permute or reverse that order. Each external
field strength occurs exactly once across all blocks of a numerator monomial.
The invariant applies to each block independently, not to the concatenation
of commuting scalar factors. Open-chain momentum endpoints are contractions,
not additional field strengths, and retain the original endpoint sampling.

This module opts into :mod:`generate`'s cyclic numerator policy while reusing
its complete generation, scrambling, validation, batching, and output pipeline.
Physical denominator channels already obey cyclic adjacency, including (N, 1);
the treatment of cancellable denominator factors is unchanged. Both supported
dataset modes (``oneshot`` and ``step``) use the existing Yang--Mills pair builder.

The existing N >= 6 limitation remains: genuine multiparticle poles
(p_i + ... + p_j)^2 are not modelled by the scalar-product pole representation.
Numerator ordering does not make this a complete N >= 6 physical pole model.

From the repository root::

    python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 4 --samples 1000 --jobs 1
    python -m data_gen.data_gen_ym.yang_mills_cyclic_generation 5 --samples 1000 --jobs 1

The historical ``python -m data_gen_ym.yang_mills_cyclic_generation`` invocation
from ``data_gen/`` and direct execution of this file are also supported. Default
CSV and log filenames contain ``cyclic`` and are distinct from the original
generator's defaults. Generation options are shared with that generator.

Add ``--test-size 100`` to hold out 100 rows whose tokenised simple targets and
scrambled inputs each appear only once in the pool. Repeated targets, including
different scrambled variants, stay together in training. Every source row is
preserved, and both train/test target and input intersections are checked to be
empty. ``--samples`` is the total source pool size, including held-out rows.
Split files and a verification manifest are
written to ``cyclic_train_test/`` beside the raw pool, or to
``--split-output-dir``. Selection uses ``--split-seed`` (default: 20260401),
and existing split outputs are protected unless ``--split-overwrite`` is set.
Splitting requires N >= 4, tokenisation, enough eligible singleton targets, and
at least one training row; without ``--test-size``, generation behaves as before.
The safeguard compares token sequences, not general algebraic equivalence.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


# The shared generator imports Tokenizer as a top-level module. Keep that
# historical import available under both package layouts and in spawn workers.
_DATA_GEN_DIR = Path(__file__).resolve().parent.parent
if str(_DATA_GEN_DIR) not in sys.path:
    sys.path.insert(0, str(_DATA_GEN_DIR))

if __package__:
    from . import generate as _generate
else:  # python data_gen/data_gen_ym/yang_mills_cyclic_generation.py
    from data_gen_ym import generate as _generate


DEFAULT_RAW_OUT_TEMPLATE = "gi_cyclic_{N}pt_{NSAMPS}k.csv"
DEFAULT_TOK_OUT_TEMPLATE = "gi_cyclic_{N}pt_tok_{NSAMPS}k.csv"
DEFAULT_LOG_OUT_TEMPLATE = "gen_data_cyclic_{N}pt_{NSAMPS}k.log"


def build_dataset(N: int, num_samples: int, **kwargs) -> list[tuple[str, str]]:
    """Build fixed-order pairs using all options of ``generate.build_dataset``.

    The ``cyclic_order`` policy is always enabled by this entry point.
    """
    return _generate.build_dataset(N, num_samples, cyclic_order=True, **kwargs)


def build_dataset_batched(N: int, num_samples: int, **kwargs) -> list[tuple[str, str]]:
    """Build fixed-order batches using the shared builder's batching options."""
    return _generate.build_dataset_batched(N, num_samples, cyclic_order=True, **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the shared CLI with cyclic numerator order and distinct output names."""
    _generate.main(
        argv,
        cyclic_order=True,
        raw_out_template=DEFAULT_RAW_OUT_TEMPLATE,
        tok_out_template=DEFAULT_TOK_OUT_TEMPLATE,
        log_out_template=DEFAULT_LOG_OUT_TEMPLATE,
    )


if __name__ == "__main__":
    main()

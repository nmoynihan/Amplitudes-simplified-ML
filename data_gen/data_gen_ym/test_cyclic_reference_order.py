"""Pin cyclic defaults and their provenance to the supplied Feynman datasets."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
import random
import re
import tempfile
import unittest
from unittest import mock

from ..Tokenizer import ScatteringAmplitudeTokenizer
from . import algebra
from . import expr_model as model
from . import generate
from . import notation
from . import split_cyclic_train_test as splitter


_ROOT = Path(__file__).resolve().parents[2]
_REFERENCES = {
    4: "data/data_ym/gluon4feyn1234.csv.gz",
    5: "data/data_ym/gluon5feyn12345.csv.gz",
}
_ORDERS = {4: (1, 2, 3, 4), 5: (1, 2, 3, 4, 5)}


def _denominator_channels(node, *, in_denominator=False):
    """Read momentum dots in denominators without policing numerator dots."""
    if isinstance(node, algebra._BinOp):
        yield from _denominator_channels(node.left, in_denominator=in_denominator)
        yield from _denominator_channels(
            node.right, in_denominator=in_denominator or node.op == "/",
        )
    elif isinstance(node, algebra._UnaryOp):
        yield from _denominator_channels(node.operand, in_denominator=in_denominator)
    elif isinstance(node, algebra._DotChain) and in_denominator:
        if len(node.parts) != 2 or any(vec.tag != "p" for vec in node.parts):
            raise AssertionError("reference denominator is not a two-momentum dot")
        yield tuple(sorted(vec.idx for vec in node.parts))


class CyclicReferenceOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        state = random.getstate()
        self.addCleanup(random.setstate, state)

    def test_explicit_defaults_match_the_reference_filename_orders(self) -> None:
        self.assertEqual(notation.DEFAULT_CYCLIC_ORDERS, _ORDERS)
        self.assertEqual(notation.CYCLIC_ORDER_REFERENCE_FILES, _REFERENCES)
        for N, expected in _ORDERS.items():
            with self.subTest(N=N):
                self.assertEqual(notation.default_cyclic_order(N), expected)
                self.assertIn("feyn" + "".join(map(str, expected)), _REFERENCES[N])

    def test_reference_contents_are_expanded_and_have_compatible_physical_channels(self) -> None:
        # These files are headerless [index, expression] rows. Their expanded
        # numerators cannot determine a unique colour cycle or an F-block word.
        # The filename supplies the cycle; the visible poles are a consistency
        # check, not the complete list of channels the generator may use.
        expected_visible = {4: {(1, 2)}, 5: {(1, 2), (4, 5)}}
        for N, relative_path in _REFERENCES.items():
            with self.subTest(N=N), gzip.open(_ROOT / relative_path, "rt", newline="") as handle:
                rows = list(csv.reader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(len(rows[0]), 2)
                self.assertEqual(rows[0][0], "1")
                expression = rows[0][1]
                self.assertNotRegex(expression, r"\b(?:F_\d+|Tr)\b")
                for tag in ("e", "p"):
                    labels = {int(label) for label in re.findall(rf"\b{tag}_(\d+)\b", expression)}
                    self.assertEqual(labels, set(_ORDERS[N]))
                tokens = algebra._tokenize(expression)
                parser = algebra._Parser(tokens)
                tree = parser.parse()
                self.assertEqual(parser.i, len(tokens))
                visible = set(_denominator_channels(tree))
                self.assertEqual(visible, expected_visible[N])
                physical = {tuple(model._dot_legs(pole)) for pole in model._all_physical_poles(N)}
                expected_physical = {
                    tuple(sorted((left, right)))
                    for left, right in zip(_ORDERS[N], _ORDERS[N][1:] + _ORDERS[N][:1])
                }
                self.assertEqual(physical, expected_physical)
                self.assertLess(visible, physical)
                self.assertIn((1, N), physical)

    def test_full_arity_words_follow_reference_orientation_with_every_wraparound(self) -> None:
        for N, cycle in _ORDERS.items():
            for offset in range(N):
                expected = cycle[offset:] + cycle[:offset]
                # Deliberately give the sampler a reversed ordering after the
                # starting label. Only its first label should choose a rotation.
                sampled = [expected[0], *reversed(expected[1:])]
                with self.subTest(N=N, offset=offset), mock.patch.object(
                    model.random, "sample", return_value=sampled,
                ):
                    ordered = tuple(model._sample_cyclic_labels(list(cycle), N, N))
                    self.assertEqual(ordered, expected)
                    self.assertNotEqual(ordered, tuple(sampled))

    def test_existing_seeded_cyclic_sampling_and_random_state_are_preserved(self) -> None:
        for N in (4, 5, 6):
            for arity in range(1, N + 1):
                for seed in (0, 17, 401):
                    with self.subTest(N=N, arity=arity, seed=seed):
                        baseline = random.Random(seed)
                        chosen = baseline.sample(list(range(1, N + 1)), arity)
                        expected = sorted(chosen, key=lambda label: (label - chosen[0]) % N)
                        random.seed(seed)
                        actual = model._sample_cyclic_labels(list(range(1, N + 1)), arity, N)
                        self.assertEqual(actual, expected)
                        self.assertEqual(random.getstate(), baseline.getstate())

    def test_six_point_fallback_keeps_the_original_cycle_and_all_adjacent_channels(self) -> None:
        self.assertEqual(notation.default_cyclic_order(6), (1, 2, 3, 4, 5, 6))
        self.assertNotIn(6, notation.CYCLIC_ORDER_REFERENCE_FILES)
        physical = {tuple(model._dot_legs(pole)) for pole in model._all_physical_poles(6)}
        self.assertEqual(physical, {(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 6)})

    def test_direct_and_batched_logs_record_reference_defaults_only_for_cyclic_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            log_path = Path(temp) / "generation.log"
            for N in (4, 5):
                for batched in (False, True):
                    for enabled in (False, True):
                        with self.subTest(N=N, batched=batched, cyclic_order=enabled), mock.patch.object(
                            generate, "_worker_build_dataset", return_value=[],
                        ):
                            kwargs = dict(log_path=str(log_path), cyclic_order=enabled, seed=17)
                            if batched:
                                generate.build_dataset_batched(N, 1, jobs=1, progress=False, **kwargs)
                            else:
                                generate.build_dataset(N, 0, **kwargs)
                            log = log_path.read_text(encoding="utf-8")
                            if enabled:
                                self.assertIn("particle_order=" + ",".join(map(str, _ORDERS[N])), log)
                                self.assertIn("ordering_reference=" + _REFERENCES[N], log)
                            else:
                                self.assertNotIn("particle_order=", log)
                                self.assertNotIn("ordering_reference=", log)

    def test_split_manifests_record_particle_order_and_reference_provenance(self) -> None:
        tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
        for N in (4, 5, 6):
            with self.subTest(N=N), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                raw_path = directory / "pool.csv"
                token_path = directory / "pool_tok.csv"
                block = "Tr(" + " · ".join(f"F_{label}" for label in range(1, N + 1)) + ")"
                rows = [(f"{coefficient}*{block}", f"{block}*{coefficient}") for coefficient in (2, 3)]
                for path, encoded in ((raw_path, False), (token_path, True)):
                    with path.open("w", newline="", encoding="utf-8") as handle:
                        writer = csv.writer(handle)
                        writer.writerow(("simple", "scrambled"))
                        for row in rows:
                            writer.writerow([
                                json.dumps(tokenizer.encode_infix(expr)) if encoded else expr
                                for expr in row
                            ])
                report = splitter.split_cyclic_dataset(
                    raw_path, token_path, n_particles=N, test_size=1, seed=17,
                    output_dir=directory / "split",
                )
                saved = json.loads(Path(report["manifest_path"]).read_text(encoding="utf-8"))
                for manifest in (report, saved):
                    self.assertEqual(manifest["dataset"]["particle_order"], list(range(1, N + 1)))
                    self.assertEqual(manifest["dataset"]["ordering_reference"], _REFERENCES.get(N))
                    self.assertEqual(manifest["verification"]["train_test_target_overlap"], 0)
                    self.assertEqual(manifest["verification"]["train_test_input_overlap"], 0)


if __name__ == "__main__":
    unittest.main()

"""Bounded generator fixtures: no production loop, corpus, or model required."""
from collections import Counter, defaultdict
from fractions import Fraction
import gzip
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from . import gravity_v2 as g
from .Tokenizer import ScatteringAmplitudeTokenizer
from .data_gen_gravity.v2_validation import (
    FrozenFamilyGuard, canonical_family, exact_equivalent, parse_laurent,
)


class GeneratorConfigurationTests(unittest.TestCase):
    def test_future_cli_is_train_only_100000_plus_200(self):
        args = g.build_parser().parse_args([
            "--output-dir", "/unused/future-release", "--train-rows", "100000",
            "--test-rows", "200", "--direct-fraction", "3/4",
            "--coefficient-numerator-max", "12", "--coefficient-denominators",
            *(str(n) for n in range(1, 13)), "24",
        ])
        wanted = g.quotas(args.train_rows, args.test_rows)
        self.assertEqual(wanted["train"]["4s1h_replay"], 50000)
        self.assertEqual(wanted["test"]["4s1h_replay"], 100)
        self.assertEqual([wanted["train"][c] for c in g.CATEGORIES[:-1]], [10000]*5)
        self.assertEqual([wanted["test"][c] for c in g.CATEGORIES[:-1]], [20]*5)
        self.assertEqual(args.train_rows + args.test_rows, 100200)

    def test_exact_coefficient_support_and_configuration(self):
        support = g.coefficient_pool()
        for value in (Fraction(1, 6), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3), 2, 3, 4, 6):
            self.assertIn(value, support)
            self.assertIn(-value, support)
        self.assertTrue(all(isinstance(c, Fraction) for c in support))
        self.assertEqual(g.coefficient_pool(1, (2,)), (Fraction(-1, 2), Fraction(1, 2)))
        for maximum, denominators in ((0, (1,)), (12, (0,)), (12, ())):
            with self.assertRaises(ValueError):
                g.coefficient_pool(maximum, denominators)

    def test_configurable_actual_target_schedule(self):
        for ratio in (Fraction(0), Fraction(1, 2), Fraction(3, 4), Fraction(1)):
            stages = Counter(g.stage_for_row(n, ratio) for n in range(20))
            self.assertEqual(stages["direct"], 20 * ratio)
        self.assertEqual([g.stage_for_row(n) for n in range(4)],
                         ["direct", "direct", "direct", "intermediate"])
        with self.assertRaises(ValueError):
            g.stage_for_row(0, Fraction(5, 4))

    def test_bounded_sampling_is_seeded_and_uses_configured_coefficients(self):
        support = g.coefficient_pool(1, (2,))
        for category in g.CATEGORIES:
            kwargs = dict(coefficients=support)
            left = g.sample_origin(category, random.Random(791), 5, **kwargs)
            right = g.sample_origin(category, random.Random(791), 5, **kwargs)
            self.assertEqual(left, right)
            if category not in ("legacy_single_f", "4s1h_replay"):
                self.assertTrue(all(Fraction(c) in support for c in left[3]))

    def test_family_assignment_keeps_descendants_together(self):
        expression = "Tr(F_4 · F_5)^2/(p_1 · p_4)^2"
        descendant = "-7*Tr(F_5 · F_4)^2/(11*(p_3 · p_5)^2)"
        families = [canonical_family(s, "3s2h") for s in (expression, descendant)]
        self.assertEqual(families[0], families[1])
        self.assertEqual(g.family_split(families[0], 730021), g.family_split(families[1], 730021))

    def test_helicity_only_holdout_is_conservatively_reserved(self):
        trace = "Tr(F_4 · F_5)^2/(4*(p_4 · p_5)^2)"
        single_f = ("(p_1 · F_4 · p_5)^2*(p_1 · F_5 · p_4)^2/"
                    "((p_1 · p_4)^2*(p_1 · p_5)^2*(p_4 · p_5)^2)")
        self.assertFalse(exact_equivalent(trace, single_f))
        guard = FrozenFamilyGuard({"3s2h": [trace]}, seed=519113)
        self.assertTrue(guard.is_reserved(single_f, "3s2h"))
        # The implementation must not silently assume the numeric match was
        # general-polarization equivalence.
        values = guard.fingerprint(single_f, "3s2h")
        self.assertFalse(guard._matches_grid(values, guard._matrices["3s2h"]))

    def test_required_holdout_file_cannot_be_silently_omitted(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(g, "ORIGINAL", Path(directory)):
                with self.assertRaisesRegex(FileNotFoundError, "Required holdout reference"):
                    g.freeze_inputs(Path(directory))
            self.assertFalse((Path(directory)/"frozen_inputs.json").exists())


class PublicationFixtureTests(unittest.TestCase):
    """Each fixture contains only two temporary aligned rows, never a release."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.tokenizer = ScatteringAmplitudeTokenizer(max_sequence_length=4096)
        self.simple = "Tr(F_4 · F_5)*Tr(F_4 · F_5)/(24*(p_4 · p_5)*(p_4 · p_5))"
        self.scrambled = g.serialize(g.expand(parse_laurent(self.simple)))
        self.tokens = {c: self.tokenizer.encode_infix(s) for c, s in
                       (("simple", self.simple), ("scrambled", self.scrambled))}
        self.metadata = {key: "" for key in g.METADATA_FIELDS}
        self.metadata.update(split="test", process="3s2h", assigned_category="trace_mixed",
                             stage="direct", simple=self.simple, scrambled=self.scrambled,
                             feature_tags=g.json_text(g.feature_tags(parse_laurent(self.simple), self.scrambled)),
                             coefficients='["1/24"]', contraction_families='["TT"]',
                             pole_multiplicities='[{"(p_4 · p_5)":2}]',
                             transformation_path='["expand_F","full_expansion"]',
                             simple_tokens=len(self.tokens["simple"]),
                             scrambled_tokens=len(self.tokens["scrambled"]))
        self.paths = {f"test_{kind}": Path(self.temp.name)/f"{kind}.csv.gz"
                      for kind in ("raw", "tok", "metadata")}

    def write_fixture(self, *, token_rows=2, corrupt=False):
        for kind in ("raw", "tok", "metadata"):
            fields = g.METADATA_FIELDS if kind == "metadata" else ("simple", "scrambled")
            raw, text, writer = g.gzip_writer(self.paths[f"test_{kind}"], fields)
            try:
                for index in range(token_rows if kind == "tok" else 2):
                    if kind == "metadata":
                        row = {**self.metadata, "row_id": f"test-{index:06d}"}
                    elif kind == "raw":
                        row = dict(simple=self.simple, scrambled=self.scrambled)
                    else:
                        row = {c: json.dumps(t) for c, t in self.tokens.items()}
                        if corrupt and index == 1:
                            row["simple"] = "[10]"
                    writer.writerow(row)
            finally:
                text.close()
                raw.close()

    def test_publication_alignment_and_exact_counts(self):
        self.write_fixture()
        counts = g.verify_publication_alignment(self.paths, {"test": {"trace_mixed": 2}}, self.tokenizer)
        self.assertEqual(counts, {"test": 2})
        with self.assertRaisesRegex(ValueError, "publication_count_mismatch"):
            g.verify_publication_alignment(self.paths, {"test": {"trace_mixed": 3}}, self.tokenizer)

    def test_missing_or_corrupted_parallel_row_is_rejected(self):
        self.write_fixture(token_rows=1)
        with self.assertRaisesRegex(ValueError, "publication_row_alignment"):
            g.verify_publication_alignment(self.paths, {"test": {"trace_mixed": 2}}, self.tokenizer)
        self.write_fixture(corrupt=True)
        with self.assertRaisesRegex(ValueError, "publication_content_alignment"):
            g.verify_publication_alignment(self.paths, {"test": {"trace_mixed": 2}}, self.tokenizer)

    def test_accepted_coverage_counts_overlapping_features_once_per_row(self):
        counters = defaultdict(Counter)
        self.assertFalse(counters)
        g.update_accepted_coverage(counters, self.metadata, self.tokens, self.tokenizer)
        self.assertEqual(counters["assigned_category"], {"trace_mixed": 1})
        for tag in ("coefficient", "trace_or_mixed", "polarization_contraction", "repeated_pole", "combined_features"):
            self.assertEqual(counters["features"][tag], 1)
        self.assertEqual(counters["origin_coefficients"], {"1/24": 1})
        self.assertTrue(counters["simple_digits"])
        self.assertTrue(counters["scrambled_digits"])

    def test_stable_compressed_fixture_bytes(self):
        self.write_fixture()
        before = {key: path.read_bytes() for key, path in self.paths.items()}
        self.write_fixture()
        self.assertEqual(before, {key: path.read_bytes() for key, path in self.paths.items()})


if __name__ == "__main__":
    unittest.main()

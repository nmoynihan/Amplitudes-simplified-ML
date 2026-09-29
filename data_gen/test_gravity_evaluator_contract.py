"""Bounded in-memory generator -> evaluator checks, with no corpus/checkpoint."""
from fractions import Fraction
import csv
import json
import random
import unittest
from unittest import mock

from . import gravity_v2 as g
from .Tokenizer import ScatteringAmplitudeTokenizer
from .data_gen_gravity.v2_validation import FrozenFamilyGuard, check_process
from data_testing import evaluate_model as evaluator
from data_testing.evaluation_integrity import validate_input_integrity
from data_testing.evaluation_records import check_candidate


class GravityEvaluatorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=4096)
        cls.settings = mock.patch.multiple(
            evaluator, NUMERIC_BACKEND="gravity", NUMERIC_EQUIV_SEED=641191,
            NUMERIC_EQUIV_SAMPLES=3, GRAVITY_REFERENCE_MODES=["first", "last", "random"],
            GRAVITY_GAUGE_SHIFT=True, NUMERIC_TOL_ABS=2e-9, NUMERIC_TOL_REL=2e-8,
        )
        cls.settings.start()
        cls.addClassCleanup(cls.settings.stop)
        cls.points = evaluator.precompute_gravity_kinematics()

    def origin(self, family, *, coefficient=Fraction(-7, 12)):
        factors = g.numerator(family, random.Random(41))
        count = 4 if family == "XX" else g.DENOMINATORS[family]
        poles = [g.d(1, 4)] * min(count, 4) + [g.d(2, 5)] * max(count - 4, 0)
        monomial = g.monomial([(a, 1) for a in factors] + [(a, -1) for a in poles])
        return {monomial: coefficient}

    def exercise(self, origin, process, mode="full_expansion"):
        """Use precisely the emitted text and content tokens at each boundary."""
        simple = g.serialize(origin)
        source = g.source_variant(g.expand(origin), mode, random.Random(95191))
        check_process(simple, process)
        check_process(source, process)
        texts = dict(simple=simple, scrambled=source)
        tokens = {column: self.tokenizer.encode_infix(text) for column, text in texts.items()}
        self.assertLess(len(tokens["simple"]), len(tokens["scrambled"]))

        def compare(left, right, selected_process=process):
            return evaluator.numerical_comparison(left, right, self.points,
                                                  gravity_process=selected_process)

        preflight = validate_input_integrity(
            [texts], [tokens], self.tokenizer, [process], compare, reference_provided=True,
        )[0]
        for role in ("source", "reference"):
            self.assertEqual(preflight[role]["status"], "verified")
            self.assertTrue(preflight[role]["exact_check"]["ok"])
            self.assertEqual(preflight[role]["numerical_check"]["status"], "equivalent")
        self.assertEqual(preflight["source"]["original_expression"], source)
        self.assertEqual(preflight["source"]["original_tokens"], tokens["scrambled"])

        pair_result = compare(source, simple)
        self.assertEqual(pair_result["status"], "equivalent", pair_result)
        self.assertEqual(pair_result["helicity_domain"], "same-positive-helicity")
        self.assertEqual(pair_result["counts"]["valid_samples"], 12)
        raw = [2, *tokens["simple"], 3, 0]
        record = check_candidate(
            raw, self.tokenizer, source_expression=source,
            source_tokens=tokens["scrambled"], compare=compare,
            reference_expression=simple, reference_tokens=tokens["simple"],
        )
        self.assertEqual(record["raw_token_ids"], raw)
        self.assertTrue(record["syntax_valid"])
        self.assertTrue(record["completed"])
        self.assertTrue(record["shorter_equivalent"])
        self.assertTrue(record["exact_token"])
        self.assertFalse(record["copies_input"])
        self.assertEqual(record["token_reduction"], len(tokens["scrambled"]) - len(tokens["simple"]))
        json.dumps(dict(preflight=preflight, numerical=pair_result, candidate=record), allow_nan=False)
        return record

    def test_all_six_rational_repeated_pole_templates_pass_actual_evaluator(self):
        for family in g.FAMILIES:
            with self.subTest(family=family):
                self.exercise(self.origin(family), "3s2h")

    def test_4s1h_retains_its_own_degrees_and_evaluator_process(self):
        self.exercise(self.origin("XX", coefficient=Fraction(1, 3)), "4s1h")

    def test_mixed_family_sum_with_independent_coefficients(self):
        origin = g.add(self.origin("TXX", coefficient=Fraction(1, 6)),
                       self.origin("QQ", coefficient=Fraction(-2, 3)))
        self.exercise(origin, "3s2h")

    def test_collected_and_cancelled_styles_preserve_complete_contract(self):
        origin = self.origin("TT", coefficient=Fraction(11, 24))
        for mode in ("term_collection", "common_factor_cancellation", "common_denominator", "partial_fraction"):
            with self.subTest(mode=mode):
                self.exercise(origin, "3s2h", mode=mode)

    def test_supplied_user_and_known_compact_reference_families_are_frozen(self):
        amplitude = g.ORIGINAL / "Test_Amplitude/gravity5unified12345_seed.csv"
        compact = [g.DIAGNOSIS / "physics" / name for name in
                   ("compact_factored.txt", "compact_same_helicity.txt")]
        if not amplitude.is_file() or not all(path.is_file() for path in compact):
            self.skipTest("User diagnostic holdout references are unavailable in this checkout")
        with amplitude.open(newline="") as handle:
            source = next(csv.reader(handle))[1]
        references = [source, *(path.read_text().strip() for path in compact)]
        guard = FrozenFamilyGuard({"3s2h": references}, seed=862139)
        for reference in references:
            with self.subTest(reference=reference[:60]):
                self.assertTrue(guard.is_reserved(reference, "3s2h"))
                self.assertTrue(guard.is_reserved(f"-7*({reference})/12", "3s2h"))


if __name__ == "__main__":
    unittest.main()

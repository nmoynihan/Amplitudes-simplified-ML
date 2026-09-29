"""Focused independent checks on exact v2 construction and transformations."""
from fractions import Fraction
import random
import unittest

from . import gravity_v2 as g
from .Tokenizer import ScatteringAmplitudeTokenizer
from .data_gen_gravity.v2_validation import (check_process, expand_laurent,
                                            parse_laurent, PairValidator)


class GravityV2Tests(unittest.TestCase):
    def test_every_template_has_independently_correct_degree_and_expansion(self):
        for family in g.FAMILIES:
            rng = random.Random(41)
            factors = g.numerator(family, rng)
            poles = [g.d(1, 2)] * min(4, g.DENOMINATORS[family])
            poles += [g.d(2, 3)] * (g.DENOMINATORS[family] - len(poles))
            m = g.monomial([(a, 1) for a in factors] + [(a, -1) for a in poles])
            origin = {m: Fraction(-7, 12)}
            with self.subTest(family=family):
                check_process(g.serialize(origin), "3s2h")
                self.assertEqual(expand_laurent(parse_laurent(g.serialize(origin))), g.expand(origin))

    def test_all_transformations_exact_and_numerical(self):
        validator = PairValidator(seed=8793)
        rng = random.Random(8794)
        process, origin, *_ = g.sample_origin("mixed_family", rng, 5)
        expanded = g.expand(origin)
        for mode in g.TRANSFORMS:
            with self.subTest(mode=mode):
                source = g.source_variant(expanded, mode, rng)
                observed = parse_laurent(source)
                if mode in ("momentum_conservation", "transversality"):
                    self.assertEqual(g.exact_onshell(expanded), g.exact_onshell(observed))
                else:
                    self.assertEqual(expanded, observed)
                result = validator.validate(g.serialize(origin), source, process)
                self.assertTrue(result["ok"], result)

    def test_serialized_streams_exact_with_rationals_and_multidigit_integers(self):
        tok = ScatteringAmplitudeTokenizer(max_sequence_length=4096)
        m = ((("X", 4, 1, 2), 2), (("X", 5, 1, 2), 2), (g.d(1, 4), -2), (g.d(1, 5), -4))
        for c in (Fraction(1, 3), Fraction(1, 6), Fraction(-7, 12), Fraction(11, 24), Fraction(-12), Fraction(2, 3)):
            text = g.serialize({m: c})
            decoded = tok.decode_infix(tok.encode_infix(text))
            self.assertEqual(parse_laurent(text), parse_laurent(decoded))

    def test_requested_quotas_and_no_trivial_target_cancellation(self):
        wanted = g.quotas(100000, 200)
        self.assertEqual(sum(wanted["train"].values()), 100000)
        self.assertEqual(sum(wanted["test"].values()), 200)
        for category in g.CATEGORIES:
            rng = random.Random(398)
            for serial in range(15):
                try:
                    process, origin, _, _, poles = g.sample_origin(category, rng, serial)
                except ValueError:
                    continue
                check_process(origin, process)
                for m, pole_counts in zip(origin, poles):
                    extra_dots = {g.atom_string(a) for a, n in m if a[0] == "d" and n > 0}
                    self.assertFalse(extra_dots & pole_counts.keys())
                    self.assertLessEqual(max(pole_counts.values()), 4)


if __name__ == "__main__":
    unittest.main()

"""Regression tests for fixed cyclic numerator ordering in Yang--Mills pairs."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, permutations
import math
import random
import re
import unittest
from unittest import mock

from ..Tokenizer import ScatteringAmplitudeTokenizer
from . import expr_model as model
from . import generate
from . import yang_mills_cyclic_generation as cyclic
from .algebra import canonicalise_gi_product
from .kinematics import generate_kinematics
from .notation import _RE_pFchainp, _RE_TrN
from .numerics import eval_infix_numeric
from .scramble import _split_signed_terms


def _is_forward_cyclic(labels: tuple[int, ...], N: int) -> bool:
    """Independent oracle: a block is a subsequence of one rotated full cycle."""
    if not labels or len(set(labels)) != len(labels):
        return False
    if any(label < 1 or label > N for label in labels):
        return False
    for start in range(N):
        cycle = tuple(range(start + 1, N + 1)) + tuple(range(1, start + 1))
        # Filtering the cycle tests subsequence membership without replicating
        # the sampler's modular-distance sorting implementation.
        if tuple(label for label in cycle if label in labels) == labels:
            return True
    return False


def _printed_blocks(expression: str) -> list[tuple[int, ...]]:
    blocks = []
    for pattern in (_RE_TrN, _RE_pFchainp):
        blocks.extend(
            tuple(map(int, re.findall(r"F_(\d+)", match.group(0))))
            for match in pattern.finditer(expression)
        )
    return blocks


class PreservedRandomStateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        state = random.getstate()
        self.addCleanup(random.setstate, state)


class CyclicSamplerTests(PreservedRandomStateTestCase):
    def test_exhaustive_subsets_and_input_permutations(self) -> None:
        N = 5
        for arity in range(1, N + 1):
            for subset in combinations(range(1, N + 1), arity):
                observed = set()
                for chosen in permutations(subset):
                    with mock.patch.object(model.random, "sample", return_value=list(chosen)):
                        ordered = tuple(model._sample_cyclic_labels(list(subset), arity, N))
                    self.assertEqual(set(ordered), set(subset))
                    self.assertEqual(ordered[0], chosen[0])
                    self.assertTrue(_is_forward_cyclic(ordered, N), (chosen, ordered))
                    observed.add(ordered)
                expected = {
                    order for order in permutations(subset)
                    if _is_forward_cyclic(order, N)
                }
                self.assertEqual(observed, expected)

    def test_wraparound_and_omitted_labels_are_allowed(self) -> None:
        with mock.patch.object(model.random, "sample", return_value=[4, 2, 5]):
            self.assertEqual(model._sample_cyclic_labels([2, 4, 5], 3, 5), [4, 5, 2])
        self.assertTrue(_is_forward_cyclic((5, 1, 3), 5))
        self.assertTrue(_is_forward_cyclic((4, 5, 1, 2), 5))

    def test_forbidden_permutations_and_reversals_are_not_emitted(self) -> None:
        # (3,1,4) has only one *linear* descent, but crosses the cyclic boundary
        # twice when closed; a one-descent-only check would incorrectly pass it.
        forbidden = ((3, 1, 4), (1, 4, 3), (5, 3, 1), (1, 3, 2, 4))
        for chosen in forbidden:
            with self.subTest(chosen=chosen):
                self.assertFalse(_is_forward_cyclic(chosen, 5))
                with mock.patch.object(model.random, "sample", return_value=list(chosen)):
                    ordered = tuple(model._sample_cyclic_labels(list(chosen), len(chosen), 5))
                self.assertNotEqual(ordered, chosen)
                self.assertTrue(_is_forward_cyclic(ordered, 5))

    def test_one_and_two_label_blocks_have_no_distinct_reverse_orientation(self) -> None:
        for labels in ((3,), (2, 5), (5, 2)):
            with mock.patch.object(model.random, "sample", return_value=list(labels)):
                self.assertEqual(
                    model._sample_cyclic_labels(list(labels), len(labels), 5),
                    list(labels),
                )
            self.assertTrue(_is_forward_cyclic(labels, 5))


class CyclicMonomialTests(PreservedRandomStateTestCase):
    def test_generated_blocks_and_printed_numerators_cover_every_field_once(self) -> None:
        seen_kinds = set()
        saw_multiple_blocks = False
        saw_wraparound = False
        accepted = 0
        for N in (4, 5):
            for seed in range(120):
                random.seed(71000 + 1000 * N + seed)
                try:
                    spec = model._generate_gi_monomial_spec(N, cyclic_order=True)
                except model._AnsatzInfeasible:
                    continue
                accepted += 1
                raw_blocks = [block.photons for block in spec.blocks]
                printed_blocks = _printed_blocks(spec.numerator)
                self.assertEqual(len(raw_blocks), len(printed_blocks))
                for blocks in (raw_blocks, printed_blocks):
                    self.assertEqual(
                        sorted(label for block in blocks for label in block),
                        list(range(1, N + 1)),
                    )
                    self.assertTrue(all(_is_forward_cyclic(block, N) for block in blocks))
                saw_multiple_blocks |= len(spec.blocks) > 1
                seen_kinds.update(block.kind for block in spec.blocks)
                for block in spec.blocks:
                    saw_wraparound |= any(a > b for a, b in zip(block.photons, block.photons[1:]))
                    if block.kind == "chain":
                        self.assertNotEqual(block.left, block.photons[0])
                        self.assertNotEqual(block.right, block.photons[-1])
        self.assertGreater(accepted, 100)
        self.assertEqual(seen_kinds, {"trace", "chain"})
        self.assertTrue(saw_multiple_blocks)
        self.assertTrue(saw_wraparound)

    def test_commuting_block_order_is_independent_of_each_block_order(self) -> None:
        # The two disjoint blocks interleave in the global cycle. Canonical
        # scalar-factor sorting need not concatenate them into a cyclic list.
        numerator = "p_2 · F_5 · F_1 · F_3 · p_2*Tr(F_4 · F_2)"
        canonical = canonicalise_gi_product(numerator)
        self.assertEqual(canonical, "Tr(F_2 · F_4)*p_2 · F_5 · F_1 · F_3 · p_2")
        blocks = _printed_blocks(canonical)
        self.assertTrue(all(_is_forward_cyclic(block, 5) for block in blocks))
        flattened = tuple(label for block in blocks for label in block)
        self.assertEqual(sorted(flattened), list(range(1, 6)))
        self.assertFalse(_is_forward_cyclic(flattened, 5))

    def test_trace_canonicalization_rotates_but_does_not_reverse(self) -> None:
        self.assertEqual(
            canonicalise_gi_product("Tr(F_4 · F_5 · F_1 · F_3)"),
            "Tr(F_1 · F_3 · F_4 · F_5)",
        )
        reversed_trace = canonicalise_gi_product("Tr(F_4 · F_3 · F_1 · F_5)")
        self.assertFalse(_is_forward_cyclic(_printed_blocks(reversed_trace)[0], 5))
        self.assertEqual(reversed_trace, "Tr(F_1 · F_5 · F_4 · F_3)")

    def test_endpoints_may_repeat_or_be_interior_field_labels(self) -> None:
        with mock.patch.object(model.random, "choice", side_effect=[1, 1]) as choose:
            text, block = model._chain_block((5, 1, 3), 5)
        self.assertEqual((block.left, block.right), (1, 1))
        self.assertEqual(text, "p_1 · F_5 · F_1 · F_3 · p_1")
        self.assertEqual(choose.call_args_list[0].args[0], [1, 2, 3, 4])
        self.assertEqual(choose.call_args_list[1].args[0], [1, 2, 4, 5])
        # Endpoint momentum labels are not extra F occurrences and do not
        # participate in the field-strength cyclic-order test.
        self.assertTrue(_is_forward_cyclic(block.photons, 5))

    def test_legacy_monomial_default_matches_explicit_false(self) -> None:
        saw_noncyclic = False
        for seed in range(40):
            random.seed(seed)
            try:
                implicit = model._generate_gi_monomial_spec(5)
            except model._AnsatzInfeasible:
                continue
            random.seed(seed)
            explicit = model._generate_gi_monomial_spec(5, cyclic_order=False)
            self.assertEqual(implicit, explicit)
            saw_noncyclic |= any(not _is_forward_cyclic(block.photons, 5) for block in implicit.blocks)
        self.assertTrue(saw_noncyclic)


class PoleRestrictionTests(PreservedRandomStateTestCase):
    def test_physical_pool_keeps_wraparound_and_excludes_nonadjacent_channels(self) -> None:
        for N in (4, 5, 6):
            expected_pairs = {tuple(sorted((i, i % N + 1))) for i in range(1, N + 1)}
            actual = {tuple(model._dot_legs(pole)) for pole in model._all_physical_poles(N)}
            self.assertEqual(actual, expected_pairs)
            self.assertIn((1, N), actual)
            self.assertNotIn((1, 3), actual)

    def test_nonadjacent_and_repeated_denominators_keep_endpoint_budget(self) -> None:
        # Repeated p_3 endpoints each expose p_1.p_3, a nonphysical channel.
        # This intentionally retains the existing generator's cancellation model.
        spec = model.MonomialSpec(
            numerator="p_3 · F_1 · p_3*Tr(F_2 · F_3 · F_4 · F_5)",
            blocks=(model.BlockSpec("chain", (1,), 3, 3), model.BlockSpec("trace", (2, 3, 4, 5))),
            scalar_pairs=0,
            numerator_mass_dim=7,
        )
        budget = model._chain_endpoint_pole_budget(spec, 5)
        self.assertEqual(budget, {"p_1 · p_3": 2})
        physical = set(model._all_physical_poles(5))
        saw_nonphysical_repeat = False
        for seed in range(30):
            random.seed(seed)
            poles = model._physical_denominator_factors(spec, 5, repeat_probability=1.0)
            self.assertEqual(len(poles), 4)
            counts = Counter(poles)
            for pole, count in counts.items():
                self.assertLessEqual(count, int(pole in physical) + budget.get(pole, 0))
            saw_nonphysical_repeat |= counts["p_1 · p_3"] == 2
        self.assertTrue(saw_nonphysical_repeat)


class CyclicDatasetTests(PreservedRandomStateTestCase):
    def test_four_and_five_point_pairs_both_modes_are_equivalent_and_within_budget(self) -> None:
        tokenizer = ScatteringAmplitudeTokenizer(max_particles=8, max_sequence_length=None)
        for N in (4, 5):
            for mode in ("oneshot", "step"):
                with self.subTest(N=N, mode=mode):
                    pairs = cyclic.build_dataset_batched(
                        N,
                        2,
                        dataset_kind=mode,
                        seed=84100 + N,
                        min_terms=1,
                        max_terms=1,
                        min_scr=1,
                        max_scr=1,
                        scramble_names=("momentum", "ward"),
                        full_expand_scrambled=(mode == "oneshot"),
                        max_tokens=4096,
                        validate=True,
                        validation_pol_modes=("coulomb", "covariant"),
                        jobs=1,
                        batch_size=1,
                        progress=False,
                    )
                    self.assertEqual(len(pairs), 2)
                    for simple, scrambled in pairs:
                        self.assertEqual(model.manifest_mass_dimension(simple), 4 - N)
                        self.assertNotIn("F_", scrambled)
                        for _, term in _split_signed_terms(simple):
                            blocks = _printed_blocks(term)
                            self.assertEqual(
                                sorted(label for block in blocks for label in block),
                                list(range(1, N + 1)),
                            )
                            self.assertTrue(all(_is_forward_cyclic(block, N) for block in blocks))
                        for expression in (simple, scrambled):
                            tokens = tokenizer.encode_infix(expression)
                            self.assertLessEqual(len(tokens), 4096)
                            self.assertNotIn(tokenizer.vocab["<UNK>"], tokens)
                        for pol_mode in ("coulomb", "covariant"):
                            for seed in (95107, 95111):
                                momenta, pols = generate_kinematics(N, pol_mode=pol_mode, seed=seed)
                                expected = eval_infix_numeric(simple, momenta, pols, strict=True)
                                actual = eval_infix_numeric(scrambled, momenta, pols, strict=True)
                                self.assertTrue(math.isfinite(expected) and math.isfinite(actual))
                                self.assertTrue(
                                    math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10),
                                    (N, mode, pol_mode, seed, actual, expected),
                                )

    def test_original_dataset_default_matches_explicit_false(self) -> None:
        options = dict(
            seed=4821,
            min_terms=1,
            max_terms=1,
            min_scr=0,
            max_scr=0,
            scramble_names=("none",),
            validate=False,
            max_tokens=None,
        )
        self.assertEqual(
            generate.build_dataset(4, 1, **options),
            generate.build_dataset(4, 1, cyclic_order=False, **options),
        )

    def test_default_paths_are_distinct_from_original_generator(self) -> None:
        for name in ("DEFAULT_RAW_OUT_TEMPLATE", "DEFAULT_TOK_OUT_TEMPLATE", "DEFAULT_LOG_OUT_TEMPLATE"):
            for N in (4, 5):
                original = getattr(generate, name).format(N=N, NSAMPS=5)
                fixed_order = getattr(cyclic, name).format(N=N, NSAMPS=5)
                self.assertNotEqual(fixed_order, original)
                self.assertIn("cyclic", fixed_order)


if __name__ == "__main__":
    unittest.main()

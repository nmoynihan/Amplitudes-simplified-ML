"""Cyclic policy propagation and shared CLI regression checks."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest import mock

from . import generate
from . import yang_mills_cyclic_generation as cyclic


class CyclicEntrypointTests(unittest.TestCase):
    def test_convenience_builders_enable_cyclic_order(self) -> None:
        for name in ("build_dataset", "build_dataset_batched"):
            with self.subTest(builder=name), mock.patch.object(
                generate, name, return_value=[]
            ) as builder:
                self.assertEqual(getattr(cyclic, name)(5, 3, seed=17), [])
                builder.assert_called_once_with(5, 3, seed=17, cyclic_order=True)

    def test_dataset_policy_reaches_expression_builder(self) -> None:
        for enabled in (False, True):
            with self.subTest(cyclic_order=enabled), mock.patch.object(
                generate, "_build_base_expression", return_value=("1", "1")
            ) as expression_builder, mock.patch.object(
                generate, "scramble", return_value="1"
            ), mock.patch.object(generate, "simplify_to_lowest_terms", return_value="1"):
                pairs = generate.build_dataset(
                    4, 1, validate=False, max_tokens=None, cyclic_order=enabled
                )
                self.assertEqual(pairs, [("1", "1")])
                self.assertIs(expression_builder.call_args.kwargs["cyclic_order"], enabled)

    def test_batched_policy_reaches_every_worker_in_both_modes(self) -> None:
        for enabled in (False, True):
            for mode in ("oneshot", "step"):
                with self.subTest(cyclic_order=enabled, mode=mode), mock.patch.object(
                    generate, "build_dataset", return_value=[]
                ) as builder:
                    generate.build_dataset_batched(
                        5,
                        3,
                        seed=17,
                        batch_size=2,
                        jobs=1,
                        progress=False,
                        dataset_kind=mode,
                        cyclic_order=enabled,
                    )
                    self.assertEqual(builder.call_count, 2)
                    self.assertEqual([call.args for call in builder.call_args_list], [(5, 2), (5, 1)])
                    self.assertEqual(
                        [call.kwargs["seed"] for call in builder.call_args_list],
                        [17, 1000020],
                    )
                    self.assertTrue(
                        all(call.kwargs["cyclic_order"] is enabled for call in builder.call_args_list)
                    )

    def test_shared_cli_preserves_original_defaults_and_separates_cyclic_files(self) -> None:
        cases = (
            (generate.main, False, "gi_5pt_2k.csv", "gi_5pt_tok_2k.csv", "gen_data_5pt_2k.log"),
            (cyclic.main, True, "gi_cyclic_5pt_2k.csv", "gi_cyclic_5pt_tok_2k.csv", "gen_data_cyclic_5pt_2k.log"),
        )
        for entrypoint, enabled, raw_name, tok_name, log_name in cases:
            with self.subTest(cyclic_order=enabled), contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                generate, "build_dataset_batched", return_value=[]
            ) as builder, mock.patch.object(generate, "write_csv") as write_csv, mock.patch.object(
                generate, "tokenise_csv"
            ) as tokenise_csv:
                entrypoint(["5", "--samples", "2000", "--jobs", "1", "--no-progress"])
                self.assertEqual(builder.call_args.args, (5, 2400))
                self.assertEqual(builder.call_args.kwargs["seed"], generate.DEFAULT_SEED)
                self.assertEqual(builder.call_args.kwargs["log_path"], log_name)
                self.assertIs(builder.call_args.kwargs["cyclic_order"], enabled)
                write_csv.assert_called_once_with([], raw_name)
                self.assertEqual(tokenise_csv.call_args.args, (raw_name, tok_name))

    def test_shared_cli_preserves_explicit_outputs_and_options(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            generate, "build_dataset_batched", return_value=[]
        ) as builder, mock.patch.object(generate, "write_csv") as write_csv, mock.patch.object(
            generate, "tokenise_csv"
        ) as tokenise_csv:
            cyclic.main([
                "4", "--samples", "3", "--jobs", "1", "--dataset-kind", "step",
                "--seed", "9", "--raw-out", "custom.csv.gz", "--tok-out", "tokens.csv.gz",
                "--log-out", "custom.log", "--scrambles", "ward", "--grouped-scrambled",
                "--max-tokens", "0", "--no-validate", "--no-tokenise", "--no-progress",
            ])
            options = builder.call_args.kwargs
            self.assertEqual(options["dataset_kind"], "step")
            self.assertEqual(options["seed"], 9)
            self.assertEqual(options["scramble_names"], ["ward"])
            self.assertEqual(options["log_path"], "custom.log")
            self.assertFalse(options["full_expand_scrambled"])
            self.assertFalse(options["validate"])
            self.assertIsNone(options["max_tokens"])
            write_csv.assert_called_once_with([], "custom.csv.gz")
            tokenise_csv.assert_not_called()


if __name__ == "__main__":
    unittest.main()

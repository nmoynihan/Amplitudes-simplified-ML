"""Cyclic policy propagation and canonical/nonzero CLI regression checks."""

from __future__ import annotations

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from . import generate
from . import generate_clean_4pt as clean
from . import yang_mills_cyclic_generation as cyclic


TRACE = "Tr(F_1 · F_2 · F_3 · F_4)"


def _output_args(directory: Path) -> list[str]:
    return [
        "4", "--samples", "3", "--jobs", "1", "--no-progress",
        "--candidate-batch-size", "4", "--zero-checks", "1",
        "--validation-checks", "1",
        "--raw-out", str(directory / "pool.csv"),
        "--tok-out", str(directory / "pool_tok.csv"),
        "--report-out", str(directory / "pool.report.json"),
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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
                        5, 3, seed=17, batch_size=2, jobs=1, progress=False,
                        dataset_kind=mode, cyclic_order=enabled,
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

    def test_original_cli_keeps_legacy_defaults(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            generate, "build_dataset_batched", return_value=[]
        ) as builder, mock.patch.object(generate, "write_csv") as write_csv, mock.patch.object(
            generate, "tokenise_csv"
        ) as tokenise_csv:
            generate.main(["5", "--samples", "2000", "--jobs", "1", "--no-progress"])
            self.assertEqual(builder.call_args.args, (5, 2400))
            self.assertEqual(builder.call_args.kwargs["seed"], generate.DEFAULT_SEED)
            self.assertEqual(builder.call_args.kwargs["log_path"], "gen_data_5pt_2k.log")
            self.assertFalse(builder.call_args.kwargs["cyclic_order"])
            write_csv.assert_called_once_with([], "gi_5pt_2k.csv")
            self.assertEqual(tokenise_csv.call_args.args, ("gi_5pt_2k.csv", "gi_5pt_tok_2k.csv"))

    def test_cyclic_cli_uses_clean_generation_with_distinct_default_outputs(self) -> None:
        stats = clean.GenerationStats(requested=2000, accepted=2000, candidates_generated=2200)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            clean, "generate_to_files", return_value=(stats, {"raw_output": "raw.csv", "token_output": "tok.csv"})
        ) as writer:
            result = cyclic.main(["5", "--samples", "2000", "--jobs", "1", "--no-progress"])
        self.assertEqual(result, 0)
        options = writer.call_args.args[0]
        self.assertEqual(options.samples, 2000)
        self.assertEqual(options.seed, clean.DEFAULT_SEED)
        self.assertEqual(Path(options.raw_out).name, "gi_cyclic_5pt_2k.csv")
        self.assertEqual(Path(options.tok_out).name, "gi_cyclic_5pt_tok_2k.csv")
        self.assertEqual(Path(options.report_out).name, "gi_cyclic_5pt_2k.report.json")
        self.assertEqual(options.progress_every, 0)
        self.assertEqual(writer.call_args.kwargs, {
            "n_particles": 5, "cyclic_order": True, "generator_name": "clean_cyclic_5pt_yang_mills",
        })

    def test_legacy_aliases_preserve_explicit_outputs_and_options(self) -> None:
        stats = clean.GenerationStats(requested=3, accepted=3, candidates_generated=3)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            clean, "generate_to_files", return_value=(stats, {"raw_output": "raw.csv", "token_output": "tok.csv"})
        ) as writer:
            result = cyclic.main([
                "4", "--samples", "3", "--jobs", "1", "--dataset-kind", "step",
                "--seed", "9", "--raw-out", "custom.csv.gz", "--tok-out", "tokens.csv.gz",
                "--log-out", "custom.log", "--scrambles", "ward", "--grouped-scrambled",
                "--max-tokens", "0", "--no-tokenise", "--no-progress",
                "--batch-size", "2", "--mass", "3.0",
            ])
        self.assertEqual(result, 0)
        options = writer.call_args.args[0]
        self.assertEqual(options.seed, 9)
        self.assertEqual(options.scrambles, ["ward"])
        self.assertEqual(options.report_out, "custom.log")
        self.assertEqual(options.raw_out, "custom.csv.gz")
        self.assertEqual(options.tok_out, "tokens.csv.gz")
        self.assertTrue(options.grouped_scrambled)
        self.assertTrue(options.no_tokenise)
        self.assertEqual(options.max_tokens, 0)
        self.assertEqual(options.generator_batch_size, 2)
        self.assertEqual(options.energy_scale, 3.0)

    def test_no_validate_is_rejected_before_generation(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()), mock.patch.object(
            clean, "generate_to_files"
        ) as writer:
            with self.assertRaises(SystemExit) as raised:
                cyclic.main(["4", "--samples", "3", "--no-validate"])
        self.assertEqual(raised.exception.code, 2)
        writer.assert_not_called()

    def test_cleaning_and_deduplication_refill_to_exact_requested_count(self) -> None:
        zero = (f"{TRACE} - {TRACE}",) * 2
        first = (TRACE, f"{TRACE} + 0")
        duplicate_after_cleaning = ("Tr(F_2 · F_3 · F_4 · F_1)", first[1])
        second = (f"2*{TRACE}", f"2*{TRACE} + 0")
        third = (f"3*{TRACE}", f"3*{TRACE} + 0")
        surplus = (f"4*{TRACE}", f"4*{TRACE} + 0")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
                clean, "build_dataset_batched",
                side_effect=[[zero, first, duplicate_after_cleaning, second], [third, surplus]],
            ) as builder:
                result = cyclic.main(_output_args(directory))
            self.assertEqual(result, 0)
            self.assertEqual(builder.call_count, 2)
            self.assertTrue(all(call.kwargs["cyclic_order"] for call in builder.call_args_list))
            self.assertNotEqual(
                builder.call_args_list[0].kwargs["seed"], builder.call_args_list[1].kwargs["seed"],
            )
            raw = _read_csv(directory / "pool.csv")
            tokenized = _read_csv(directory / "pool_tok.csv")
            report = json.loads((directory / "pool.report.json").read_text(encoding="utf-8"))
            self.assertEqual(len(raw), 3)
            self.assertEqual(len(tokenized), 3)
            self.assertEqual(report["stats"]["accepted"], 3)
            self.assertEqual(report["stats"]["exact_zero_targets_rejected"], 1)
            self.assertEqual(report["stats"]["duplicate_rejections"], 1)
            self.assertEqual(report["generator"], "clean_cyclic_4pt_yang_mills")
            tokenizer = clean.ScatteringAmplitudeTokenizer(max_particles=8)
            for source, tokens in zip(raw, tokenized):
                for column in ("simple", "scrambled"):
                    self.assertEqual(
                        source[column], clean.parenthesize_for_semantic_tokenization(source[column]),
                    )
                    self.assertEqual(json.loads(tokens[column]), tokenizer.encode_infix(source[column]))
            self.assertEqual(len({row["simple"] for row in raw}), 3)

    def test_candidate_exhaustion_does_not_publish_an_underfilled_dataset(self) -> None:
        pair = (TRACE, f"{TRACE} + 0")
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr), mock.patch.object(
                clean, "build_dataset_batched", return_value=[pair, pair]
            ):
                result = cyclic.main(_output_args(directory) + [
                    "--samples", "2", "--candidate-batch-size", "2", "--max-candidates-factor", "2",
                ])
            self.assertEqual(result, 1)
            self.assertIn("accepted only 1/2", stderr.getvalue())
            self.assertEqual(list(directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main()

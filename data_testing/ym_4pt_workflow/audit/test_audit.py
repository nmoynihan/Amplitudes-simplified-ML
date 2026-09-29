"""Fast rejection tests; the workflow tests exercise the full historical audit."""

from pathlib import Path
import tempfile
import unittest

from . import parke_taylor, symbolic, vertices
from ._io import AuditFailure


class AuditInputTests(unittest.TestCase):
    def test_parsers_reject_non_arithmetic_syntax(self):
        for parse in (symbolic.parse, vertices.CSVAmplitude):
            for expression in ('__import__("os")', 'foo', '1.25', '[1, 2]',
                               'p1p2 + (p_1 · p_2)'):
                with self.subTest(parser=parse.__name__, expression=expression):
                    with self.assertRaises(ValueError):
                        parse(expression)

    def test_all_audits_reject_wrong_amplitude(self):
        # Constant 1 violates tensor degree, vertex equality, and all-plus zero.
        # This test also runs under python -O: none of these checks use assert.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'wrong.csv'
            source.write_text('1,1\n', encoding='utf-8')
            for module in (symbolic, vertices, parke_taylor):
                with self.subTest(audit=module.__name__):
                    with self.assertRaises(AuditFailure):
                        module.run(source, root / 'output')
            self.assertFalse((root / 'output').exists())

    def test_requires_exactly_one_two_column_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'malformed.csv'
            for text in ('', '1\n', '1,1\n2,2\n', '1,1,1\n'):
                source.write_text(text, encoding='utf-8')
                for module in (symbolic, vertices, parke_taylor):
                    with self.subTest(audit=module.__name__, text=text):
                        with self.assertRaises(AuditFailure):
                            module.run(source, root / 'output')


if __name__ == '__main__':
    unittest.main()

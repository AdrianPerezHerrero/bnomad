"""Portable documentation references and complete CI failure visibility."""
from pathlib import Path
import tempfile
import unittest

from tools.check_documentation import audit, heading_anchors, link_targets
from tools.verify_installed import failure_excerpt


class ReleaseToolTests(unittest.TestCase):
    def test_markdown_links_preserve_nested_urls_without_eating_prose(self):
        text = ('Citation ([paper](https://example.org/paper)), '
                '[nested](https://example.org/a_(b)) and [local](<a b.md>).\n'
                '[ref]: <guide.md#usage>\n')
        self.assertEqual(link_targets(text), ['https://example.org/paper',
            'https://example.org/a_(b)', 'a b.md', 'guide.md#usage'])
        self.assertEqual(heading_anchors('# Hello, `world`!\n## Same\n## Same\n'),
                         {'hello-world', 'same', 'same-1'})
        self.assertEqual(link_targets('Attribution: https://github.com/bnpy/bnpy.'),
                         ['https://github.com/bnpy/bnpy'])

    def test_audit_checks_notebook_anchors_and_ignores_fenced_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'docs').mkdir()
            (root/'tutorials').mkdir()
            (root/'pyproject.toml').write_text('version = "0.3.0"\n')
            (root/'CITATION.cff').write_text('version: 0.3.0\n')
            (root/'README.md').write_text('**Version:** 0.3.0\n[guide](docs/guide.md#usage)\n')
            (root/'docs/guide.md').write_text('# Usage\n[notebook](../tutorials/demo.ipynb#start)\n'
                                             '```md\n[placeholder](missing.md)\n```\n')
            (root/'tutorials/demo.ipynb').write_text(
                '{"cells":[{"cell_type":"markdown","source":["# Start\\n"]}]}')
            self.assertEqual(audit(root)['issues'], [])
            (root/'README.md').write_text('**Version:** 0.3.0\n[bad](docs/guide.md#absent)\n'
                                          '[missing](docs/missing.md)\n')
            self.assertEqual({issue['error'] for issue in audit(root)['issues']},
                             {'missing heading', 'missing path'})

    def test_ci_excerpt_keeps_both_failures_outside_progress_tail(self):
        text = 'FAIL: first\nfirst traceback\n=====\nERROR: second\nsecond traceback\n=====\n'
        text += '\n'.join(f'progress {i}' for i in range(100))
        excerpt = failure_excerpt(text)
        self.assertIn('FAIL: first\nfirst traceback', excerpt)
        self.assertIn('ERROR: second\nsecond traceback', excerpt)
        self.assertIn('progress 99', excerpt)


if __name__ == '__main__':
    unittest.main()

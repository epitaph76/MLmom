"""Execute the agent's actual SQL examples through the pinned CSV worker."""
import re
import tempfile
import unittest
from pathlib import Path

import polars as pl

from mlmom.csv_tools import csv_worker


ROOT = Path(__file__).resolve().parents[1]


class PolarsSQLSkillTests(unittest.TestCase):
    def test_all_skill_examples_execute_and_measure_the_intended_population(self):
        self.assertEqual(pl.__version__, '1.26.0')
        skill = ROOT / 'skills/reconnaissance_agent/polars_sql/SKILL.md'
        queries = re.findall(r'```sql\n(.*?)\n```', skill.read_text(encoding='utf-8'), re.S)
        expected = [
            [dict(category='b', n=3, labeled=3, mean_target=940 / 3,
                  min_target=40., max_target=500.),
             dict(category='a', n=2, labeled=2, mean_target=15.,
                  min_target=10., max_target=20.),
             dict(category=None, n=1, labeled=1, mean_target=30.,
                  min_target=30., max_target=30.)],
            [dict(n=6, nonmissing=4, missing=2, negative=1)],
            [dict(bin='missing', n=2, mean_target=270.),
             dict(bin='negative', n=1, mean_target=10.),
             dict(bin='nonnegative', n=3, mean_target=150.)],
            [dict(unique_keys=4)],
            [dict(duplicate_groups=2, rows_in_duplicate_groups=4, extra_rows=2)],
            [dict(overlapping_complete_keys=1)],
            [dict(source='train', n=6, missing=2), dict(source='test', n=3, missing=1)],
            [dict(n=6, at_least_300=2)],
            [dict(source='train', n=6), dict(source='test', n=3)],
            [dict(overlapping_bins=1)],
            [dict(nonzero=4)],
        ]
        self.assertEqual(len(queries), len(expected), 'Every documented SQL example needs a result check')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pl.DataFrame(dict(category=['a', 'a', 'b', 'b', 'b', None],
                              x=[-1., 1., None, 2., None, 3.],
                              target=[10., 20., 40., 400., 500., 30.],
                              key_a=['A', 'A', 'B', 'B', 'B', None],
                              key_b=[1, 1, 2, 2, 3, 4])).write_csv(root / 'train.csv')
            # A duplicate test key must count once in a distinct overlap; test has no target.
            pl.DataFrame(dict(category=['a', 'c', None], x=[4., None, 5.],
                              key_a=['A', 'A', 'C'], key_b=[1, 1, 9])).write_csv(root / 'test.csv')
            tables = {name: dict(path=str(root / f'{name}.csv'), role=role)
                      for name, role in [('train', 'training'), ('test', 'test_features')]}
            for index, (sql, rows) in enumerate(zip(queries, expected)):
                with self.subTest(sql=sql):
                    output = root / str(index)
                    output.mkdir()
                    result = csv_worker(dict(operation='query', sql=sql, tables=tables,
                                             cache_dir=str(root / 'cache')), output)
                    self.assertEqual(result['status'], 'completed')
                    self.assertFalse(result['truncated'])
                    self.assertEqual(result['rows'], rows)


if __name__ == '__main__':
    unittest.main()

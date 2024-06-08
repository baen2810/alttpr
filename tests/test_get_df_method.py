import sys
import os

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint


def main():
    # Load crawler object with 30 races
    test_name = os.path.split(__file__)[-1].replace('.py', '()')
    pprint(f'----- Starting test \'{test_name}\'', start='\n')
    pprint('Loading test data')
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'data', 'racetime_crawler_30_races.pkl')))
    pprint('Testing', end='...')
    df = gg.get_df()
    assert df.shape[0] == 449, f'Param \'host_ids\', Test 1 failed. {df.shape=}'
    df = gg.get_df(game_filter=False)
    assert df.shape[0] == 509, f'Param \'game_filter\', Test 1 failed. {df.shape=}'
    df = gg.get_df(host_ids='XzVwZWqJmkB5k8eb')
    assert df.shape[0] == 438, f'Param \'host_ids\', Test 2 failed. {df.shape=}'
    df = gg.get_df(host_ids=['jb8GPMWwXbB1nEk0'])
    assert df.shape[0] == 302, f'Param \'host_ids\', Test 3 failed. {df.shape=}'
    df = gg.get_df(drop_forfeits=True)
    assert df.shape[0] == 377, f'Param \'drop_forfeits\', Test 1 failed. {df.shape=}'
    df = gg.get_df(cols=['race_id', 'entrant_id'])
    assert df.shape[0] == 449, f'Param \'cols\', Test 1 failed. {df.shape=}'
    df = gg.get_df(host_rows_only=True)
    assert df.shape[0] == 35, f'Param \'host_rows_only\', Test 1 failed. {df.shape=}'
    df = gg.get_df(cols=['race_id', 'race_start'], unique=True)
    assert df.shape[0] == 21, f'Param \'unique\', Test 1 failed. {df.shape=}'

    pprint('All tests successfully passed.', start='done.\n')

if __name__ == "__main__":
    main()

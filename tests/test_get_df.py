import sys
import os

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint
from tests.data.racetime_crawler_1_page_2_hosts import test_get_df_method_dict as results


def main():
    # Load crawler object with 30 races
    test_name = os.path.split(__file__)[-1].replace('.py', '()')
    pprint(f'----- Starting test \'{test_name}\'', start='\n')
    pprint('Loading test data')
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'data', 'racetime_crawler_1_page_2_hosts.pkl')))
    pprint('Testing', end='...')
    df = gg.get_df()
    assert df.shape[0] == results['T01'], f'T1 failed. {df.shape=}'
    df = gg.get_df(game_filter=False)
    assert df.shape[0] == results['T02'], f'T2 failed. {df.shape=}'
    df = gg.get_df(host_ids='XzVwZWqJmkB5k8eb')
    assert df.shape[0] == results['T03'], f'T3 failed. {df.shape=}'
    df = gg.get_df(host_ids=['jb8GPMWwXbB1nEk0'])
    assert df.shape[0] == results['T04'], f'T4 failed. {df.shape=}'
    df = gg.get_df(drop_forfeits=True)
    assert df.shape[0] == results['T05'], f'T5 failed. {df.shape=}'
    df = gg.get_df(cols=['race_id', 'entrant_id'])
    assert df.shape == results['T06'], f'T6 failed. {df.shape=}'
    df = gg.get_df(host_rows_only=True)
    assert df.shape[0] == results['T07'], f'T7 failed. {df.shape=}'
    df = gg.get_df(host_rows_only=True, host_ids='XzVwZWqJmkB5k8eb', generic_filter=('race_group', 'Community-/Weekly-Race'))
    assert df.shape[0] == results['T08'], f'T8 failed. {df.shape=}'
    df = gg.get_df(windowed={'last_10_days': 10})
    assert df.shape[0] == results['T09'], f'T9 failed. {df.shape=}'
    df = gg.get_df(cols=['race_id', 'race_start'], unique=True)
    assert df.shape[0] == results['T10'], f'T10 failed. {df.shape=}'
    df = gg.get_df(game_filter=False)
    assert df.shape[0] == results['T11'], f'T11 failed. {df.shape=}'
    df = gg.get_df(entrant_has_medal=None)
    assert df.shape[0] == results['T12'], f'T12 failed. {df.shape=}'

    pprint('All tests successfully passed.', start='done.\n')

if __name__ == "__main__":
    main()

import sys
import os

from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint
from pathlib import Path

def main():
    # Load the crawler object
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'tests', 'data', 'racetime_crawler_30_races.pkl')))
    pprint(f'Loaded Crawler host_ids: {gg.host_ids}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    pprint(f'Loaded Crawler host_df.shape: {gg.hosts_df.shape}')  # Output: DataFrame with combined hosts data
    pprint(f'Loaded Crawler len(self.race_ids): {len(gg.race_ids)}')  # Output: DataFrame with combined hosts data
    pprint(f'Loaded Crawler Last updated: {to_tstr(gg.last_updated)}')

    df = gg.get_df()
    assert df.shape == (509, 20), 'Param \'host_ids\', Test 1 failed'
    df = gg.get_df(host_ids='XzVwZWqJmkB5k8eb')
    assert df.shape == (438, 20), 'Param \'host_ids\', Test 2 failed'
    df = gg.get_df(host_ids=['jb8GPMWwXbB1nEk0'])
    assert df.shape == (362, 20), 'Param \'host_ids\', Test 3 failed'
    df = gg.get_df(drop_forfeits=True)
    assert df.shape == (427, 20), 'Param \'drop_forfeits\', Test 1 failed'
    df = gg.get_df(cols=['race_id', 'entrant_id'])
    assert df.shape == (509, 2), 'Param \'cols\', Test 1 failed'
    df = gg.get_df(cols=['race_id', 'entrant_id'])
    assert df.shape == (509, 2), 'Param \'cols\', Test 1 failed'
    df = gg.get_df(host_rows_only=True)
    assert df.shape == (44, 20), 'Param \'host_rows_only\', Test 1 failed'

    pprint('Finished. All tests successfully passed')

if __name__ == "__main__":
    main()

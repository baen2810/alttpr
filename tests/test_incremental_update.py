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
    # Load crawler object with some races
    test_name = os.path.split(__file__)[-1].replace('.py', '()')
    pprint(f'----- Starting test \'{test_name}\'', start='\n')
    loaded_gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'data', 'racetime_crawler_1_page_2_hosts.pkl')))
    loaded_gg.refresh_transforms()
    # loaded_gg.races_df.race_start = loaded_gg.races_df.race_start.dt.tz_localize(None)
    pprint(f'Loaded Crawler host_ids: {loaded_gg.host_ids}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    pprint(f'Loaded Crawler host_df.shape: {loaded_gg.hosts_df.shape}')  # Output: DataFrame with combined hosts data
    pprint(f'Loaded Crawler races_df.shape: {loaded_gg.races_df.shape}') 
    pprint(f'Loaded Crawler len(self.race_ids): {len(loaded_gg.race_ids)}')  # Output: DataFrame with combined hosts data
    pprint(f'Loaded Crawler Last updated: {loaded_gg.last_updated}')

    # loaded_gg.crawl(host_ids=["XzVwZWqJmkB5k8eb", "jb8GPMWwXbB1nEk0"], n_pages=1)
    # pprint(f'Updated Crawler races_df.shape: {loaded_gg.races_df.shape}') 
    # pprint(f'Updated Crawler len(self.race_ids): {len(loaded_gg.race_ids)}')  # Output: DataFrame with combined hosts data
    # pprint(f'Updated Crawler Last updated: {to_tstr(loaded_gg.last_updated)}')
    pprint('Finished.')

if __name__ == "__main__":
    main()

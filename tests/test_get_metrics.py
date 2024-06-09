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
    # Create test data
    # gg = RacetimeCrawler()
    # gg.crawl(host_ids=["XzVwZWqJmkB5k8eb", "jb8GPMWwXbB1nEk0"])
    # pprint(f'{gg.host_ids=}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    # pprint(f'{gg.hosts_df.shape=}')  # Output: DataFrame with combined hosts data
    # pprint(f'{to_tstr(gg.last_updated)=}')
    # # Export the DataFrame to 'export/hosts_df.xlsx'
    # gg.export()
    # # Save the crawler object to 'export/racetime_crawler.pkl'
    # gg.save()

    # Load crawler object
    test_name = os.path.split(__file__)[-1].replace('.py', '()')
    pprint(f'----- Starting test \'{test_name}\'', start='\n')
    pprint('Loading test data')
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'data', 'racetime_crawler_830_races.pkl')))
    # gg.refresh_transforms()
    gg.export()
    pprint('Testing', end='...')
    df = gg.metrics_df
    assert df.shape == (1065, 2), f'Test 1 failed. {df.shape=}'

    pprint('All tests successfully passed.', start='done.\n')

if __name__ == "__main__":
    main()

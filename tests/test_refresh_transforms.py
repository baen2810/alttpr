import sys
import os
import pandas as pd

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint
from tests.data.racetime_crawler_1_page_2_hosts import test_refresh_transforms_dict as results

CRAWLER_NAME = 'racetime_crawler_1_page_2_hosts.pkl'
TEST_DATA_PATH = Path(os.path.join(os.path.dirname(__file__), 'data'))

def main():
    # Load crawler object
    test_name = os.path.split(__file__)[-1].replace('.py', '()')
    pprint(f'----- Starting test \'{test_name}\'', start='\n')
    pprint('----- Preparing test data')
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'data', CRAWLER_NAME)))
    gg.refresh_transforms()
    pprint('----- Testing')
    
    # T01
    assert gg.metrics_df.shape == results['T01'], f'Metrics-Test 1 failed. {gg.metrics_df.shape=}'
    
    # Test facts against their reference
    df_stats = gg.stats_df[['ID', 'Dennsen86']].set_index('ID')
    df_ref = pd.read_excel(Path(TEST_DATA_PATH, results['stats_df_path']))[['ID', 'Dennsen86']].rename(columns={'Dennsen86': 'Dennsen86_REF'})
    df_ref = df_ref[df_ref.ID.isin(results['reference_fact_ids'])].set_index('ID')
    df_stats = df_stats.join(df_ref, how='inner')
    for i, row in df_stats.iterrows():
        assert row.Dennsen86 == row.Dennsen86_REF, f'Facts-Test on Fact-ID {i} failed. {row.Dennsen86=}\n{row.Dennsen86_REF=}'
    pprint('----- All tests successfully passed.')

if __name__ == "__main__":
    main()

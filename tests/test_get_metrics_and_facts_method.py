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
    pprint('----- Preparing test data')
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'data', 'racetime_crawler_test_stats_2024-06-15_REFERENCE.pkl')))
    gg.stats_template_path = Path(os.getcwd(), 'stats_template_alttpr.xlsx')
    gg.game_filter = 'ALttPR'
    gg.lang='DE'
    gg.weekday_dict_EN = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday', '6': 'Sunday'}
    gg.weekday_dict_DE = {'Monday': 'Montag', 'Tuesday': 'Dienstag', 'Wednesday': 'Mittwoch', 'Thursday': 'Donnerstag', 'Friday': 'Freitag', 'Saturday': 'Samstag', 'Sunday': 'Sonntag'}
    gg.refresh_transforms()
    # pprint('----- Exporting test data')
    # gg.output_path = Path(os.getcwd(), 'tests', 'data', 'temp_data')
    # gg.save()
    # gg.export()
    pprint('----- Testing')
    # Test metrics completeness
    assert gg.metrics_df.shape == (26329, 10), f'Metrics-Test 1 failed. {gg.metrics_df.shape=}'
    # Test facts against their reference
    df_stats = gg.stats_df[['ID', 'Dennsen86']].set_index('ID')
    df_ref = pd.read_excel(Path(os.getcwd(), 'tests', 'data', 'stats_df_2024-06-15_REFERENCE.xlsx'))[['ID', 'Dennsen86_REF']].set_index('ID')
    df_stats = df_stats.join(df_ref, how='inner')
    for i, row in df_stats.iterrows():
        assert row.Dennsen86 == row.Dennsen86_REF, f'Facts-Test on Fact-ID {i} failed. {row.Dennsen86=}\n{row.Dennsen86_REF=}'
    pprint('----- All tests successfully passed.')

if __name__ == "__main__":
    main()

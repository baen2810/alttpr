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
    # Load crawler object with 830 races
    test_name = os.path.split(__file__)[-1].replace('.py', '()')
    pprint(f'----- Starting test \'{test_name}\'', start='\n')
    pprint('Loading test data')
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'data', 'racetime_crawler_830_races.pkl')))
    gg.refresh_transforms()
    gg.races_df_cols = [
    'race_id', 'race_goal', 'race_permalink', 'race_info', 'race_state',
    'race_start', 'race_timer', 'race_n_entrants', 'race_info_norm',
    'race_timer_sec', 'is_game', 'mode_boots','race_mode', 'race_mode_simple', 'race_tournament',
    'entrant_id', 'entrant_name', 'entrant_place', 'entrant_rank', 'entrant_finishtime',
    'is_cr', 'race_start_weekday', 'entrant_has_medal', 'entrant_has_won',
    'entrant_has_top10', 'entrant_has_forfeited',
    ]
    pprint('Testing', end='...')
    df = gg.get_metrics()
    assert df.shape == (165, 2), 'Test 1 failed'

    pprint('All tests successfully passed.', start='done.\n')

if __name__ == "__main__":
    main()

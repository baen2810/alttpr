
import os
import sys
import shutil
import pandas as pd

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from alttpr.scanners import DunkaScanner
from alttpr.config_dunka_compact_tests import ITEMTRACKER_POINTS, LIGHTWORLD_MAP_POINTS, DARKWORLD_MAP_POINTS
from alttpr.config_dunka_compact_tests import ITEMTRACKER_BOX, LIGHTWORLD_MAP_BOX, DARKWORLD_MAP_BOX
from alttpr.utils import pprint
from alttpr.utils import to_tstr, pprint


def main():
    test_name = os.path.split(__file__)[-1].replace('.py', '()')
    pprint(f'----- Starting test \'{test_name}\'', start='\n')
    pprint('Loading test data')
    video_path = os.path.join(os.getcwd(), "tests", "data", "alttpr_bow.mp4")
    tmp_output_folder = os.path.join(os.getcwd(), "tests", "data", "_tmp")
    if os.path.exists(tmp_output_folder):
        shutil.rmtree(tmp_output_folder)
    scanner = DunkaScanner(
        input_video_path=video_path,
        output_path=tmp_output_folder,
        start_ts='00:00:00',
        end_ts='00:00:10',
        frames_per_second=1,
        itemtracker_box=ITEMTRACKER_BOX,
        lightworld_map_box=LIGHTWORLD_MAP_BOX,
        darkworld_map_box=DARKWORLD_MAP_BOX,
        itemtracker_points=ITEMTRACKER_POINTS,
        lightworld_map_tracker_points=LIGHTWORLD_MAP_POINTS,
        darkworld_map_tracker_points=DARKWORLD_MAP_POINTS,
    )

    # Run the scan
    scanner.scan()
    pprint('Testing', end='...')
    assert scanner.color_coord_df.shape == (1419, 9), f'ALttPR Scanner, Test 1 failed. {scanner.color_coord_df.shape=}'
    pprint('All tests successfully passed.', start='done.\n')

if __name__ == "__main__":
    main()

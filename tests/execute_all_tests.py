import sys
import os

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint
from test_get_df_method import main as test_get_df_method
from test_incremental_update import main as test_incremental_update


def main():
    test_get_df_method()
    test_incremental_update()

if __name__ == "__main__":
    main()
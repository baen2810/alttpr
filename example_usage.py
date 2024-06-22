import sys
import os
import pandas as pd

from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint
from pathlib import Path

def main():
    # Load the crawler object
    gg = RacetimeCrawler.load(Path(os.path.join(os.path.dirname(__file__), 'tests', 'data', 'racetime_crawler_30_races.pkl')))
    gg.export()

    pprint('Finished. All tests successfully passed')

if __name__ == "__main__":
    main()

import sys
import os

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from pathlib import Path
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint
from execute_crawler_tests import main as execute_crawler_tests
from execute_scanner_tests import main as execute_scanner_tests

def main():
    pprint('-----Executing crawler tests', start='\n')
    execute_crawler_tests()
    pprint('-----Executing scanner tests', start='\n')
    execute_scanner_tests()
    pprint('-----Executing scheduler tests', start='\n')
    pprint('No scheduler tests defined')

if __name__ == "__main__":
    main()

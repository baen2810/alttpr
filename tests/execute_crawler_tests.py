import sys
import os

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from test_get_df_method import main as test_get_df_method
from test_incremental_update import main as test_incremental_update
from test_get_metrics_and_facts_method import main as test_get_metrics_and_facts_method

def main():
    test_incremental_update()
    test_get_df_method()
    test_get_metrics_and_facts_method()

if __name__ == "__main__":
    main()

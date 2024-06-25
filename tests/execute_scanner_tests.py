import sys
import os

# Add the directory containing the alttpr module to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from test_alttpr_scanner_output_df import main as test_alttpr_scanner_output_df

def main():
    test_alttpr_scanner_output_df()

if __name__ == "__main__":
    main()

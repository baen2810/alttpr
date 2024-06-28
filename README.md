# alttpr
A collection of tools to help with analytics on 'A Link to the Past Randomizer' races.

ALttPR is a collection of tools to build game analytics pipelines für 'A Link to the Past Randomizer' games.
ALttPR is pretty cool. If you want to learn more about Randomizers and Romhacking, checkout  (https://alttpr.com/en).
Each playthrough shuffles the location of all the important items in the game making it a unique adventure.

Currently, the main elements of alttpr are
- the `RacetimeCrawler` which is a wrapper that helps crawl race data from racetime.gg
- the `DunkaScanner` which helps to scan ALttPR VODs that use the Dunka Tracker

<!---
This repository provides a library that's distributed by `pip` that you
use for building your own bots.  See the [documentation](https://hubotio.github.io/hubot/docs.html)
for details on getting up and running with your very own robot friend.
--->

# Crawl racetime.gg data using the RacetimeCrawler
# RacetimeCrawler
RacetimeCrawler is a tool that allows users to collect and analyse data from racetime.gg, a website that allows users to host online races for video games.
The tool works in four stages: data collection, data parsing and cleaning, metric creation, and stat creation. In this document, we will explain each stage and show how the tool can help users gain insights into their racing performance and preferences.

*Note*: Currently, the tool only supports crawling of ‘A Link to the Past Randomizer’-Races. Other games are work in progress and may cause issues.

## Data Collection
The tool uses a web crawler to access the racetime.gg website and download the data for each race that the user has participated in or watched. The data includes information such as the race start time and entrant finish times.
Data collection is based on hosts: Based on user specified host IDs all races the user has participated in are identified and subsequently crawled.

## Data Parsing and Cleaning
The tool reads the collected data and converts them into a structured format. The tool also performs some data cleaning operations, such as removing duplicates, fixing errors, and handling missing values.
Attributes such as race group (Tournament, Community-/Weekly-Race) or race mode are generated by parsing the race’s descriptive field specified by the race-creator.

*Note*: User specified text is inherently unstable and may cause issues while parsing. Currently, adapting the crawler to parsing issues requires source code changes. A more flexible approach may be included later.

## Metric Creation
The tool uses the data to create various metrics that measure different aspects of the user's racing performance and preferences. Some examples of metrics are median finish time, win rates, number of racers per game, number of races per category or number of races per weekday. The tool customizes the metrics by choosing the time window, the category, and the level of aggregation. The tool then saves the metrics in a file.

## Stat Creation
The tool uses the metrics and a stats template file to create a report that summarizes and visualizes the user's racing performance. The stats template file is a text file that contains placeholders for the metrics. The user can edit the stats template file to add or edit stats. The tool then generates the stats report in a file.


# How to create your own ALttPR-RacetimeCrawler

Clone the repository. This will create a directory `alttpr` in the current working directory.

Create a Python 3.10 environment using a package manager of your choice, for example conda or venv, and install requirements:

```py
pip install -r requirements.txt
```

Change the working dir to './alttpr:
```cmd
cd ./alttpr
```

Run the script

```py
python ./example_create_save_load_crawler.py
```

or run the code snippet using an IDE of your choice:

```py
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint

# Example usage
gg = RacetimeCrawler()
gg.crawl(host_ids=["XzVwZWqJmkB5k8eb", "jb8GPMWwXbB1nEk0"], n_pages=2)  # crawl 

pprint(f'Crawler host_ids: {gg.host_ids}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
pprint(f'Crawler host_df.shape: {gg.hosts_df.shape}')  # Output: DataFrame with combined hosts data
pprint(f'Crawler Last updated: {to_tstr(gg.last_updated)}')

df = gg.get_df()  # get crawled raw data
pprint(f'Crawler raw data shape: {df.shape}')

# Export DataFrames to './export/'
gg.export()

# Save the crawler object to './export/racetime_crawler.pkl'
gg.save()

# Load the crawler object
loaded_gg = RacetimeCrawler.load("export/racetime_crawler.pkl")
pprint(f'Loaded Crawler host_ids: {loaded_gg.host_ids}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
pprint(f'Loaded Crawler host_df.shape: {loaded_gg.hosts_df.shape}')  # Output: Shape of DataFrame with combined hosts data
pprint(f'Loaded Crawler len(self.race_ids): {len(loaded_gg.race_ids)}')  # Output: Length of list with race ids
pprint(f'Loaded Crawler races_df.shape: {loaded_gg.races_df.shape}')  # Output: Shape of DataFrame with races data
pprint(f'Loaded Crawler Last updated: {to_tstr(loaded_gg.last_updated)}')

pprint('Finished.')
```

# Scan a local VOD of an ALttPR race using the DunkaScanner
If racers use the DunkaTracker, the DunkaScanner can scan the tracker status directly from the VOD. It will create a protocol file containing the tracker status at any given time troughout the VOD.

Simply do

```py
python ./example_create_save_vod_scanner.py
```

It will scan a short example video `tests/data/alttpr_bow.mp4`.

The scanner may require some calibration, it will walk you through five steps:
1) Select race start: Use the keybindings to find the race start frame in the VOD
2) Select race end: Use the keybindings to find the race end frame in the VOD
3) Select item tracker box and sensor points: Use the keybindings to align the item tracker scanner box with the location of the Dunka item tracker in the VOD
4) Select lightworld map tracker box and sensor points: Use the keybindings to align the lightworld map tracker scanner box with the location of the Dunka lightworld map tracker in the VOD
5) Select darkworld map tracker box and sensor points: Use the keybindings to align the darkworld map tracker scanner box with the location of the Dunka darkworld map tracker in the VOD

Once these steps have been completed, the scanner parses the video and produces output artifacts. Check the console output.

# Testing
To check code integrity, simply do

```py
python .\tests\execute_all_tests.py 
```

Various tests will be executed. Check console output for errors.

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

# alttpr
A collection of tools to help with analytics on 'A Link to the Past Randomizer' races.

ALttPR is a collection of tools to build game analytics pipelines für 'A Link to the Past Randomizer' games.
ALttPR pretty cool. If you want to learn more about Randomizers and Romhacking, checkout  (https://alttpr.com/en).
Each playthrough shuffles the location of all the important items in the game making it a unique adventure.

Currently, the main element of alttpr is the `RacetimeCrawler` which is a wrapper that helps crawl race data fr

<!---
This repository provides a library that's distributed by `pip` that you
use for building your own bots.  See the [documentation](https://hubotio.github.io/hubot/docs.html)
for details on getting up and running with your very own robot friend.
--->

# Create your own ALttPR instance

Clone the repository. This will create a directory `alttpr` in the current working directory.

Install requirements:

```py
pip install -r requirements.txt
```

Change the working dir to './alttpr:
```cmd
cd ./alttpr
```

Run the code snippet:

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
or simply do

```py
python ./example_create_save_load_crawler.py
```

## License

See the [LICENSE](LICENSE.md) file for license rights and limitations (MIT).

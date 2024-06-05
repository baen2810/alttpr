from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint

def main():
    # Load crawler object with 30 races
    loaded_gg = RacetimeCrawler.load("tests/racetime_crawler_30_races.pkl")
    pprint(f'Loaded Crawler host_ids: {loaded_gg.host_ids}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    pprint(f'Loaded Crawler host_df.shape: {loaded_gg.hosts_df.shape}')  # Output: DataFrame with combined hosts data
    pprint(f'Loaded Crawler races_df.shape: {loaded_gg.races_df.shape}') 
    pprint(f'Loaded Crawler len(self.race_ids): {len(loaded_gg.race_ids)}')  # Output: DataFrame with combined hosts data
    pprint(f'Loaded Crawler Last updated: {to_tstr(loaded_gg.last_updated)}')

    loaded_gg.get(host_ids=["XzVwZWqJmkB5k8eb", "jb8GPMWwXbB1nEk0"], n_pages=1)
    pprint(f'Updated Crawler races_df.shape: {loaded_gg.races_df.shape}') 
    pprint(f'Updated Crawler len(self.race_ids): {len(loaded_gg.race_ids)}')  # Output: DataFrame with combined hosts data
    pprint(f'Updated Crawler Last updated: {to_tstr(loaded_gg.last_updated)}')

    pprint('Finished.')

if __name__ == "__main__":
    main()

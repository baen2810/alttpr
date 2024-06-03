from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr

def main():
    # Example usage
    gg = RacetimeCrawler()
    gg.get(host_ids=["XzVwZWqJmkB5k8eb", "jb8GPMWwXbB1nEk0"])
    # print('HERE')
    print(f'Crawler host_ids: {gg.host_ids}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    print(f'Crawler host_df.shape: {gg.hosts_df.shape}')  # Output: DataFrame with combined hosts data
    print(f'Crawler Last updated: {to_tstr(gg.last_updated)}')

    # Export the DataFrame to 'export/hosts_df.xlsx'
    gg.export()

    # Save the crawler object to 'export/racetime_crawler.pkl'
    gg.save()

    # Load the crawler object
    loaded_gg = RacetimeCrawler.load("export/racetime_crawler.pkl")
    print(f'Loaded Crawler host_ids: {loaded_gg.host_ids}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    print(f'Loaded Crawler host_df.shape: {loaded_gg.hosts_df.shape}')  # Output: DataFrame with combined hosts data
    print(f'Loaded Crawler len(self.race_ids): {len(loaded_gg.race_ids)}')  # Output: DataFrame with combined hosts data
    print(f'Loaded Crawler Last updated: {to_tstr(loaded_gg.last_updated)}')

if __name__ == "__main__":
    main()

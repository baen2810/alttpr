from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import to_tstr, pprint

def main():
    # Example usage
    gg = RacetimeCrawler()
    gg.crawl(host_ids=[
        "XzVwZWqJmkB5k8eb", "jb8GPMWwXbB1nEk0"
        ],
        n_pages=2,
        )
    # print('HERE')
    pprint(f'{gg.host_ids=}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    pprint(f'{gg.hosts_df.shape=}')  # Output: DataFrame with combined hosts data
    pprint(f'{to_tstr(gg.last_updated)=}')

    # Export the DataFrame to 'export/hosts_df.xlsx'
    gg.export()

    # Save the crawler object to 'export/racetime_crawler.pkl'
    gg.save()

    # Load the crawler object
    loaded_gg = RacetimeCrawler.load("export/racetime_crawler.pkl")
    pprint(f'{loaded_gg.host_ids=}')  # Output: ['XzVwZWqJmkB5k8eb', 'jb8GPMWwXbB1nEk0']
    pprint(f'{loaded_gg.hosts_df.shape=}')  # Output: DataFrame with combined hosts data
    pprint(f'{len(loaded_gg.race_ids)=}')  # Output: DataFrame with combined hosts data
    pprint(f'{loaded_gg.metrics_df.shape=}')  # Output: DataFrame with combined hosts data
    pprint(f'{to_tstr(loaded_gg.last_updated)=}')    
    pprint('Finished.')

if __name__ == "__main__":
    main()

from pathlib import Path
from alttpr.crawlers import VodCrawler
from alttpr.utils import pprint, dotenv2int, dotenv2dict, dotenv2lst, get_workspace_vars, clear_console
from datetime import datetime as dt

from dotenv import load_dotenv
load_dotenv()  # override=True

import os
import pandas as pd

# Clear the console at the beginning of the script
clear_console()

USERNAME = os.getenv("USERNAME")
ALTTPR_DEBUG = bool(os.getenv('ALTTPR_DEBUG_BOOL'))
ALTTPR_VOD_CRAWLER_NAME = "alttpr_vod_crawler"
ALTTPR_VOD_SCANNERS_PATH = "E:/Projekte/alttpr/vod_scanner"
ALTTPR_OUTPUT_FOLDER_PRIVATE = Path("E:/Projekte/alttpr")
ALTTPR_MAX_RACES = None
ALTTPR_CRAWLER_FILE = Path("C:/Users/Weissb/OneDrive/Dokumente/Projekte/alttpr/export/alttpr_crawler/private/racetime_crawler.pkl")

pprint('---Workspace configuration:')
workspace_vars = get_workspace_vars(locals().copy())
print(workspace_vars)

def main():
    vod_crawler_path = Path(ALTTPR_OUTPUT_FOLDER_PRIVATE, 'vod_crawler.pkl')
    if vod_crawler_path.exists():
        vc = VodCrawler.load(vod_crawler_path)
        vc.eval_vod_metadata()
    else:
        vc = VodCrawler(
            gg_path=ALTTPR_CRAWLER_FILE,
            vod_path=ALTTPR_VOD_SCANNERS_PATH,
        )
        vc.init_gg()
        vc.init_vod()

    vc.crawl(
        max_races=ALTTPR_MAX_RACES
        )

    vc.set_output_path(ALTTPR_OUTPUT_FOLDER_PRIVATE)
    vc.save()
    # print('\n')
    # try:
    #     pprint(f'--- Crawler host names: {list(gg.hosts_df.host_name)}')
    #     pprint(f'--- Crawler last updated at: {gg.last_updated}')
    #     pprint(f'--- Crawler # races: {len(gg.race_ids)}')
    # except:
    #     pprint(f'--- Found vanilla crawler')
    #     # n_pages = None
    # gg.set_host_ids(host_ids)
    # gg.set_stats_template_path(ALTTPR_STATS_TEMPLATE_FILE)
    # gg.set_metrics_dicts(
    #     windows_dict=ALTTPR_WINDOWS,
    #     drop_forfeits_dict=ALTTPR_DROP_FORFEITS,
    #     entrant_has_medal_dict=ALTTPR_ENTRANT_HAS_MEDAL,
    # )
    # mailer = MailgunMailer(
    #     domain=MAILGUN_DOMAIN,
    #     api_key=MAILGUN_API_KEY,
    #     sender=MAILGUN_SENDER,
    #     recipients=MAILGUN_RECIPIENTS,
    # )
    # scheduler = DailyScheduler(
    #     name=ALTTPR_CRAWLER_NAME,
    #     runtimes=runtimes,  # ['05:00', '17:00'],  # runtimes,
    #     crawler=gg,
    #     private_folder=ALTTPR_OUTPUT_FOLDER_PRIVATE,
    #     public_folder=ALTTPR_OUTPUT_FOLDER_PUBLIC,
    #     private_dfs=ALTTPR_OUTPUT_DFS_PUBLIC,
    #     n_pages=n_pages,
    #     mailer=mailer,
    #     workspace=workspace_vars.replace(USERNAME, '<user>').replace(MAILGUN_DOMAIN, '<domain>').replace(MAILGUN_API_KEY, '<key>').replace(str(MAILGUN_RECIPIENTS), '<recipients>').replace('<mailgun@<domain>>', '<domain>'),
    #     logs_dir=ALTTPR_LOGS_DIR,
    # )
    # scheduler.run()

if __name__ == "__main__":
    main()

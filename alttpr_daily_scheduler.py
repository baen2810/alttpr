from pathlib import Path
from alttpr.schedulers import DailyScheduler, MailgunMailer
from alttpr.utils import pprint, dotenv2int, dotenv2dict, dotenv2lst, get_workspace_vars, clear_console
from alttpr.crawlers import RacetimeCrawler
from datetime import datetime as dt

from dotenv import load_dotenv
load_dotenv()  # override=True

import os
import pandas as pd

# Clear the console at the beginning of the script
clear_console()

HREF_LOG = os.getenv("HREF_LOG")
USERNAME = os.getenv("USERNAME")
HREF_EXPORT_PRIVATE = os.getenv("HREF_EXPORT_PRIVATE")
MAILGUN_DOMAIN = os.getenv("MAILGUN_DOMAIN")
HREF_EXPORT_PUBLIC = os.getenv("HREF_EXPORT_PUBLIC")
MAILGUN_API_KEY = os.getenv("MAILGUN_API_KEY")
MAILGUN_SENDER = os.getenv("MAILGUN_SENDER")
MAILGUN_RECIPIENTS = dotenv2lst("MAILGUN_RECIPIENTS")
ALTTPR_DEBUG = bool(os.getenv('ALTTPR_DEBUG_BOOL'))
ALTTPR_N_PAGES = dotenv2int('ALTTPR_N_PAGES_INT')
ALTTPR_CRAWLER_NAME = os.getenv('ALTTPR_CRAWLER_NAME_STR')
ALTTPR_LOGS_DIR = os.getenv("ALTTPR_LOGS_DIR")
ALTTPR_OUTPUT_FOLDER_PRIVATE = Path(os.getenv('ALTTPR_OUTPUT_FOLDER_PRIVATE_PATH'))
ALTTPR_OUTPUT_FOLDER_PUBLIC = Path(os.getenv('ALTTPR_OUTPUT_FOLDER_PUBLIC_PATH'))
ALTTPR_CRAWLER_FILE = Path(os.getenv('ALTTPR_CRAWLER_FILE_PATH'))
ALTTPR_STATS_TEMPLATE_FILE = Path(os.getenv('ALTTPR_STATS_TEMPLATE_FILE_PATH'))
ALTTPR_PARSE_TEMPLATE_FILE = Path(os.getenv('ALTTPR_PARSE_TEMPLATE_FILE_PATH'))
ALTTPR_OUTPUT_DFS_PUBLIC = dotenv2lst('ALTTPR_OUTPUT_DFS_PUBLIC_LST')
ALTTPR_RUNTIMES = dotenv2lst('ALTTPR_RUNTIMES_LST')
ALTTPR_HOSTS = dotenv2dict('ALTTPR_HOSTS_DICT')
ALTTPR_WINDOWS = dotenv2dict('ALTTPR_WINDOWS_DICT')
ALTTPR_DROP_FORFEITS = dotenv2dict('ALTTPR_DROP_FORFEITS_DICT')
ALTTPR_ENTRANT_HAS_MEDAL = dotenv2dict('ENTRANT_HAS_MEDAL_DICT')

runtimes = [(dt.now() + pd.Timedelta(seconds=5)).strftime('%H:%M:%S') if x=='' else x for x in ALTTPR_RUNTIMES]
host_ids = list(ALTTPR_HOSTS.keys())
host_names = list(ALTTPR_HOSTS.values())

pprint('---Workspace configuration:')
workspace_vars = get_workspace_vars(locals().copy())
print(workspace_vars)

def main():
    gg = RacetimeCrawler.load(ALTTPR_CRAWLER_FILE)
    n_pages = ALTTPR_N_PAGES
    print('\n')
    try:
        pprint(f'--- Crawler host names: {list(gg.hosts_df.host_name)}')
        pprint(f'--- Crawler last updated at: {gg.last_updated}')
        pprint(f'--- Crawler # races: {len(gg.race_ids)}')
    except:
        pprint(f'--- Found vanilla crawler')
        # n_pages = None
    gg.set_host_ids(host_ids)
    gg.set_stats_template_path(ALTTPR_STATS_TEMPLATE_FILE)
    gg.set_parse_template_path(ALTTPR_PARSE_TEMPLATE_FILE)
    gg.set_metrics_dicts(
        windows_dict=ALTTPR_WINDOWS,
        drop_forfeits_dict=ALTTPR_DROP_FORFEITS,
        entrant_has_medal_dict=ALTTPR_ENTRANT_HAS_MEDAL,
    )
    mailer = MailgunMailer(
        domain=MAILGUN_DOMAIN,
        api_key=MAILGUN_API_KEY,
        sender=MAILGUN_SENDER,
        recipients=MAILGUN_RECIPIENTS,
    )
    scheduler = DailyScheduler(
        name=ALTTPR_CRAWLER_NAME,
        runtimes=runtimes,  # ['05:00', '17:00'],  # runtimes,
        crawler=gg,
        private_folder=ALTTPR_OUTPUT_FOLDER_PRIVATE,
        public_folder=ALTTPR_OUTPUT_FOLDER_PUBLIC,
        private_dfs=ALTTPR_OUTPUT_DFS_PUBLIC,
        n_pages=n_pages,
        mailer=mailer,
        workspace=workspace_vars.replace(USERNAME, '<user>').replace(MAILGUN_DOMAIN, '<domain>').replace(MAILGUN_API_KEY, '<key>').replace(str(MAILGUN_RECIPIENTS), '<recipients>').replace('<mailgun@<domain>>', '<domain>'),
        logs_dir=ALTTPR_LOGS_DIR,
    )
    scheduler.run()

if __name__ == "__main__":
    main()

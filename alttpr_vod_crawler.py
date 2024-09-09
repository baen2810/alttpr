from pathlib import Path
from alttpr.crawlers import VodCrawler
from alttpr.utils import pprint, dotenv2int, dotenv2dict, dotenv2lst, get_workspace_vars, clear_console
from alttpr.config_dunka_compact_sensor_points_2024_Q3_2 import ITEMTRACKER_POINTS, LIGHTWORLD_MAP_POINTS, DARKWORLD_MAP_POINTS
point_reference_lst = list(ITEMTRACKER_POINTS.keys()) + list(LIGHTWORLD_MAP_POINTS.keys()) + list(DARKWORLD_MAP_POINTS.keys())
del ITEMTRACKER_POINTS, LIGHTWORLD_MAP_POINTS, DARKWORLD_MAP_POINTS
# from alttpr.config_dunka_compact_sensor_points_2024_Q3_2 import ITEMTRACKER_BOX, LIGHTWORLD_MAP_BOX, DARKWORLD_MAP_BOX
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
WHITELIST = [
    # "ALTTP Rando #26.02.2023",
    # "ALTTP Rando #02.01.2024",
    # "ALTTP Rando #30.01.2024",
    # "ALTTP Rando #27.02.2024",
    # "ALTTP Rando #25.03.2024",
    # "ALTTP Rando #02.05.2024",
    # "ALTTP Rando #20.05.2024",
    # "ALTTP Rando #17.06.2024",
    # "ALTTP Rando #10.07.2024",
    # "ALTTP Rando #14.07.2024",
    # "ALTTP Rando #22.07.2024",
    # "ALTTP Rando #01.08.2024",
    # "ALTTP Rando #29.01.2024",
    # "ALTTP Rando #10.09.2023",
            ]
BLACKLIST = [
            # 'ALTTP Rando #28.07.2024',
            # 'ALTTP Rando #22.01.2024',
            ]
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
            point_reference=point_reference_lst,
        )
        vc.init_gg()
        vc.init_vod(whitelist=WHITELIST, blacklist=BLACKLIST)
        vc.crawl(max_races=ALTTPR_MAX_RACES)
        vc.eval_vod_metadata()
        vc.set_output_path(ALTTPR_OUTPUT_FOLDER_PRIVATE)
        vc.save()

    df, df_metrics, df_metrics_points = vc.sanity_checks()  # remove_stutter=None
    # dd, mm, yyyy = WHITELIST[-1][-10:].split('.')
    # df_tmp  = df[df.vod_date>='-'.join([yyyy, mm, dd])][[  # '2024-07-29' '2024-08-01' '2024-07-28'
    #     'vod_name', 'time', 'timer', 'time_adj', 'timer_adj',
    #     'point_name', 'point_label', 'point_label_update', 'is_stutter', 'has_change', 'tracker_id'
    #     ]].sort_values(['time', 'tracker_id', 'point_name',])
    # df_tmp.time = [str(t)[-8:] for t in df_tmp.time]
    # df_tmp.time_adj = [str(t)[-8:] for t in df_tmp.time_adj]
    # df_tmp.timer = [str(t)[-8:] for t in df_tmp.timer]
    # df_tmp.timer_adj = [str(t)[-8:] for t in df_tmp.timer_adj]
    # df_m_p_tmp = df_metrics_points[df_metrics_points.vod_name == WHITELIST[-1]]

    pprint('Finished processing')
    # # Checks
    # # always 141 items
    # df[['vod_name', 'point_name']].drop_duplicates().groupby('vod_name').count()
    # df_last = vc.get_df(last_state_only=True)
    # # always 7 for casboots / open races
    # df_last[['vod_name', 'race_mode_simple', 'vod_date']][df_last.point_label_update=='CRYSTAL'].groupby(['vod_name', 'race_mode_simple']).count().reset_index().sort_values('race_mode_simple')
    # # check boss-konsistenz
    # # check last-state konsistenz: titans, mirror, master, crystal-bosses, hookshot


if __name__ == "__main__":
    main()

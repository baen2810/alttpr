# RaceTimeCrawler - a racetime.gg crawler
# Copyright (c) 2023-2024 Benjamin Weiss
#
# Sources on GitHub:
# https://github.com/baen2810/alttpr/

# MIT License

# Copyright (c) 2023-2024 Benjamin Weiss

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Racetime.gg Crawler"""

# from __future__ import annotations
# from collections.abc import Callable, Iterator
# from functools import reduce
# from sys import stderr
from datetime import datetime as dt
from bs4 import BeautifulSoup
from typing import Any
from warnings import warn
from typing import List, Union
from pathlib import Path
from tqdm import trange, tqdm
from alttpr.utils import pprint, pdidx, pprintdesc, get_list, clean_race_info_str, to_tstr, to_dstr

# import base64
# import io
# import re
# import struct
import os
import numpy as np
import pandas as pd
import requests
import pickle
import warnings


DEBUG = True  # bool(os.environ.get('ALTTPR_DEBUG'))  # some of the crawlers can print debug info
pprint('DEBUG mode active') if DEBUG else None
NAN_VALUE = np.nan
if DEBUG:
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None
class RacetimeCrawlerException(Exception):
    """Base class for exceptions."""


class ParseError(RacetimeCrawlerException):
    """Parsing a url failed."""


class UnsupportedFormatError(RacetimeCrawlerException):
    """File format is not supported."""


class RacetimeCrawler:
    def __init__(self) -> None:
        self.game_filter: str = 'ALttPR'
        self.host_ids: List[str] = []
        self.race_ids: List[str] = []
        self.races_df_cols_cr: List[str] = [
            'race_id', 'race_goal', 'race_permalink', 'race_info', 'race_state', 'race_start',
            'race_timer', 'race_n_entrants', 'entrant_place', 'entrant_name', 'entrant_id', 'entrant_finishtime',
        ]  # columns that get crawled
        self.races_df_cols_tf: List[str] = []  # columns that are transformations
        self.hosts_df: pd.DataFrame = pd.DataFrame()
        self.races_df: pd.DataFrame = pd.DataFrame(columns=self.races_df_cols_cr)
        self.stats_df: pd.DataFrame = pd.DataFrame()
        self.metrics_df: pd.DataFrame = pd.DataFrame()
        self.output_path: Path = Path(os.getcwd(), 'export')
        self.stats_template_path: Path = Path(os.getcwd(), 'stats_template_alttpr.xlsx')
        self.base_url: str = r"https://racetime.gg"
        self.last_updated: pd.Timestamp = pd.Timestamp.now()
        self.community_race_thres = 5
        self.weekday_dict_DE = {'0': 'Montag', '1': 'Dienstag', '2': 'Mittwoch', '3': 'Donnerstag', '4': 'Freitag', '5': 'Samstag', '6': 'Sonntag'}

    def get_df(self, host_ids: Union[str, List[str]] = [], drop_forfeits: bool = False, cols: List[str] = [],
               host_rows_only: bool = False, windowed: Union[int, tuple] = None, unique=False, game_filter: bool = True) -> pd.DataFrame:
        host_ids = [host_ids] if isinstance(host_ids, str) else host_ids
        host_ids = list(self.hosts_df.host_id) if len(host_ids) == 0 else host_ids
        cols = self.races_df_cols_cr + self.races_df_cols_tf if len(cols) == 0 else cols
        df = self.races_df[self.races_df.race_id.isin(self.races_df[self.races_df.entrant_id.isin(host_ids)].race_id)]
        df = df[[self.game_filter.lower() in r for r in df.race_id]] if game_filter else df
        df = df.dropna(subset=['entrant_finishtime']) if drop_forfeits else df
        df = df[df.entrant_id.isin(host_ids)] if host_rows_only else df
        if type(windowed) == int:
            df = df[df.race_start >= dt.now() - pd.Timedelta(days=windowed)]
        elif type(windowed) == tuple:
            min_race_date, max_race_date = windowed
            df = df[df.race_start >= min_race_date]
            df = df[df.race_start <= max_race_date]
        df = df[cols]
        df = df.drop_duplicates() if unique else df            
        return df
    
    def refresh_transforms(self) -> None:
        pprint('Refreshing all transforms')
        self._parse_race_info()
        self.get_metrics()
        self.get_stats()
        pprint('All transforms refreshed')
        pprint(f'Number of race ids in races_df ({len(self._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(self.race_ids)})') if len(self._list_ids_in_races_df()) != len(self.race_ids) else None
        pprint(f'Number of columns in races_df ({len(self.races_df.columns)}) does not match number of cols in self.races_df_cols_cr + _tf ({len(self.races_df_cols_cr) + len(self.races_df_cols_tf)})') if len(self.races_df_cols_cr) + len(self.races_df_cols_tf) != len(self.races_df.columns) else None
    
    def get_metrics(self, windowed: Union[int, tuple] = None):
        df = self.get_df(host_rows_only=True, windowed=windowed)
        idx_col = 'entrant_name'
        groupby_col_lst = ['race_category', 'race_group', 'race_start_weekday', 'entrant_place', 'race_mode_simple']
        to_date_cols_lst = ['race_start_', 'entrant_has_won_', 'entrant_has_medal_', 'entrant_has_forfeited_']
        to_time_cols_lst = ['entrant_finishtime_']
        agg_dict = {
            'race_start' : ['min', 'count', 'max'],
            'entrant_finishtime': ['min', 'max', 'median'],
            'entrant_has_won' : ['count', 'min', 'max'],
            'entrant_has_medal' : ['count', 'min', 'max'],
            'entrant_has_forfeited' : ['count', 'min', 'max'],
        }
        values_lst = []
        for k, v_lst in agg_dict.items():
            values_lst += [k + '_' + v for v in v_lst]
        metrics_cols = list(agg_dict.keys())
        df_out = pdidx(df[[idx_col] + metrics_cols].groupby(idx_col).agg(agg_dict))
        for groupby_col in groupby_col_lst:
            df_tmp = pdidx(pdidx(
                df[[idx_col, groupby_col] + metrics_cols].groupby([idx_col, groupby_col]).agg(agg_dict)).reset_index().pivot(
                index=idx_col,
                columns=groupby_col,
                values=values_lst))
            df_tmp.columns = [c.replace(c.split('_')[-1], 'if_' + groupby_col + '_is_' + c.split('_')[-1]) for c in df_tmp.columns]
            df_out = pd.concat([df_out, df_tmp], axis=1)
        # streamline timestamps
        for c in df_out.columns:
            if any([d == c[:len(d)] for d in to_date_cols_lst]) and not(any([d in c for d in ['count']])):
                df_out[c] = [to_dstr(e) if type(e).__name__ != 'NaTType' else e for e in df_out[c]]
            elif any([d == c[:len(d)] for d in to_time_cols_lst]) and not(any([d == c[:len(d)] for d in ['count']])):
                df_out[c] = [to_tstr(e) if type(e).__name__ != 'NaTType' else e for e in df_out[c]]
        # ad special metrics
        df_raw_counts = pdidx(self.get_df(host_rows_only=True, game_filter=False)[['entrant_name', 'race_start']].groupby('entrant_name').count())
        df_raw_counts.columns = ['unfiltered_race_starts']
        df_out = pd.concat([df_out, df_raw_counts], axis=1)
        df_out = df_out.T.rename_axis('metric').sort_values('metric').reset_index()
        pprint(f'Created {df_out.shape[0]} metrics for {df_out.shape[1]-1} racers')
        self.metrics_df = df_out

    def get_stats(self):
        df_stats = pd.read_excel(self.stats_template_path).dropna()
        df_stats.ID = df_stats.ID.astype(int)
        metrics_lst = '<' + self.metrics_df.metric + '>'
        for host_name in list(self.hosts_df.host_name):
            metrics_dict = dict(zip(metrics_lst, self.metrics_df[host_name]))
            df_stats[host_name] = df_stats.Template.str.replace('<host_name>', host_name)
            df_stats[host_name] = df_stats[host_name].str.replace('<game_filter>', self.game_filter)
            for k, v in metrics_dict.items():
                df_stats[host_name] = [f.replace(k, str(v)) for f in df_stats[host_name]]
        self.stats_df = df_stats
    
    def crawl(self, host_ids: Union[str, List[str]], n_pages: int = None) -> None:
        self._get_hosts(host_ids)
        self._get_race_ids(n_pages=n_pages if DEBUG else None)
        self._get_races_data()
        self._parse_race_info()
        self.get_metrics()
        self.get_stats()
        pprint(f'Number of race ids in races_df ({len(self._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(self.race_ids)})') if len(self._list_ids_in_races_df()) != len(self.race_ids) else None
        pprint(f'Number of columns in races_df ({len(self.races_df.columns)}) does not match number of cols in self.races_df_cols_cr + _tf ({len(self.races_df_cols_cr) + len(self.races_df_cols_tf)})') if len(self.races_df_cols_cr) + len(self.races_df_cols_tf) != len(self.races_df.columns) else None
            
    def _get_hosts(self, host_ids: Union[str, List[str]]) -> None:
        self.host_ids = [host_ids] if isinstance(host_ids, str) else host_ids
        all_hosts_data = []
        for host_id in self.host_ids:
            url = self.base_url + '/user/' + host_id
            response = self._scrape(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                n_pages = int(soup.find("div", {"class": "pagination"}).decode_contents().strip().split(' of ')[1].split('\n')[0])
                host_name = soup.find("div", {"class": "user-profile"}).find("span", {"class": "name"}).text
                cols = [v.text.lower().replace(' ', '_') for v in soup.find('aside').find_all('dt')[1:]]
                vals = [int(v.text.split(' ')[0]) for v in soup.find('aside').find_all('dd')[1:]]
                df_user_stats = pd.DataFrame([vals], columns=cols)
                df_user = pd.DataFrame([[host_id, host_name, n_pages]], columns=['host_id', 'host_name', 'n_pages'])
                df_user_stats = pd.concat([df_user, df_user_stats], axis=1)
                all_hosts_data.append(df_user_stats)
            else:
                raise ParseError(f'unable to process host_id \'{host_id}\'')
        self.hosts_df = pd.concat(all_hosts_data, ignore_index=True)
        self.last_updated = pd.Timestamp.now()

    def _get_race_ids(self, n_pages: int = None) -> None:
        pprint('Finding race ids')
        for i, row in self.hosts_df.iterrows():
            host_n_pages = row.n_pages if n_pages is None else n_pages                
            for p in trange(host_n_pages, desc=pprintdesc(f'Host \'{row.host_name}\'')):
                url = self.base_url + '/user/' + row.host_id + f'?page={p+1}'
                response = self._scrape(url)
                if response.status_code == 200:
                    psoup = BeautifulSoup(response.content, "html.parser")
                    page_races_lst = psoup.find("div", {"class": "user-races race-list"}).find_all('ol')[0].find_all('li', recursive=False)
                    # page_races_lst = [r.find("span", {"class": "slug"}).text for r in page_races_lst]
                    page_races_lst = [r.find('a', {"class": "race"})['href'] for r in page_races_lst]
                    self.race_ids = list(sorted(set(self.race_ids + page_races_lst)))
                else:
                    raise ParseError(f'unable to process page \'{url}\'')
        self.last_updated = pd.Timestamp.now()
        pprint(f'Collected {len(self.race_ids)} race ids from {i+1} hosts.')
    
    def _get_races_data(self):
        race_ids_to_crawl = [i for i in self.race_ids if i not in self._list_ids_in_races_df()]  # /alttpr/banzai-redboom-2002
        race_ids_already_crawled = [i for i in self.race_ids if i in self._list_ids_in_races_df()]
        pprint(f'{len(race_ids_already_crawled)} races were already parsed.')
        if len(race_ids_to_crawl) > 0:
            pprint(f'Crawling {len(race_ids_to_crawl)} new races.')
            new_races_df = pd.concat([self._get_race_data(r) for r in tqdm(race_ids_to_crawl, desc=pprintdesc('Crawling races'))], ignore_index=True)
            self.races_df = pd.concat([self.races_df, new_races_df], ignore_index=True).sort_values(['race_start', 'entrant_place'], ascending=[False, True]).reset_index(drop=True)
        else:
            pprint('No new races detected. Crawler is up to date.')
        self.last_updated = pd.Timestamp.now()

    def _list_ids_in_races_df(self):
        return list(sorted(set(self.races_df.race_id)))

    def _get_race_data(self, url: str) -> pd.DataFrame:
        race_url = self.base_url + url
        response = self._scrape(race_url)
        class_options = [
            {"class": "user-pop inline supporter"},
            {"class": "user-pop inline moderator"},
            {"class": "user-pop inline"},
            {"class": "user-pop inline supporter moderator"}
        ]
        if response.status_code == 200:
            rsoup = BeautifulSoup(response.content, "html.parser")
            rsoup_info = rsoup.find('div', {"class": "race-intro"})
            try:
                rr_permalink = rsoup_info.find('a', {"rel": "nofollow"}).text
            except Exception as e:
                rr_permalink = NAN_VALUE
            try:
                rr_info = rsoup_info.find('span', {"class": "info"}).text
            except Exception as e:
                rr_info = NAN_VALUE
            entrants_lst = rsoup.find('div', {"class": "race-entrants"}).find_all('li', {"class": "entrant-row"})
            race_goal = rsoup.find('span', {'class': 'goal'}).text.strip()
            race_state = rsoup.find('div', {'class': 'state'}).find('span', {'class': 'value'}).text
            race_start = pd.to_datetime(rsoup.find('time', {'class': 'timer'})['datetime'])
            race_n_entrants = int(rsoup.find('div', {'class': 'count'}).text.strip().split(' entrants')[0])
            df = pd.DataFrame(columns=self.races_df_cols_cr)
            for e in entrants_lst:
                href_user = None
                for class_option in class_options:
                    try:
                        href_user = e.find('a', class_option)['href']
                        break
                    except (AttributeError, TypeError, KeyError):
                        continue
                if href_user is None:
                    raise ParseError(f"Could not parse href_user for entrant \'{e}\' in race \'{race_url}\'.")
                entrant_finishtime = e.find('time', {"class": "finish-time"}).text
                df = pd.concat([df, pd.DataFrame({
                    'race_id': [url],
                    'race_goal': [race_goal],
                    'race_permalink': [rr_permalink],
                    'race_info': [rr_info],
                    'race_state': [race_state],
                    'race_start': [race_start],
                    'race_timer': [rsoup.find('time', {'class': 'timer'}).text.strip()[:-2]],
                    'race_n_entrants': [race_n_entrants],
                    'entrant_place': [e.find('span', {"class": "place"}).text.strip()],
                    'entrant_name': [e.find('span', {"class": "name"}).text],
                    'entrant_id': [href_user],
                    'entrant_finishtime': [entrant_finishtime],
                })])
            df = df.sort_values(['race_start', 'entrant_place'], ascending=[False, True]).reset_index(drop=True)
            self.last_updated = pd.Timestamp.now()
            return df
        else:
            pprint(f'unable to process race_id \'{race_url}\'')
            self.last_updated = pd.Timestamp.now()

    def _parse_race_info(self):
        # prettify crawled columns
        df = self.races_df[self.races_df_cols_cr]  # remove existing transformations from df
        df.race_goal = df.race_goal.str.replace('\n', '')
        df.race_info = df.race_info.fillna('')
        df.race_state = df.race_state.str.replace('\n', '')
        df.race_start = df.race_start.dt.tz_localize(None)
        df.race_start = [r.replace(microsecond=0) for r in df.race_start]
        df.race_n_entrants = pd.to_numeric(df.race_n_entrants)
        df.entrant_place = [e.strip() if type(e) != float else e for e in df.entrant_place]
        df.entrant_place = df.entrant_place.replace('—', NAN_VALUE)
        df.entrant_id = df.entrant_id.str.replace('/user/', '')
        # df.entrant_finishtime = pd.to_datetime(['1900-01-01 ' + t if t != '—' else NAN_VALUE for t in df.entrant_finishtime])
        df.entrant_finishtime = [pd.Timedelta(t) if t != '—' else NAN_VALUE for t in df.entrant_finishtime]
        # add new transformations
        df['entrant_rank'] = [int(e.replace('th', '').replace('rd', '').replace('nd', '').replace('st', '').replace('—', '10_000')) if type(e) == str else e for e in df.entrant_place]
        df.entrant_rank = df.entrant_rank.replace(10_000, NAN_VALUE)
        df['race_info_norm'] = [clean_race_info_str(txt) for txt in df.race_info]
        df['race_timer_sec'] = [int(r.split(':')[0])*60*60 + int(r.split(':')[1])*60 + int(r.split(':')[2]) for r in df.race_timer]
        df['race_n_forfeits'] = df.set_index('race_id').join(df[df.entrant_place.isna()][['race_id', 'race_goal']].groupby('race_id').count(), rsuffix='_cnt').reset_index().race_goal_cnt.fillna(0).astype(int)
        df['is_game'] = [1 if self.game_filter in r.lower() else 0 for r in df.race_id]
        df['race_start_weekday'] = df.race_start.dt.weekday.astype(str).replace(self.weekday_dict_DE)  # 0=Montag
        df['entrant_has_medal'] = pd.to_datetime([d.date() if r <= 3 else NAN_VALUE for d, r in zip(df.race_start, df.entrant_rank)])
        df['entrant_has_won'] = pd.to_datetime([d.date() if r <= 1 else NAN_VALUE for d, r in zip(df.race_start, df.entrant_rank)])
        df['entrant_has_top10'] = pd.to_datetime([d.date() if r <= 10 else NAN_VALUE for d, r in zip(df.race_start, df.entrant_rank)])
        df['entrant_has_forfeited'] = pd.to_datetime([d.date() if pd.isna(r) else NAN_VALUE for d, r in zip(df.race_start, df.entrant_place)])
        df['entrant_below_2h00m'] = pd.to_datetime([d.date() if r < pd.Timedelta('02:00:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
        df['entrant_below_1h45m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:45:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
        df['entrant_below_1h30m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:30:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
        df['entrant_below_1h15m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:15:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
        df['entrant_below_1h15m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:00:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
        df['mode_open'] = [1 if 'open' in r.lower() else 0 for r in df.race_info_norm]
        df['mode_mcshuffle'] = [1 if 'mcshuffle' in r else 0 for r in df.race_info_norm]
        df['mode_pedestal'] = [1 if 'pedestal' in r else 0 for r in df.race_info_norm]
        df['mode_keysanity'] = [1 if 'keysanity' in r else 0 for r in df.race_info_norm]
        df['mode_enemizer'] = [1 if 'enemizer' in r else 0 for r in df.race_info_norm]
        df['mode_swordless'] = [1 if 'swordless' in r else 0 for r in df.race_info_norm]
        df['mode_inverted'] = [1 if 'inverted' in r else 0 for r in df.race_info_norm]
        df['mode_fastganon'] = [1 if 'fastganon' in r else 0 for r in df.race_info_norm]
        df['mode_boots'] = [1 if 'boot' in r.lower() else 0 for r in df.race_info_norm]
        df['mode_casualboots'] = [1 if 'casualboots' in r else 0 for r in df.race_info_norm]
        df['mode_ambrosia'] = [1 if 'ambrosia' in r else 0 for r in df.race_info_norm]
        df['mode_alldungeons'] = [1 if 'alldungeon' in r or 'ad' in r else 0 for r in df.race_info_norm]
        df['mode_standard'] = [1 if 'standard' in r else 0 for r in df.race_info_norm]
        df['mode_bossshuffle'] = [1 if 'bossshuffle' in r else 0 for r in df.race_info_norm]
        df['mode_champgnhunt'] = [1 if 'championshunt' in r or 'ganonhunt' in r else 0 for r in df.race_info_norm]
        df['mode_bigkeyshuffle'] = [1 if 'bigkeyshuffle' in r or 'bkshuffle' in r.split('(')[0].lower().replace('_', '').replace(' ', '') else 0 for r in df.race_info_norm]
        df['mode_swordless'] = [1 if 'swordless' in r else 0 for r in df.race_info_norm]
        df['mode_crosskeys'] = [1 if 'crosskeys' in r else 0 for r in df.race_info_norm]
        df['mode_mystery'] = [1 if 'mystery' in r else 0 for r in df.race_info_norm]
        df['mode_friendly'] = [1 if 'friendly' in r else 0 for r in df.race_info_norm]
        df['mode_weighted'] = [1 if 'weighted' in r else 0 for r in df.race_info_norm]
        df['mode_coop'] = [1 if 'co-op' in r else 0 for r in df.race_info_norm]
        df_sum_mode_flags = df[[c for c in df.columns if 'mode' == c[:4]]]
        df['sum_modes'] = df_sum_mode_flags.sum(axis=1)
        # Mode
        mode_lst = []
        for i, row in df.iterrows():
            mode = ''
            if row.mode_standard == 1:
                mode += 'Standard '
            if row.mode_casualboots == 1:
                mode += 'Casual '
            if row.mode_swordless == 1:
                mode += 'Swordless (Hard) '
            if row.mode_alldungeons == 1:
                mode += 'AD '
            if row.mode_open == 1:
                mode += 'Open '
            if row.mode_enemizer == 1:
                mode += 'Enemizer (Assured) '
            if row.mode_inverted == 1:
                mode += 'Inverted '
            if row.mode_boots == 1:
                mode += 'Boots '
            if row.mode_fastganon == 1:
                mode += 'Fast Ganon '
            if row.mode_ambrosia == 1:
                mode += 'Ambrosia '
            if row.mode_bossshuffle == 1:
                mode += 'Boss Shuffle '
            if row.mode_bigkeyshuffle == 1:
                mode += 'Bigkey Shuffle '
            if row.mode_champgnhunt == 1:
                mode += 'Champions (Ganon) Hunt '
            if row.mode_mcshuffle == 1:
                mode += 'MC Shuffle '
            if row.mode_pedestal == 1:
                mode += 'Pedestal '
            if row.mode_keysanity == 1:
                mode += 'Keysanity '
            if row.mode_crosskeys == 1:
                mode += 'Crosskeys '
            if row.mode_mystery == 1:
                mode += 'Mystery '
            if row.mode_weighted == 1:
                mode += 'Weighted '
            if row.mode_friendly == 1:
                mode += 'Friendly '
            if row.mode_coop == 1:
                mode += 'Co-op '
            if mode == '':
                mode = 'Unknown'
            mode = mode.replace('Casual Open Boots', 'Open Boots')
            mode = 'Casual Boots' if mode == 'Casual' else mode
            mode_lst += [mode.strip()]
        df['race_mode'] = mode_lst
        # Modus (vereinfacht)
        mode_simple_lst = []
        for m in df.race_mode:
            mode_simple = m
            if m == 'Casual Boots' or m == 'Open Boots' or m == 'Standard Boots' or m == 'Casual':
                mode_simple = 'Casual/Open Boots'
            if 'AD' in m:
                mode_simple = 'All Dungeons'
            if 'AD Boots' in m:
                mode_simple = 'AD Boots'
            if m in ['Open Fast Ganon', 'Standard Fast Ganon']:
                mode_simple = 'Fast Ganon'
            if 'Champions (Ganon) Hunt' == m:
                mode_simple = 'Champions Hunt'
            if 'Keysanity' in m:
                mode_simple = 'Keysanity'
            if 'Mystery' in m:
                mode_simple = 'Mystery'
            mode_simple = mode_simple.replace(' (Assured)', ''
            ).replace('Standard Swordless', 'Swordless'
            ).replace(' (Hard)', '')
            mode_simple_lst += [mode_simple]
        df['race_mode_simple'] = mode_simple_lst
        # Tournament
        tournament_lst, r_lst = [], []
        for r in df.race_info_norm:
            if 'rivalcup' in r:
                tournament = 'Rival Cup'
            elif 'deutschesweekly' in r or 'sgdeweeklyrace' in r or 'sgdeweekly' in r:
                tournament = 'German Weekly'        
            elif 'deutschesalttprturnier' in r:
                tournament = 'Deutsches ALttPR Turnier'
            elif 'alttprtournament' in r or 'alttprmaintournament' in r or'alttpr2021tournament' in r:
                tournament = 'ALttPR Tournament'
            elif 'coopduality' in r:
                tournament = 'Co-op Duality'
            elif 'alttprleague' in r:
                tournament = 'ALttPR League'
            elif 'sgdeminiturnier' in r.replace('-', '') or 'deutschesminiturnier' in r.replace('-', ''):
                tournament = 'SGDE Miniturnier'
            elif 'speedgamingdailyraceseries' in r:
                tournament = 'SpeedGaming Daily Race Series'
            elif 'crosskeystourn' in r or 'crosskeys202' in r or 'crosskeysswiss' in r:
                tournament = 'Crosskeys Tournament'
            elif 'enemizertournament' in r:
                tournament = 'Enemizer Tournament'
            elif 'alttprswordless' in r:
                tournament = 'ALttPR Swordless'
            elif 'communityrace' in r:
                tournament = 'Community Race'
            elif 'qualifier' in r:
                tournament = 'Qualifier'
            else:
                tournament = NAN_VALUE
            # print(rd, tournament)
            r_lst += [r]
            tournament_lst += [tournament]
        # df['rr_info_cleansed'] = r_lst
        df['race_tournament'] = tournament_lst
        df.race_tournament = ['Community Race' if e >= self.community_race_thres and type(t) == float else t for e,t in zip(df.race_n_entrants, df.race_tournament)]
        df['race_category'] = [r if r in ['German Weekly', 'Community Race'] else 'Tournament' for r in df.race_tournament]
        df['race_group'] = ['Community-/Weekly-Race' if r in ['German Weekly', 'Community Race'] else 'Tournament' for r in df.race_tournament]
        # df_races = df_races_tmp.set_index('race_id').join(df[['race_id', 'rr_permalink', 'rr_info', 'race_tournament', 'race_mode', 'race_mode_simple', 'mode_boots']].set_index('race_id')).reset_index()
        df = df.sort_values(['race_start', 'entrant_place'], ascending=[False, True]).reset_index(drop=True)
        self.races_df_cols_tf = [c for c in df.columns if c not in self.races_df_cols_cr]
        self.races_df = df
        self.last_updated = pd.Timestamp.now()

    def add_races(self, tbd):
        '''Add one or more races and pull data'''
        pass

    def update(self, tbd):
        '''Update populated crawler by adding new races'''
        pass

    def _scrape(self, url: str) -> requests.Response:
        return requests.get(url)
    
    def set_output_path(self, path: Union[Path, str]) -> None:
        self.output_path = Path(path)

    def export(self, path: Union[Path, str] = None, dfs: List[str] = ['hosts_df', 'races_df', 'metrics_df', 'stats_df']) -> None:
        if path:
            self.set_output_path(path)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        pprint(f'Exporting data to: {self.output_path}', end='...')
        self.hosts_df.to_excel(Path(self.output_path, 'hosts_df.xlsx'), index=False, engine='openpyxl') if 'hosts_df' in dfs else None
        self.get_df().to_excel(Path(self.output_path, 'races_df.xlsx'), index=False, engine='openpyxl') if 'races_df' in dfs else None
        df_metrics = self.metrics_df.astype(str)
        for c in df_metrics.columns:
            df_metrics[c] = [d.replace('NaT', '').replace('nan', '').replace('0 days ', '') for d in df_metrics[c]]
        df_metrics.to_excel(Path(self.output_path, 'metrics_df.xlsx'), index=False, engine='openpyxl') if 'metrics_df' in dfs else None
        df_stats = self.stats_df.astype(str)
        df_stats.to_excel(Path(self.output_path, 'stats_df.xlsx'), index=False, engine='openpyxl') if 'stats_df' in dfs else None
        print('done.')

    def save(self) -> None:
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        save_path = Path(self.output_path, 'racetime_crawler.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        pprint(f'Crawler object saved to: {save_path}')
    
    @staticmethod
    def load(path: Union[Path, str]) -> 'RacetimeCrawler':
        pprint(f'Loading Crawler from: {path}', end='...')
        with open(path, 'rb') as f:
            crawler = pickle.load(f)
        print('done.')
        pprint(f'Number of race ids in races_df ({len(crawler._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(crawler.race_ids)})') if len(crawler._list_ids_in_races_df()) != len(crawler.race_ids) else None
        pprint(f'Number of columns in races_df ({len(crawler.races_df.columns)}) does not match number of cols in crawler.races_df_cols_cr + _tf ({len(crawler.races_df_cols_cr) + len(crawler.races_df_cols_tf)})') if len(crawler.races_df_cols_cr) + len(crawler.races_df_cols_tf) != len(crawler.races_df.columns) else None
        return crawler
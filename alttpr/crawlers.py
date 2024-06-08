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
from alttpr.utils import pprint, pdidx, pprintdesc, get_list, clean_race_info_str

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
class RacetimeCrawlerException(Exception):
    """Base class for exceptions."""


class ParseError(RacetimeCrawlerException):
    """Parsing a url failed."""


class UnsupportedFormatError(RacetimeCrawlerException):
    """File format is not supported."""


class RacetimeCrawler:
    def __init__(self) -> None:
        self.game_filter: str = 'alttpr'
        self.host_ids: List[str] = []
        self.race_ids: List[str] = []
        self.hosts_df: pd.DataFrame = pd.DataFrame()
        self.races_df_cols = [
            'race_id', 'race_goal', 'race_permalink', 'race_info', 'race_state',
            'race_start', 'race_timer', 'race_n_entrants', 'race_info_norm',
            'race_timer_sec', 'is_game', 'mode_boots','race_mode', 'race_mode_simple', 'race_tournament',
            'entrant_id', 'entrant_name', 'entrant_place', 'entrant_rank', 'entrant_finishtime',
            'is_cr', 'race_start_weekday', 'entrant_has_medal', 'entrant_has_won',
            'entrant_has_top10', 'entrant_has_forfeited',
            ]
        self.races_df: pd.DataFrame = pd.DataFrame(columns=self.races_df_cols)
        self.output_path: Path = Path(os.getcwd(), 'export')
        self.base_url: str = r"https://racetime.gg"
        self.last_updated: pd.Timestamp = pd.Timestamp.now()
        self.community_race_thres = 5
        self.weekday_dict_DE = {'0': 'Montag', '1': 'Dienstag', '2': 'Mittwoch', '3': 'Donnerstag', '4': 'Freitag', '5': 'Samstag', '6': 'Sonntag'}

    def get_df(self, host_ids: Union[str, List[str]] = [], drop_forfeits: bool = False, cols: List[str] = [],
               host_rows_only: bool = False, windowed: Union[int, tuple] = None, unique=False, game_filter: bool = True) -> pd.DataFrame:
        host_ids = [host_ids] if isinstance(host_ids, str) else host_ids
        host_ids = list(self.hosts_df.host_id) if len(host_ids) == 0 else host_ids
        cols = list(self.races_df_cols) if len(cols) == 0 else cols
        df = self.races_df[self.races_df.race_id.isin(self.races_df[self.races_df.entrant_id.isin(host_ids)].race_id)]
        df = df[[self.game_filter in r for r in df.race_id]] if game_filter else df
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
        pprint('Refreshing all transforms', end='...')
        self.races_df = self._parse_race_info(self.races_df)
        print('done.')
    
    def get_metrics(self, windowed: Union[int, tuple] = None) -> pd.DataFrame:
        df = self.get_df(host_rows_only=True, windowed=windowed)
        idx_col = 'entrant_name'
        groupby_col_lst = ['is_cr', 'race_start_weekday']
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
            df_out = pd.concat([df_out, df_tmp], axis=1)
        df_out = df_out.T.rename_axis('metric').sort_values('metric')
        return df_out

    def get_facts():
        pass
    
    def crawl(self, host_ids: Union[str, List[str]], n_pages: int = None) -> None:
        self._get_hosts(host_ids)
        self._get_race_ids(n_pages=n_pages if DEBUG else None)
        self._get_races_data()
        pprint(f'Number of race ids in races_df ({len(self._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(self.race_ids)})') if len(self._list_ids_in_races_df()) != len(self.race_ids) else None
        pprint(f'Number of columns in races_df ({len(self.races_df.columns)}) does not match number of cols in self.races_df_cols ({len(self.races_df_cols)})') if len(self.races_df_cols) != len(self.races_df.columns) else None
            
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
            pprint(f'Parsing {len(race_ids_to_crawl)} new races.')
            new_races_df = pd.concat([self._get_race_data(r) for r in tqdm(race_ids_to_crawl, desc='Parsing races')], ignore_index=True)
            self.races_df = pd.concat([self.races_df, new_races_df], ignore_index=True).sort_values(['race_start', 'entrant_place'], ascending=[False, True])
        else:
            pprint('No new races detected. Crawler is up to date.')
        self.last_updated = pd.Timestamp.now()

    def _list_ids_in_races_df(self):
        return list(sorted(set(self.races_df.race_id)))

    def _get_race_data(self, url: str) -> pd.DataFrame:
        race_url = self.base_url + url
        response = self._scrape(race_url)
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
            df_races = pd.DataFrame()
            for e in entrants_lst:
                href_user = NAN_VALUE
                try:
                    href_user = e.find('a', {"class": "user-pop inline supporter"})['href']
                except:
                    try:
                        href_user = e.find('a', {"class": "user-pop inline moderator"})['href']
                    except:
                        try:
                            href_user = e.find('a', {"class": "user-pop inline"})['href']
                        except:
                            href_user = e.find('a', {"class": "user-pop inline supporter moderator"})['href']
                entrant_rank = int(e.find('span', {"class": "place"}).text.strip().replace('th', '').replace('rd', '').replace('nd', '').replace('st', '').replace('—', '10_000'))
                entrant_finishtime = e.find('time', {"class": "finish-time"}).text
                entrant_finishtime = NAN_VALUE if entrant_finishtime=='None' else entrant_finishtime
                df_races = pd.concat([df_races, pd.DataFrame({
                    'race_id': [url],
                    'race_goal': [race_goal],
                    'race_permalink': [rr_permalink],
                    'race_info': [rr_info],
                    'race_goal': [race_goal],
                    'race_state': [race_state],
                    'race_start': [race_start],
                    'race_timer': [rsoup.find('time', {'class': 'timer'}).text.strip()[:-2]],
                    'race_n_entrants': [race_n_entrants],
                    'entrant_place': [e.find('span', {"class": "place"}).text.strip()],
                    'entrant_rank': [entrant_rank],
                    'entrant_name': [e.find('span', {"class": "name"}).text],
                    'entrant_id': [href_user],
                    'entrant_finishtime': [entrant_finishtime],
                })], ignore_index=True)
            df_races.race_start = df_races.race_start.dt.tz_localize(None)
            df_races.race_n_entrants = pd.to_numeric(df_races.race_n_entrants)
            df_races.race_goal = df_races.race_goal.str.replace('\n', '')
            df_races.race_state = df_races.race_state.str.replace('\n', '')
            df_races.entrant_id = df_races.entrant_id.str.replace('/user/', '')
            df_races.entrant_finishtime = pd.to_datetime(df_races.entrant_finishtime.replace('—', NAN_VALUE))
            df_races.entrant_rank = df_races.entrant_rank.replace(10_000, NAN_VALUE)
            df_races.race_info = df_races.race_info.fillna('')
            df_races['race_info_norm'] = [clean_race_info_str(txt) for txt in df_races.race_info]
            df_races['race_timer_sec'] = [int(r.split(':')[0])*60*60 + int(r.split(':')[1])*60 + int(r.split(':')[2]) for r in df_races.race_timer]
            df_races.sort_values(['entrant_rank'], ascending=True).reset_index(drop=True)
            df_races['is_game'] = [1 if self.game_filter in r.lower() else 0 for r in df_races.race_id]
            df_races = self._parse_race_info(df_races)
            df_races = df_races[self.races_df_cols]
            return df_races
        else:
            pprint(f'unable to process race_id \'{race_url}\'')
        self.last_updated = pd.Timestamp.now()

    def _parse_race_info(self, df_rr_details: pd.DataFrame):
        df_rr_details['is_cr'] = [r if r in ['Deutsches Weekly', 'Community Race'] else 'Turnier' for r in df_rr_details.race_tournament]
        df_rr_details['race_start_weekday'] = df_rr_details.race_start.dt.weekday.astype(str).replace(self.weekday_dict_DE)  # 0=Montag
        df_rr_details['entrant_has_medal'] = [d if r <= 3 else NAN_VALUE for d, r in zip(df_rr_details.race_start, df_rr_details.entrant_rank)]
        df_rr_details['entrant_has_won'] = [d if r <= 1 else NAN_VALUE for d, r in zip(df_rr_details.race_start, df_rr_details.entrant_rank)]
        df_rr_details['entrant_has_top10'] = [d if r <= 10 else NAN_VALUE for d, r in zip(df_rr_details.race_start, df_rr_details.entrant_rank)]
        df_rr_details['entrant_has_forfeited'] = [d if r == '—' else NAN_VALUE for d, r in zip(df_rr_details.race_start, df_rr_details.entrant_place)]
        df_rr_details['mode_open'] = [1 if 'open' in r.lower() else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_mcshuffle'] = [1 if 'mcshuffle' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_pedestal'] = [1 if 'pedestal' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_keysanity'] = [1 if 'keysanity' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_enemizer'] = [1 if 'enemizer' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_swordless'] = [1 if 'swordless' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_inverted'] = [1 if 'inverted' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_fastganon'] = [1 if 'fastganon' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_boots'] = [1 if 'boot' in r.lower() else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_casualboots'] = [1 if 'casualboots' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_ambrosia'] = [1 if 'ambrosia' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_alldungeons'] = [1 if 'alldungeon' in r or 'ad' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_standard'] = [1 if 'standard' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_bossshuffle'] = [1 if 'bossshuffle' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_champgnhunt'] = [1 if 'championshunt' in r or 'ganonhunt' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_bigkeyshuffle'] = [1 if 'bigkeyshuffle' in r or 'bkshuffle' in r.split('(')[0].lower().replace('_', '').replace(' ', '') else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_swordless'] = [1 if 'swordless' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_crosskeys'] = [1 if 'crosskeys' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_mystery'] = [1 if 'mystery' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_friendly'] = [1 if 'friendly' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_weighted'] = [1 if 'weighted' in r else 0 for r in df_rr_details.race_info_norm]
        df_rr_details['mode_coop'] = [1 if 'co-op' in r else 0 for r in df_rr_details.race_info_norm]
        df_sum_mode_flags = df_rr_details[[c for c in df_rr_details.columns if 'mode' == c[:4]]]
        df_rr_details['sum_modes'] = df_sum_mode_flags.sum(axis=1)
        # Mode
        mode_lst = []
        for i, row in df_rr_details.iterrows():
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
        df_rr_details['race_mode'] = mode_lst
        # Modus (vereinfacht)
        mode_simple_lst = []
        for m in df_rr_details.race_mode:
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
        df_rr_details['race_mode_simple'] = mode_simple_lst
        # Tournament
        tournament_lst, r_lst = [], []
        for r in df_rr_details.race_info_norm:
            if 'rivalcup' in r:
                tournament = 'Rival Cup'
            elif 'deutschesweekly' in r:  # Edge Cases; don't use rd
                tournament = 'Deutsches Weekly'
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
            elif 'sgdeweeklyrace' in r or 'sgdeweekly' in r:
                tournament = 'Deutsches Weekly'        
            elif 'speedgamingdailyraceseries' in r:  # Edge Cases; don't use rd
                tournament = 'SpeedGaming Daily Race Series'
            elif 'crosskeystourn' in r or 'crosskeys202' in r or 'crosskeysswiss' in r:  # Edge Cases; don't use rd
                tournament = 'Crosskeys Tournament'
            elif 'enemizertournament' in r:  # Edge Cases; don't use rd
                tournament = 'Enemizer Tournament'
            elif 'alttprswordless' in r:  # Edge Cases; don't use rd
                tournament = 'ALttPR Swordless'
            elif 'communityrace' in r:  # Edge Cases; don't use rd
                tournament = 'Community Race'
            elif 'qualifier' in r:  # Edge Cases; don't use rd
                tournament = 'Qualifier'
            else:
                tournament = NAN_VALUE
            # print(rd, tournament)
            r_lst += [r]
            tournament_lst += [tournament]
        # df_rr_details['rr_info_cleansed'] = r_lst
        df_rr_details['race_tournament'] = tournament_lst
        # df_races = df_races_tmp.set_index('race_id').join(df_rr_details[['race_id', 'rr_permalink', 'rr_info', 'race_tournament', 'race_mode', 'race_mode_simple', 'mode_boots']].set_index('race_id')).reset_index()
        df_rr_details.race_tournament = ['Community Race' if e >= self.community_race_thres and type(t) == float else t for e,t in zip(df_rr_details.race_n_entrants, df_rr_details.race_tournament)]
        self.last_updated = pd.Timestamp.now()
        return df_rr_details

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

    def export(self, path: Union[Path, str] = None) -> None:
        if path:
            self.set_output_path(path)
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        self.hosts_df.to_excel(Path(self.output_path, 'hosts_df.xlsx'), index=False, engine='openpyxl')
        self.get_df().to_excel(Path(self.output_path, 'races_df.xlsx'), index=False, engine='openpyxl')
        print(f'Exported data to: {self.output_path}')

    def save(self) -> None:
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        save_path = Path(self.output_path, 'racetime_crawler.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        pprint(f'Crawler object saved to: {save_path}')
    
    @staticmethod
    def load(path: Union[Path, str]) -> 'RacetimeCrawler':
        with open(path, 'rb') as f:
            crawler = pickle.load(f)
        pprint(f'Crawler object loaded from: {path}')
        pprint(f'Number of race ids in races_df ({len(crawler._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(crawler.race_ids)})') if len(crawler._list_ids_in_races_df()) != len(crawler.race_ids) else None
        pprint(f'Number of columns in races_df ({len(crawler.races_df.columns)}) does not match number of cols in self.races_df_cols ({len(crawler.races_df_cols)})') if len(crawler.races_df_cols) != len(crawler.races_df.columns) else None
        return crawler
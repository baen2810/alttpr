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
from bs4 import BeautifulSoup
from typing import Any
from warnings import warn
from typing import List, Union
from pathlib import Path
from tqdm import trange, tqdm
from alttpr.utils import pprint, get_list, clean_race_info_str

# import base64
# import io
# import re
# import struct
import os
import numpy as np
import pandas as pd
import requests
import pickle


DEBUG = True  # bool(os.environ.get('ALTTPR_DEBUG'))  # some of the crawlers can print debug info
pprint('DEBUG mode active') if DEBUG else None
NAN_VALUE = np.nan

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
            'race_href', 'race_goal', 'race_permalink', 'race_info', 'rr_info_cleansed', 'race_state',
            'race_start', 'race_timer', 'race_n_entrants', 'entrant_place', 'race_info_norm',
            'race_timer_sec', 'is_game', 'mode_boots','race_mode', 'race_mode_simple', 'race_tournament',
            'entrant_rank', 'entrant_name', 'entrant_href', 'entrant_finishtime',
            ]
        self.races_df: pd.DataFrame = pd.DataFrame(columns=self.races_df_cols)
        self.output_path: Path = Path(os.getcwd(), 'export')
        self.base_url: str = r"https://racetime.gg/"
        self.last_updated: pd.Timestamp = pd.Timestamp.now()
        self.community_race_thres = 5
        self.weekday_dict_DE = {'0': 'Montag', '1': 'Dienstag', '2': 'Mittwoch', '3': 'Donnerstag', '4': 'Freitag', '5': 'Samstag', '6': 'Sonntag'}

    def get(self, host_ids: Union[str, List[str]]) -> None:
        self._get_hosts(host_ids)
        self._get_race_ids(n_pages=2 if DEBUG else None)
        self._get_races_data()
        self.last_updated = pd.Timestamp.now()
    
    def _get_hosts(self, host_ids: Union[str, List[str]]) -> None:
        self.host_ids = [host_ids] if isinstance(host_ids, str) else host_ids
        all_hosts_data = []
        for host_id in self.host_ids:
            url = self.base_url + 'user/' + host_id
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

    def _get_race_ids(self, n_pages: int = None) -> None:
        for i, row in self.hosts_df.iterrows():
            if n_pages is None:
                n_pages = row.n_pages
            for p in trange(n_pages, desc=f'Extracting races from host \'{row.host_name}\''):
                url = self.base_url + 'user/' + row.host_id + f'?page={p+1}'
                response = self._scrape(url)
                if response.status_code == 200:
                    psoup = BeautifulSoup(response.content, "html.parser")
                    page_races_lst = psoup.find("div", {"class": "user-races race-list"}).find_all('ol')[0].find_all('li', recursive=False)
                    page_races_lst = [r.find("span", {"class": "slug"}).text for r in page_races_lst]
                    self.race_ids = list(sorted(set(self.race_ids + page_races_lst)))
                else:
                    raise ParseError(f'unable to process page \'{url}\'')
        pprint(f'Collected {len(self.race_ids)} races from {i+1} hosts.')
    
    def _get_races_data(self):
        self.races_df = pd.concat([self._get_race_data(r) for r in tqdm(self.race_ids, desc='Parsing races')], ignore_index=True)            
    
    def _get_race_data(self, url: str) -> pd.DataFrame:
        response = self._scrape(self.base_url + self.game_filter + '/' + url)
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
                try:
                    href_user = e.find('a', {"class": "user-pop inline"})['href']
                except:
                    href_user = e.find('a', {"class": "user-pop inline supporter"})['href']
                entrant_rank = int(e.find('span', {"class": "place"}).text.strip().replace('th', '').replace('rd', '').replace('nd', '').replace('st', '').replace('—', '10_000'))
                entrant_finishtime = e.find('time', {"class": "finish-time"}).text
                entrant_finishtime = NAN_VALUE if entrant_finishtime=='None' else entrant_finishtime
                df_races = pd.concat([df_races, pd.DataFrame({
                    'race_href': [url],
                    'race_goal': [race_goal],
                    'race_permalink': [rr_permalink],
                    'race_info': [rr_info],
                    'race_href': [url],
                    'race_goal': [race_goal],
                    'race_state': [race_state],
                    'race_start': [race_start],
                    'race_timer': [rsoup.find('time', {'class': 'timer'}).text.strip()[:-2]],
                    'race_n_entrants': [race_n_entrants],
                    'entrant_place': [e.find('span', {"class": "place"}).text.strip()],
                    'entrant_rank': [entrant_rank],
                    'entrant_name': [e.find('span', {"class": "name"}).text],
                    'entrant_href': [href_user],
                    'entrant_finishtime': [entrant_finishtime],
                })], ignore_index=True)
            df_races.race_start = df_races.race_start.dt.tz_localize(None)
            df_races.race_n_entrants = pd.to_numeric(df_races.race_n_entrants)
            df_races.race_goal = df_races.race_goal.str.replace('\n', '')
            df_races.race_state = df_races.race_state.str.replace('\n', '')
            df_races.entrant_href = df_races.entrant_href.str.replace('/user/', '')
            df_races.entrant_finishtime = pd.to_datetime(df_races.entrant_finishtime.replace('—', NAN_VALUE))
            df_races.entrant_rank = df_races.entrant_rank.replace(10_000, NAN_VALUE)
            df_races.race_info = df_races.race_info.fillna('')
            df_races['race_info_norm'] = [clean_race_info_str(txt) for txt in df_races.race_info]
            df_races['race_timer_sec'] = [int(r.split(':')[0])*60*60 + int(r.split(':')[1])*60 + int(r.split(':')[2]) for r in df_races.race_timer]
            df_races.sort_values(['entrant_rank'], ascending=True).reset_index(drop=True)
            # df_races.entrant_finishtime = get_list(df_races.entrant_finishtime)
            df_races['is_game'] = [1 if self.game_filter in r.lower() else 0 for r in df_races.race_goal]
            df_races = self._parse_race_info(df_races)
            # df_races = df_races[self.races_df_cols]
            return df_races
        else:
            raise ParseError(f'unable to process race_id \'{url}\'')

    def _parse_race_info(self, df_rr_details: pd.DataFrame):
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
        # df_races = df_races_tmp.set_index('race_href').join(df_rr_details[['race_href', 'rr_permalink', 'rr_info', 'race_tournament', 'race_mode', 'race_mode_simple', 'mode_boots']].set_index('race_href')).reset_index()
        df_rr_details.race_tournament = ['Community Race' if e >= self.community_race_thres and type(t) == float else t for e,t in zip(df_rr_details.race_n_entrants, df_rr_details.race_tournament)]
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
        export_path = Path(self.output_path, 'hosts_df.xlsx')
        self.hosts_df.to_excel(export_path, index=False, engine='openpyxl')
        print(f'Exported data to: {self.output_path}')

    def save(self) -> None:
        if not self.output_path.exists():
            self.output_path.mkdir(parents=True)
        save_path = Path(self.output_path, 'racetime_crawler.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)
        print(f'Crawler object saved to: {save_path}')
    
    @staticmethod
    def load(path: Union[Path, str]) -> 'RacetimeCrawler':
        with open(path, 'rb') as f:
            crawler = pickle.load(f)
        print(f'Crawler object loaded from: {path}')
        return crawler
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
from alttpr.utils import pprint, get_list

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
        self.races_df: pd.DataFrame = pd.DataFrame()
        self.output_path: Path = Path(os.getcwd(), 'export')
        self.base_url: str = r"https://racetime.gg/"
        self.last_updated: pd.Timestamp = pd.Timestamp.now()

    def get(self, host_ids: Union[str, List[str]]) -> None:
        self._get_hosts(host_ids)
        self._get_race_ids(n_pages=2 if DEBUG else None)
        self._get_racerooms()
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
    
    def _get_racerooms(self):
        self.races_df = pd.concat([self._get_raceroom(r) for r in tqdm(self.race_ids, desc='Parsing races')], ignore_index=True)            
    
    def _get_raceroom(self, url: str) -> pd.DataFrame:
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
            df_races['race_timer_sec'] = [int(r.split(':')[0])*60*60 + int(r.split(':')[1])*60 + int(r.split(':')[2]) for r in df_races.race_timer]
            df_races.sort_values(['entrant_rank'], ascending=True).reset_index(drop=True)
            # df_races.entrant_finishtime = get_list(df_races.entrant_finishtime)
            df_races['is_game'] = [1 if self.game_filter in r.lower() else 0 for r in df_races.race_goal]
            return df_races
        else:
            raise ParseError(f'unable to process race_id \'{url}\'')

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
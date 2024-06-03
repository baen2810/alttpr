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
from tqdm import trange
from alttpr.utils import pprint

# import base64
# import io
# import re
# import struct
import os
import pandas as pd
import requests
import pickle


DEBUG = bool(os.environ.get('ALTTPR_DEBUG'))  # some of the crawlers can print debug info


class RacetimeCrawlerException(Exception):
    """Base class for exceptions."""


class ParseError(RacetimeCrawlerException):
    """Parsing a url failed."""


class UnsupportedFormatError(RacetimeCrawlerException):
    """File format is not supported."""


class RacetimeCrawler:
    def __init__(self) -> None:
        self.host_ids: List[str] = []
        self.race_ids: List[str] = []
        self.hosts_df: pd.DataFrame = pd.DataFrame()
        self.output_path: Path = Path(os.getcwd(), 'export')
        self.base_url: str = r"https://racetime.gg/user/"
        self.last_updated: pd.Timestamp = pd.Timestamp.now()

    def get(self, host_ids: Union[str, List[str]]) -> None:
        self._get_hosts(host_ids)
        self._get_races()
        self.last_updated = pd.Timestamp.now()
    
    def _get_hosts(self, host_ids: Union[str, List[str]]) -> None:
        self.host_ids = [host_ids] if isinstance(host_ids, str) else host_ids
        all_hosts_data = []
        for host_id in self.host_ids:
            url = self.base_url + host_id
            response = self._scrape(url)
            soup = BeautifulSoup(response.content, 'html.parser')
            if response.status_code == 200:
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

    def _get_races(self, n_pages: int = None) -> None:
        for i, row in self.hosts_df.iterrows():
            if n_pages is None:
                n_pages = row.n_pages
            for p in trange(n_pages, desc=f'Extracting races from host \'{row.host_name}\''):
                url = self.base_url + row.host_id + f'?page={p+1}'
                response = self._scrape(url)
                psoup = BeautifulSoup(response.content, "html.parser")
                page_races_lst = psoup.find("div", {"class": "user-races race-list"}).find_all('ol')[0].find_all('li', recursive=False)
                page_races_lst = [r.find("span", {"class": "slug"}).text for r in page_races_lst]
                self.race_ids = list(sorted(set(self.race_ids + page_races_lst)))
        pprint(f'Collected {len(self.race_ids)} races from {i+1} hosts.')
    
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
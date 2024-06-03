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
        self.hosts_df: pd.DataFrame = pd.DataFrame()
        self.output_path: Path = Path(os.getcwd(), 'export')
        self.base_url: str = r"https://racetime.gg/user/"
        self.last_updated: pd.Timestamp = pd.Timestamp.now()

    def get(self, host_ids: Union[str, List[str]]) -> None:
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
                raise ValueError(f'unable to process host_id \'{host_id}\'')
        self.hosts_df = pd.concat(all_hosts_data, ignore_index=True)
        self.last_updated = pd.Timestamp.now()  # Update the last_updated attribute

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
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

# TODO check negative values for race_last_place
# TODO add_race() method and check error in race 'https://racetime.gg/ootr/adequate-starfox-1938'
# TODO improved parsing and transform speed
# TODO implement game filter early while parsing
# TODO determe where and whether self.hosts_df gets updated in crawl()
# TODO make sure ongoing races don't get crawled

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
    """Parsing a beautifulsoup-object failed."""

class ScrapeError(RacetimeCrawlerException):
    """Scraping a url failed."""

class CrawlError(RacetimeCrawlerException):
    """Crawling a url failed."""

class DataRetrievalError(RacetimeCrawlerException):
    """Retrieving data from the crawler failed."""

class MetricsError(RacetimeCrawlerException):
    """Creating metrics failed."""
    
class StatsError(RacetimeCrawlerException):
    """Creating stats failed."""

class ExportError(RacetimeCrawlerException):
    """Exporting data failed."""

class SaveError(RacetimeCrawlerException):
    """Saving the crawler failed."""

class LoadError(RacetimeCrawlerException):
    """Loading the crawler failed."""
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
        self.stats_template_path: Path = Path(os.getcwd(), 'input', 'stats_template_alttpr.xlsx')
        self.parse_template_path: Path = Path(os.getcwd(), 'input', 'parse_template_alttpr.xlsx')
        self.base_url: str = r"https://racetime.gg"
        self.last_updated: pd.Timestamp = pd.Timestamp.now()
        self.community_race_thres = 5
        self.weekday_dict_DE = {'Monday': 'Montag', 'Tuesday': 'Dienstag', 'Wednesday': 'Mittwoch', 'Thursday': 'Donnerstag', 'Friday': 'Freitag', 'Saturday': 'Samstag', 'Sunday': 'Sonntag'}
        self.weekday_dict_EN = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday', '6': 'Sunday'}
        self.lang='DE'
        self.windows_dict: dict = {'total': None, 'last_race': 0, 'last_30_days': 30, }
        self.drop_forfeits_dict: dict = {'forfeits_included': False, 'forfeits_excluded': True, }
        self.entrant_has_medal_dict: dict = {'all_races': None, 'medal_races': True, }
    
    def initialize_from_crawler(self, gg: 'RacetimeCrawler') -> None:
        try:
            self.game_filter = gg.game_filter
            pprint(f'Sucessfully initialized \'game_filter\': {self.game_filter}')
        except:
            pprint('Could not initialize class attribute \'game_filter\'')
        try:
            self.host_ids = gg.host_ids
            pprint(f'Sucessfully initialized \'host_ids\': {len(self.host_ids)} entries')
        except:
            pprint('Could not initialize class attribute \'host_ids\'')
        try:
            self.race_ids = gg.race_ids
            pprint(f'Sucessfully initialized \'race_ids\': {len(self.race_ids)} entries')
        except:
            pprint('Could not initialize class attribute \'race_ids\'')
        try:
            self.races_df_cols_cr = gg.races_df_cols_cr
            pprint(f'Sucessfully initialized \'races_df_cols_cr\': {len(self.races_df_cols_cr)} entries')
        except:
            pprint('Could not initialize class attribute \'races_df_cols_cr\'')
        try:
            self.races_df_cols_tf = gg.races_df_cols_tf
            pprint(f'Sucessfully initialized \'races_df_cols_tf\': {len(self.races_df_cols_tf)} entries')
        except:
            pprint('Could not initialize class attribute \'races_df_cols_tf\'')
        try:
            self.hosts_df = gg.hosts_df
            pprint(f'Sucessfully initialized \'hosts_df\': {self.hosts_df.shape}')
        except:
            pprint('Could not initialize class attribute \'hosts_df\'')
        try:
            self.races_df = gg.races_df
            pprint(f'Sucessfully initialized \'races_df\': {self.races_df.shape}')
        except:
            pprint('Could not initialize class attribute \'races_df\'')
        try:
            self.stats_df = gg.stats_df
            pprint(f'Sucessfully initialized \'stats_df\': {self.stats_df.shape}')
        except:
            pprint('Could not initialize class attribute \'stats_df\'')
        try:
            self.metrics_df = gg.metrics_df
            pprint(f'Sucessfully initialized \'metrics_df\': {self.metrics_df.shape}')
        except:
            pprint('Could not initialize class attribute \'metrics_df\'')
        try:
            self.output_path = gg.output_path
            pprint(f'Sucessfully initialized \'output_path\': {self.output_path}')
        except:
            pprint('Could not initialize class attribute \'output_path\'')
        try:
            self.stats_template_path = gg.stats_template_path
            pprint(f'Sucessfully initialized \'stats_template_path\': {self.stats_template_path}')
        except:
            pprint('Could not initialize class attribute \'stats_template_path\'')
        try:
            self.base_url = gg.base_url
            pprint(f'Sucessfully initialized \'base_url\': {self.base_url}')
        except:
            pprint('Could not initialize class attribute \'base_url\'')
        try:
            self.last_updated = gg.last_updated
            pprint(f'Sucessfully initialized \'last_updated\': {self.last_updated}')
        except:
            pprint('Could not initialize class attribute \'last_updated\'')
        try:
            self.community_race_thres = gg.community_race_thres
            pprint(f'Sucessfully initialized \'community_race_thres\': {self.community_race_thres}')
        except:
            pprint('Could not initialize class attribute \'community_race_thres\'')
        try:
            self.weekday_dict_DE = gg.weekday_dict_DE
            pprint(f'Sucessfully initialized \'weekday_dict_DE\': {len(self.weekday_dict_DE)} entries')
        except:
            pprint('Could not initialize class attribute \'weekday_dict_DE\'')
        try:
            self.weekday_dict_EN = gg.weekday_dict_EN
            pprint(f'Sucessfully initialized \'weekday_dict_EN\': {len(self.weekday_dict_EN)} entries')
        except:
            pprint('Could not initialize class attribute \'weekday_dict_EN\'')
        try:
            self.lang = gg.lang
            pprint(f'Sucessfully initialized \'lang\': {self.lang}')
        except:
            pprint('Could not initialize class attribute \'lang\'')
        try:
            self.windows_dict = gg.windows_dict
            pprint(f'Sucessfully initialized \'windows_dict\': {len(self.windows_dict)} entries')
        except:
            pprint('Could not initialize class attribute \'windows_dict\'')
        try:
            self.drop_forfeits_dict = gg.drop_forfeits_dict
            pprint(f'Sucessfully initialized \'drop_forfeits_dict\': {len(self.drop_forfeits_dict)} entries')
        except:
            pprint('Could not initialize class attribute \'drop_forfeits_dict\'')
        try:
            self.entrant_has_medal_dict = gg.entrant_has_medal_dict
            pprint(f'Sucessfully initialized \'entrant_has_medal_dict\': {len(self.entrant_has_medal_dict)} entries')
        except:
            pprint('Could not initialize class attribute \'entrant_has_medal_dict\'')

    def sanity_checks(self):
        pprint('---Check entrant_id lengths')
        pprint('Known issues: racetime.gg may change href structure')
        pprint('Required: String length 16')
        df_tmp = self.races_df
        df_tmp['entrant_id_len'] = [len(x) for x in df_tmp['entrant_id']]
        val_cnt = df_tmp['entrant_id_len'].value_counts()
        pprint(f'Value Counts: {val_cnt}')

    def get_df(self, host_ids: Union[str, List[str]] = [], generic_filter: tuple = (None, None), drop_forfeits: bool = False, cols: List[str] = [],
               host_rows_only: bool = False, windowed: Union[int, tuple] = None, unique: bool = False,
               game_filter: bool = True, entrant_has_medal: bool = None, ) -> pd.DataFrame:
        # TODO integrate host_name filter
        try:
            filter_col, filter_val = generic_filter
            host_ids = [host_ids] if isinstance(host_ids, str) else host_ids
            host_ids = list(self.hosts_df.host_id) if len(host_ids) == 0 else host_ids
            cols = self.races_df_cols_cr + self.races_df_cols_tf if len(cols) == 0 else cols
            df = self.races_df[self.races_df.race_id.isin(self.races_df[self.races_df.entrant_id.isin(host_ids)].race_id)]
            df = df if filter_col is None else df[df[filter_col]==filter_val]
            df = df[[self.game_filter.lower() in r for r in df.race_id]] if game_filter else df
            df = df.dropna(subset=['entrant_finishtime']) if drop_forfeits else df
            df = df[df.entrant_id.isin(host_ids)] if host_rows_only else df
            df = df[~df.entrant_has_medal.isna()] if entrant_has_medal else df
            if type(windowed) == int:
                if windowed == 0:  # get last race per host
                    df = df.set_index(['entrant_name', 'race_start']).join(
                        df[['entrant_name', 'race_start']].groupby('entrant_name').max().reset_index().set_index(
                            ['entrant_name', 'race_start']), how='inner').reset_index()
                else:
                    df = df[df.race_start >= dt.now() - pd.Timedelta(days=windowed)]
            elif type(windowed) == tuple:
                min_race_date, max_race_date = windowed
                df = df[df.race_start >= min_race_date]
                df = df[df.race_start <= max_race_date]
            df = df[cols]
            df = df.drop_duplicates() if unique else df
        except Exception as e:
            raise DataRetrievalError(f'unable to retrieve data for: {host_ids=}, {generic_filter=}, {drop_forfeits=}, {cols=}, {host_rows_only=}, {windowed=}, {unique=}') from e
        return df
    
    def refresh_transforms(self) -> None:
        pprint('Refreshing all transforms')
        self._parse_race_info()
        pprint('Refreshing all metrics')
        self.get_metrics()
        pprint('Refreshing all stats')
        self.get_stats()
        pprint('Everything up to date')
        pprint(f'Number of race ids in races_df ({len(self._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(self.race_ids)})') if len(self._list_ids_in_races_df()) != len(self.race_ids) else None
        pprint(f'Number of columns in races_df ({len(self.races_df.columns)}) does not match number of cols in self.races_df_cols_cr + _tf ({len(self.races_df_cols_cr) + len(self.races_df_cols_tf)})') if len(self.races_df_cols_cr) + len(self.races_df_cols_tf) != len(self.races_df.columns) else None
    
    def get_metrics(self):
        df = pd.DataFrame()
        for window_name, window in self.windows_dict.items():
            for forfeits_name, drop_forfeits in self.drop_forfeits_dict.items():
                for entrant_has_medal_name, entrant_has_medal in self.entrant_has_medal_dict.items():
                    # if window_name=='total' and forfeits_name=='forfeits_included' and entrant_has_medal_name=='all_races':
                    #     print('now')
                    pprint(f'Creating metrics for {window_name=}, {forfeits_name=}, {entrant_has_medal_name=}')
                    df_tmp = self._get_metrics_wrapped(windowed=window, drop_forfeits=drop_forfeits, entrant_has_medal=entrant_has_medal)
                    df_tmp.metric = window_name + '|' + forfeits_name + '|' + entrant_has_medal_name + '|' + df_tmp.metric
                    df = pd.concat([df, df_tmp], axis=0, ignore_index=True)
        # add custom metrics
        df_raw_counts = pdidx(self.get_df(host_rows_only=True, game_filter=False)[['entrant_name', 'race_start']].groupby('entrant_name').count())
        df_raw_counts.columns = ['unfiltered|forfeits_included|all_races|race_start|count']
        df_raw_counts = df_raw_counts.T.rename_axis('metric').reset_index()
        df = pd.concat([df, df_raw_counts], axis=0, ignore_index=True)
        df_expanded = df.metric.str.split('|', expand=True)
        df_expanded.columns = ['scope', 'forfeits', 'win_filter', 'name', 'aggregation', 'if', 'pivoted_by', 'is', 'pivot_label']
        df_expanded = df_expanded.drop(columns=['if', 'is'])
        df = pd.concat([df_expanded, df], axis=1)
        df.metric = '<' + df.metric + '>'
        pprint(f'Created {df.shape[0]} metrics for {self.hosts_df.shape[0]} racers')
        self.metrics_df = df

    def _get_metrics_wrapped(self, windowed: Union[int, tuple] = None, drop_forfeits: bool = False, entrant_has_medal: bool = None):
        try:
            df = self.get_df(host_rows_only=True, windowed=windowed, drop_forfeits=drop_forfeits, entrant_has_medal=entrant_has_medal)
            if df.shape[0] == 0:
                pprint(f'get_df() returned empty dataframe for {windowed=}, {drop_forfeits=}, {entrant_has_medal=}!')
                return pd.DataFrame(columns=['metric'] + list(self.hosts_df.host_name))
            # df = df[df.race_group=='Community-/Weekly-Race']
            idx_col = 'entrant_name'
            groupby_col_lst = ['race_category', 'race_group', 'race_start_weekday', 'entrant_place', 'race_mode_simple']
            to_date_cols_lst = [
                'race_start|', 'entrant_has_won|', 'entrant_has_medal|', 'entrant_has_forfeited|',  'entrant_has_top10|',
                'entrant_below_2h00m|', 'entrant_below_1h45m|', 'entrant_below_1h30m|', 'entrant_below_1h15m|', 'entrant_below_1h00m|',
                'entrant_is_last|',]
            to_time_cols_lst = ['entrant_finishtime|']
            pct_base_metrics_lst = [
                'race_start|count|if|race_group|is|Community-/Weekly-Race'
                ]
            agg_dict = {
                'entrant_finishtime':           ['min', 'max', 'median'],
                'entrant_rank':                 ['min', 'max', 'median'],
                'race_n_forfeits':              ['min', 'max', 'median'],
                'race_last_place':              ['min', 'max', 'median'],
                'entrant_distance_to_last':     ['min', 'max', 'median'],
                'race_n_entrants':              ['min', 'max', 'median'],
                'race_start' :                  ['count', 'min', 'max'],
                'entrant_has_won':              ['count', 'min', 'max'],
                'entrant_has_medal':            ['count', 'min', 'max'],
                'entrant_has_top10':            ['count', 'min', 'max'],
                'entrant_has_forfeited':        ['count', 'min', 'max'],
                'entrant_below_2h00m':          ['count', 'min', 'max'],
                'entrant_below_1h45m':          ['count', 'min', 'max'],
                'entrant_below_1h30m':          ['count', 'min', 'max'],
                'entrant_below_1h15m':          ['count', 'min', 'max'],
                'entrant_below_1h00m':          ['count', 'min', 'max'],
                'entrant_is_last':              ['count', 'min', 'max'],
            }
            values_lst = []
            for k, v_lst in agg_dict.items():
                values_lst += [k + '|' + v for v in v_lst]
            metrics_cols = list(agg_dict.keys())
            df_out = pdidx(df[[idx_col] + metrics_cols].groupby(idx_col).agg(agg_dict))
            for groupby_col in groupby_col_lst:
                df_tmp = pdidx(pdidx(
                    df[[idx_col, groupby_col] + metrics_cols].groupby([idx_col, groupby_col]).agg(agg_dict)).reset_index().pivot(
                    index=idx_col,
                    columns=groupby_col,
                    values=values_lst))
                df_tmp.columns = [c.replace(c.split('|')[-1], 'if|' + groupby_col + '|is|' + c.split('|')[-1]) for c in df_tmp.columns]
                df_out = pd.concat([df_out, df_tmp], axis=1)
            # get topN stats
            df_cr = df[df.race_group=='Community-/Weekly-Race']
            topN_groupby_col_lst = [
                # (topN_groupby_col, ranked_col, ascending, topN_limit)
                ('race_mode_simple', 'race_start|count', False, 3, None, None),
                ('race_mode_simple', 'race_start|count', True, 3, None, None),
                ('race_mode_simple', 'entrant_finishtime|median', True, 5, 'race_start|count', 10),
                ('race_id', 'race_n_entrants|max', False, 1, None, None),
                ('entrant_place', 'entrant_rank|count', False, 5, None, None),
                ('race_start_weekday', 'race_start|count', False, None, None, None),
                ('race_start_weekday', 'race_start|count', True, None, None, None),
                ]
            topN_agg_dict = {
                'entrant_rank':                 ['count', 'median',],
                'race_start':                   ['max', 'min', 'count'],
                'race_n_entrants':              ['min', 'max', 'median'],
                'entrant_finishtime':              ['min', 'max', 'median'],
            }
            for topN_groupby_col_tuple in tqdm(topN_groupby_col_lst, desc=pprintdesc('Aggregating')):
                topN_groupby_col, ranked_col, ascending, topN_limit, filter_varname, filter_limit = topN_groupby_col_tuple
                if idx_col == topN_groupby_col:
                    topN_groupby_col += '_gr'
                    df[topN_groupby_col] = df[idx_col]
                df_tmp = pdidx(df_cr[list(set([idx_col] + [topN_groupby_col] + list(topN_agg_dict.keys())))].groupby(
                    [idx_col, topN_groupby_col]).agg(topN_agg_dict)).reset_index().sort_values([idx_col, ranked_col], ascending=False)
                rank_varname = '|' + topN_groupby_col + '_topN|'
                asc_str = '_topN_sorted_asc_by_' if ascending else '_topN_sorted_desc_by_'
                if filter_limit is not None:
                    df_tmp = df_tmp[df_tmp[filter_varname] >= filter_limit]
                    fvn = filter_varname.replace('|', '_')
                    asc_str = f'_topN_filtered_by_{fvn}>{str(filter_limit)}_and_sorted_asc_by_' if ascending else f'_topN_filtered_by_{fvn}>{str(filter_limit)}_and_sorted_desc_by_'
                df_tmp[rank_varname] = df_tmp.groupby(idx_col)[ranked_col].rank("dense", ascending=ascending)
                if topN_limit is not None:
                    df_tmp = df_tmp[df_tmp[rank_varname] <= topN_limit]
                df_tmp = df_tmp.sort_values(by=[idx_col, rank_varname], ascending=False)
                df_tmp[rank_varname] = df_tmp[rank_varname].astype(int).astype(str)
                df_tmp_str = df_tmp[[idx_col, rank_varname, topN_groupby_col]].pivot_table(
                    index=idx_col,
                    columns=rank_varname,
                    values=[topN_groupby_col],
                    aggfunc='first'
                )
                df_tmp = df_tmp.pivot_table(
                    index=idx_col,
                    columns=rank_varname,
                    values=list(df_tmp.columns.drop([idx_col, rank_varname, topN_groupby_col]))
                )
                df_tmp = pdidx(df_tmp, delimiter='$')
                df_tmp.columns = [c.replace('$', '|if' + rank_varname + 'is|') for c in df_tmp.columns]
                df_tmp_str = pdidx(df_tmp_str, delimiter='$')
                df_tmp_str.columns = [c.replace('$', '|first|if' + rank_varname + 'is|') for c in df_tmp_str.columns]
                df_tmp = pd.concat([df_tmp, df_tmp_str], axis=1)
                df_tmp.columns = [c.replace('_topN', asc_str + ranked_col.replace('|', '_')) for c in df_tmp.columns]
                df_out = pd.concat([df_out, df_tmp], axis=1)
            for c in tqdm(df_out.columns, desc=pprintdesc('Formatting')):
                col_name = c.split('|')[0] + '|'         
                aggregation_name = c.split('|')[1]  
                if col_name in to_date_cols_lst and aggregation_name != 'count':
                    df_out[c] = [to_dstr(e) if type(e).__name__ != 'NaTType' else e for e in df_out[c]]
                    df_out = df_out.rename(columns={c: c.replace(
                        '|min', '|min-date').replace(
                        '|max', '|max-date').replace(
                        '|median', '|median-date')})
                elif col_name in to_time_cols_lst and aggregation_name != 'count':
                    df_out[c] = [to_tstr(e) if type(e).__name__ != 'NaTType' else e for e in df_out[c]]
                    df_out = df_out.rename(columns={c: c.replace(
                        '|min', '|min-time').replace(
                        '|max', '|max-time').replace(
                        '|median', '|median-time')})
                elif aggregation_name != 'first':
                    df_out[c] = [str(int(round(e, 0))) if not(pd.isna(e)) else e for e in df_out[c]]
            # add days past for date columns
            df_days_past = df_out[[c for c in df_out.columns if '-date' in c]]
            for c in tqdm(df_days_past.columns, desc=pprintdesc('Add dayspast-metrics')):
                if any([d == c[:len(d)] for d in to_date_cols_lst]) and not(any([d in c for d in ['count']])):
                    df_days_past[c] = [abs((pd.Timestamp(dt.now()) - pd.Timestamp(e))).days if type(e).__name__ != 'NaTType' else e for e in df_days_past[c]]
                    df_days_past = df_days_past.rename(columns={c: c.replace('-date', '-date-days-past')})
            df_out = pd.concat([df_out, df_days_past], axis=1)
            df_out = df_out.T.rename_axis('metric').reset_index()
            # add pct metrics
            df_pct_metrics = df_out[['|count|' in m for m in df_out.metric]]
            for i, base_metric in enumerate(tqdm(pct_base_metrics_lst, desc=pprintdesc('Add %-metrics'))):
                base_metric_df = df_out[df_out.metric==base_metric]
                df_pct = df_pct_metrics.copy()
                for c in list(self.hosts_df.host_name):
                    try:
                        df_pct[c] = df_pct_metrics[c].astype(float) / list(base_metric_df[c].astype(float))[0]
                        df_pct[c] = [str(int(round(v*100, 0))) + '%' if not(pd.isna(v)) else v for v in df_pct[c]]
                    except Exception as e:
                        err_str = ''
                        err_str = err_str + 'df_pct_metrics, ' if c in df_pct_metrics.keys() else err_str
                        err_str = err_str + 'base_metric_df, ' if c in base_metric_df.keys() else err_str
                        df_pct[c] = 0.0
                        pprint(f'Error when building pct metrics. No race available for {c}? Cannot be found in {err_str}. Setting df_pct[\'{c}\']=0.0')
                df_pct.metric = df_pct.metric.str.replace('|count|', f'|pct_b{i+1}|')
                df_out = pd.concat([df_out, df_pct], axis=0)
            df_out = df_out.set_index('metric').dropna(how='all').sort_index().reset_index()
        except Exception as e:
            raise MetricsError(f'unable to process metrics for {windowed=}, {drop_forfeits=}, {entrant_has_medal=}') from e 
        return df_out

    def get_stats(self):
        try:
            df_stats_static_cols = ['ID', 'Kategorie', 'Template']
            df_stats = pd.read_excel(self.stats_template_path).dropna(subset=df_stats_static_cols)
            for host_name in list(self.hosts_df.host_name):
                if host_name not in df_stats.columns:
                    df_stats[host_name] = 'x'              
            df_stats = df_stats[df_stats_static_cols + list(self.hosts_df.host_name)]
            df_stats.ID = df_stats.ID.astype(int)
            pprint('Creating stats')
            pprint(f'Stats template file: {self.stats_template_path}')
        except Exception as e:
            raise StatsError(f'unable to retrieve stats: {self.stats_template_path=}') from e
        for host_name in list(self.hosts_df.host_name):
            try:
                metrics_dict = dict(zip(self.metrics_df.metric, self.metrics_df[host_name]))
                df_stats[host_name] = [t if not(pd.isna(h)) else np.nan for t, h in zip(df_stats.Template, df_stats[host_name])]
                df_stats[host_name] = df_stats[host_name].str.replace('<host_name>', host_name)
                df_stats[host_name] = df_stats[host_name].str.replace('<game_filter>', self.game_filter)
                df_stats[host_name] = df_stats[host_name].str.replace('<LAST_UPDATED>', self.last_updated.strftime('%d.%m.%Y'))
                for k, v in tqdm(metrics_dict.items(), desc=pprintdesc(f'Host \'{host_name}\'')):
                    df_stats[host_name] = [f.replace(k, str(v)) if not(pd.isna(f)) else f for f in df_stats[host_name]]
                if self.lang=='DE':
                    for k, v in self.weekday_dict_DE.items():
                        df_stats[host_name] = [f.replace(k, str(v)) if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    for k, v in {'st Platz': '. Platz', 'nd Platz': '. Platz', 'rd Platz': '. Platz', 'th Platz': '. Platz'}.items():
                        df_stats[host_name] = [f.replace(k, str(v)) if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    df_stats[host_name] = [f.replace('vor 0 Tag(en)', 'heute') if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    df_stats[host_name] = [f.replace('in 1 Races', 'in 1 Race') if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    df_stats[host_name] = [f.replace('vor 1 Tag(en)', 'gestern').replace('vor 1 Tag', 'gestern') if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    df_stats[host_name] = [f.replace('vor 2 Tag(en)', 'vorgestern').replace('vor 2 Tagen', 'vorgestern') if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    df_stats[host_name] = [f.replace('Tag(en)', 'Tagen') if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    df_stats[host_name] = [f.replace(', nan (NaT) und nan (NaT)', '') if not(pd.isna(f)) else f for f in df_stats[host_name]]
                    df_stats[host_name] = [f.replace(' und nan (NaT)', '') if not(pd.isna(f)) else f for f in df_stats[host_name]]
            except Exception as e:
                raise StatsError(f'unable to retrieve stats: {host_name=}, {self.stats_template_path=}') from e
        self.stats_df = df_stats

    def crawl(self, n_pages: int = None, max_races: int = None) -> None:
        self._get_hosts()
        pprint(f'Required host names to parse: {list(self.hosts_df.host_name)}')
        self._get_race_ids(n_pages)
        if max_races:
            self.race_ids = self.race_ids[:max_races]
            pprint(f'>>>>> WARNING: Crawling was explicitly limited to {max_races=}')
        race_ids_crawled = self._get_races_data()
        host_names_changed_flag, host_names_missing, host_names_invalid, host_names_valid = self._validate_host_names()
        if host_names_changed_flag or race_ids_crawled > 0:
            if host_names_changed_flag:
                pprint('Host names require update:')
                pprint(f'Valid host names: {host_names_valid}')
                pprint(f'Invalid host names: {host_names_invalid}. Will be removed from metrics_df and stats_df') if len (host_names_invalid) > 0 else None
                pprint(f'Missing host names: {host_names_missing}. Will be added to metrics_df and stats_df') if len (host_names_missing) > 0 else None
                pprint(f'No data will be deleted from races_df')
            self.refresh_transforms()
        pprint(f'Number of race ids in races_df ({len(self._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(self.race_ids)})') if len(self._list_ids_in_races_df()) != len(self.race_ids) else None
        pprint(f'Number of columns in races_df ({len(self.races_df.columns)}) does not match number of cols in self.races_df_cols_cr + _tf ({len(self.races_df_cols_cr) + len(self.races_df_cols_tf)})') if len(self.races_df_cols_cr) + len(self.races_df_cols_tf) != len(self.races_df.columns) else None
                
    def _get_hosts(self) -> None:
        all_hosts_data = []
        for host_id in self.host_ids:
            try:
                url = self.base_url + '/user/' + host_id
                response = self._scrape(url)
                soup = BeautifulSoup(response.content, 'html.parser')
                n_pages = int(soup.find("div", {"class": "pagination"}).decode_contents().strip().split(' of ')[1].split('\n')[0])
                host_name = soup.find("div", {"class": "user-profile"}).find("span", {"class": "name"}).text
                host_name_url = requests.get(url).url.split('/')[-1]  # get redirect url and get stripped/cleansed username used by racetime.gg in URL
                cols = [v.text.lower().replace(' ', '_') for v in soup.find('aside').find_all('dt')[1:]]
                vals = [int(v.text.split(' ')[0]) for v in soup.find('aside').find_all('dd')[1:]]
                df_user_stats = pd.DataFrame([vals], columns=cols)
                df_user = pd.DataFrame([[host_id, host_name_url, host_name, n_pages]], columns=['host_id', 'host_name_url', 'host_name', 'n_pages'])
                df_user_stats = pd.concat([df_user, df_user_stats], axis=1)
                all_hosts_data.append(df_user_stats)
            except Exception as e:
                raise DataRetrievalError(f'unable to process host_id "{host_id}"') from e
        self.hosts_df = pd.concat(all_hosts_data, ignore_index=True)
        self.last_updated = pd.Timestamp.now()

    def _get_race_ids(self, n_pages: int) -> None:
        pprint('Finding race ids')
        for i, row in self.hosts_df.iterrows():
            host_n_pages = row.n_pages if n_pages is None else n_pages
            for p in trange(host_n_pages, desc=pprintdesc(f'Host \'{row.host_name}\'')):
                try:
                    url = self.base_url + f'/user/{row.host_id}/{row.host_name_url}?page={p+1}'
                    response = self._scrape(url)
                    if response.status_code != 200:
                        raise ScrapeError(f'Error when crawling: Response code "{response.status_code}"')
                    psoup = BeautifulSoup(response.content, "html.parser")
                    page_races_lst = psoup.find("div", {"class": "user-races race-list"}).find_all('ol')[0].find_all('li', recursive=False)
                    page_races_lst = [r.find('a', {"class": "race"})['href'] for r in page_races_lst]
                    self.race_ids = list(sorted(set(self.race_ids + page_races_lst)))
                except Exception as e:
                    raise CrawlError(f'unable to process url "{url}")') from e
        self.last_updated = pd.Timestamp.now()
        pprint(f'Collected {len(self.race_ids)} race ids from {i+1} hosts:')
    
    def _get_races_data(self):
        race_ids_to_crawl = [i for i in self.race_ids if i not in self._list_ids_in_races_df()]  # /alttpr/banzai-redboom-2002
        race_ids_already_crawled = [i for i in self.race_ids if i in self._list_ids_in_races_df()]
        pprint(f'{len(race_ids_already_crawled)} races were already parsed.')
        if len(race_ids_to_crawl) > 0:
            pprint(f'Crawling {len(race_ids_to_crawl)} new races (max 30 shown): {race_ids_to_crawl[:30]}')
            new_races_df = pd.concat([self._get_race_data(r) for r in tqdm(race_ids_to_crawl, desc=pprintdesc('Crawling races'))], ignore_index=True)
            new_races_df.race_start = new_races_df.race_start.dt.tz_localize(None)  # TODO messy. store raw data separately from transformed data
            self.races_df = pd.concat([self.races_df, new_races_df], ignore_index=True).sort_values(['race_start', 'entrant_place'], ascending=[False, True]).reset_index(drop=True)
        else:
            pprint('No new races detected. Crawler is up to date.')
        self.last_updated = pd.Timestamp.now()
        return len(race_ids_to_crawl)

    def _list_ids_in_races_df(self):
        return list(sorted(set(self.races_df.race_id)))

    def _list_host_names_in_hosts_df(self):
        return list(sorted(set(self.hosts_df.host_name)))
 
    def _list_host_names_in_metrics_df(self):
        return list(sorted(set([c for c in self.metrics_df.columns if c not in ['scope', 'forfeits', 'win_filter', 'name', 'aggregation', 'pivoted_by', 'pivot_label', 'metric']])))
    
    def _list_host_names_in_stats_df(self):
        return list(sorted(set([c for c in self.stats_df.columns if c not in ['ID', 'Kategorie', 'Template']])))
    
    def _validate_host_names(self):
        host_names_valid = self._list_host_names_in_hosts_df()
        host_names_existing = list(sorted(set(self._list_host_names_in_metrics_df() + self._list_host_names_in_stats_df())))
        host_names_missing = [x for x in host_names_valid if not x in host_names_existing]
        host_names_invalid = [x for x in host_names_existing if not x in host_names_valid]
        update_required = True if len(host_names_missing + host_names_invalid) != 0 else False
        return update_required, host_names_missing, host_names_invalid, host_names_valid

    def _get_race_data(self, url: str) -> pd.DataFrame:
        try:
            dfs = pd.read_excel(self.parse_template_path, sheet_name=['racetime-user-class-options'])
            race_url = self.base_url + url
            response = self._scrape(race_url)
            # raise ScrapeError('This is a dummy error.')
            # class_options = [
            #     {"class": "user-pop inline"},
            #     {"class": "user-pop inline supporter moderator"},
            #     {"class": "user-pop inline supporter"},
            #     {"class": "user-pop inline moderator"}, 
            #     {"class": "user-pop inline staff supporter moderator"},
            #     {"class": "user-pop inline staff supporter"},
            #     {"class": "user-pop inline staff moderator"},    
            # ]
            class_options = [{"class": c} for c in dfs['racetime-user-class-options'].class_options]
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
        except Exception as e:
            raise CrawlError(f'Failed to execute _get_race_data() on: "{url}"') from e

    def _parse_race_info(self):
        try:
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
            df.entrant_id = df.entrant_id.str.split('/', expand=True)[0]
            df.entrant_finishtime = [pd.Timedelta(t) if t != '—' else NAN_VALUE for t in df.entrant_finishtime]
            # add new transformations
            df['entrant_rank'] = [int(e.replace('th', '').replace('rd', '').replace('nd', '').replace('st', '').replace('—', '10_000')) if type(e) == str else e for e in df.entrant_place]
            df.entrant_rank = df.entrant_rank.replace(10_000, NAN_VALUE)
            df['race_info_norm'] = [clean_race_info_str(txt) for txt in df.race_info]
            df['race_timer_sec'] = [int(r.split(':')[0])*60*60 + int(r.split(':')[1])*60 + int(r.split(':')[2]) for r in df.race_timer]
            # df = df_bck.copy()
            df_n_finished = df.dropna(subset='entrant_finishtime')[['race_id', 'race_start']].groupby('race_id').count().sort_values('race_id').rename(columns={'race_start': 'race_n_finished'})
            df = df.set_index('race_id').join(df_n_finished, how='left').reset_index()
            df.race_n_finished = df.race_n_finished.fillna(0)
            df['race_n_forfeits'] = df['race_n_entrants'] - df['race_n_finished']
            # df['race_n_finished'] = df.set_index('race_id').join(df.dropna(subset='entrant_finishtime')[['race_id', 'race_start']].groupby('race_id').count(), rsuffix='_cnt', how='left').reset_index().race_start_cnt.fillna(0).astype(int)
            # df['race_n_forfeits'] = list(df.set_index('race_id').join(df[~df.entrant_place.isna()][['race_id', 'race_start']].groupby('race_id').count(), rsuffix='_cnt', how='left').reset_index().race_start_cnt.fillna(0).astype(int))
            df['race_last_place'] = [e - f for f, e in zip(df.race_n_forfeits, df.race_n_entrants)]
            df['is_game'] = df.race_id.str.split('/',expand=True)[1]
            df['race_start_weekday'] = df.race_start.dt.weekday.astype(str).replace(self.weekday_dict_EN)  # 0=Montag
            df['entrant_has_medal'] = pd.to_datetime([d.date() if r <= 3 else NAN_VALUE for d, r in zip(df.race_start, df.entrant_rank)])
            df['entrant_has_won'] = pd.to_datetime([d.date() if r <= 1 else NAN_VALUE for d, r in zip(df.race_start, df.entrant_rank)])
            df['entrant_has_top10'] = pd.to_datetime([d.date() if r <= 10 else NAN_VALUE for d, r in zip(df.race_start, df.entrant_rank)])
            df['entrant_has_forfeited'] = pd.to_datetime([d.date() if pd.isna(r) else NAN_VALUE for d, r in zip(df.race_start, df.entrant_place)])
            df['entrant_is_last'] = pd.to_datetime([d.date() if r == l else NAN_VALUE for d, r, l in zip(df.race_start, df.entrant_rank, df.race_last_place)])
            df['entrant_distance_to_last'] = [l - r for r, l in zip(df.entrant_rank, df.race_last_place)]
            df['entrant_below_2h00m'] = pd.to_datetime([d.date() if r < pd.Timedelta('02:00:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
            df['entrant_below_1h45m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:45:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
            df['entrant_below_1h30m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:30:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
            df['entrant_below_1h15m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:15:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
            df['entrant_below_1h00m'] = pd.to_datetime([d.date() if r < pd.Timedelta('01:00:00') else NAN_VALUE for d, r in zip(df.race_start, df.entrant_finishtime)])
            dfs = pd.read_excel(self.parse_template_path, sheet_name=['race-mode-definitions', 'race-mode-replaces', 'race-mode-simple-mapping', 'race-mode-simple-replaces', 'race-tournament-mapping', 'race-tournament-replaces'])
            for i, row in dfs['race-mode-definitions'].iterrows():
                df[row.colname] = [1 if any([k.strip() in r.lower() for k in row.keyword_lst.split(',')]) else 0 for r in df.race_info_norm]
            df_sum_mode_flags = df[[c for c in df.columns if 'mode' == c[:4]]]
            df['sum_modes'] = df_sum_mode_flags.sum(axis=1)
            # Mode
            mode_lst = []
            for i, row in tqdm(df.iterrows(), total=df.shape[0], desc=pprintdesc('Parsing race mode')):
                mode = ''
                for i, mode_def in dfs['race-mode-definitions'].iterrows():
                    if row[mode_def.colname] == 1:
                        mode += f'{mode_def.tag} '
                if mode == '':
                    mode = 'Unknown'
                for i, mode_rep in dfs['race-mode-replaces'].iterrows():
                    mode = mode.replace(mode_rep.search_str.replace('\xa0', ' '), mode_rep.replace_str.replace('\xa0', ' '))
                mode_lst += [mode.strip()]
            df['race_mode'] = mode_lst
            # Modus (vereinfacht)
            mode_simple_lst = []
            for m in tqdm(df.race_mode, total=df.shape[0], desc=pprintdesc('Parsing race mode (simple)')):
                mode_simple = m
                for i, mode_def in dfs['race-mode-simple-mapping'].iterrows():
                    if mode_def.search_mode == 'is_in' and m in [x.strip() for x in mode_def.search_terms.split(',')]:
                        mode_simple = mode_def.race_mode_simple
                    elif mode_def.search_mode == 'contains' and any([t.strip() in m for t in mode_def.search_terms.split(',')]):
                        mode_simple = mode_def.race_mode_simple
                for i, mode_rep in dfs['race-mode-simple-replaces'].fillna('').iterrows():
                    mode_simple = mode_simple.replace(mode_rep.search_str.replace('\xa0', ' '), mode_rep.replace_str.replace('\xa0', ' '))
                mode_simple_lst += [mode_simple]
            df['race_mode_simple'] = mode_simple_lst
            # Tournament
            tournament_lst = []
            for r in tqdm(df.race_info_norm, total=df.shape[0], desc=pprintdesc('Parsing race tournament')):
                tournament = NAN_VALUE
                for i, tourn_def in dfs['race-tournament-mapping'].iterrows():
                    if tourn_def.search_mode == 'is_in' and r in [x.strip() for x in tourn_def.search_terms.split(',')]:
                        tournament = tourn_def.tournament
                    elif tourn_def.search_mode == 'contains' and any([t.strip() in r for t in tourn_def.search_terms.split(',')]):
                        tournament = tourn_def.tournament
                tournament_lst += [tournament]
            df['race_tournament'] = tournament_lst
            df.race_tournament = ['Community Race' if e >= self.community_race_thres and type(t) == float else t for e,t in zip(df.race_n_entrants, df.race_tournament)]
            df.race_tournament = [f'Community Race (<{self.community_race_thres})' if e < self.community_race_thres and t == 'Community Race' else t for e,t in zip(df.race_n_entrants, df.race_tournament)]
            df.race_tournament = ['Unknown 1on1' if e == 2 and type(t) == float else t for e,t in zip(df.race_n_entrants, df.race_tournament)]
            df.race_tournament = ['Unknown 1on1on1' if e == 3 and type(t) == float else t for e,t in zip(df.race_n_entrants, df.race_tournament)]
            df.race_tournament = ['Unknown 2on2' if e == 4 and type(t) == float else t for e,t in zip(df.race_n_entrants, df.race_tournament)]
            df['race_category'] = [r if r in ['German Weekly', 'Community Race'] else 'Tournament' for r in df.race_tournament]
            df['race_group'] = ['Community-/Weekly-Race' if r in ['German Weekly', 'Community Race'] else 'Tournament' for r in df.race_tournament]
            df = df.sort_values(['race_start', 'entrant_place'], ascending=[False, True]).reset_index(drop=True)
            # TODO alles in einen df und dazu noch race id und tournament und game filter
            self.race_mode_map_df = df[['is_game', 'race_tournament', 'race_category', 'race_group', 'race_mode', 'race_info_norm']].drop_duplicates().sort_values(['race_tournament', 'race_category', 'race_group', 'race_mode', 'race_info_norm'])
            self.race_mode_simple_map_df = df[['is_game', 'race_group', 'race_mode_simple', 'race_mode']][df.is_game==self.game_filter.lower()].drop_duplicates().sort_values(['race_group', 'race_mode_simple', 'race_mode'])
            self.race_tournament_map_df = df[['is_game', 'race_group', 'race_category', 'race_tournament', 'race_info_norm', 'race_n_entrants']][df.is_game==self.game_filter.lower()].drop_duplicates().sort_values(['race_group', 'race_category', 'race_tournament', 'race_info_norm'])
            self.races_df_cols_tf = [c for c in df.columns if c not in self.races_df_cols_cr]
            self.races_df = df
            self.last_updated = pd.Timestamp.now()
        except Exception as e:
            raise ParseError(f'Failed to execute _parse_race_info()') from e

    def add_races(self):
        '''Add one or more races and pull data'''
        pass

    def _scrape(self, url: str) -> requests.Response:
        response = requests.get(url)
        if response.status_code != 200:
            raise ScrapeError(f'Error when scraping: Response code "{response.status_code}"')
        return response
    
    def set_output_path(self, path: Union[Path, str]) -> None:
        path = Path(path)
        if len(path.suffix) > 0:
            raise RacetimeCrawlerException('self.output_path can\'t have file extension.')
        self.output_path = path

    def set_host_ids(self, host_ids: Union[list, str]) -> None:
        self.host_ids = [host_ids] if isinstance(host_ids, str) else host_ids

    def set_stats_template_path(self, path: Union[Path, str]) -> None:
        self.stats_template_path = Path(path)

    def set_parse_template_path(self, path: Union[Path, str]) -> None:
        self.parse_template_path = Path(path)

    def set_metrics_dicts(self, windows_dict: dict, drop_forfeits_dict: dict, entrant_has_medal_dict: dict) -> None:
        self.windows_dict=windows_dict
        self.drop_forfeits_dict=drop_forfeits_dict
        self.entrant_has_medal_dict=entrant_has_medal_dict

    def export(self, path: Union[Path, str] = None, dfs: List[str] = ['hosts_df', 'races_df', 'metrics_df', 'stats_df', 'race_mode_map_df', 'race_mode_simple_map_df', 'race_tournament_map_df'], host_names: Union[str, List[str]] = [], dropna=False) -> None:
        try:
            # assert self.metrics_df is not None, "self.metrics_df is None"
            # assert self.hosts_df is not None, "self.hosts_df is None"
            # assert self.stats_df is not None, "self.stats_df is None"
            host_names = [host_names] if isinstance(host_names, str) else host_names
            host_names = list(self.hosts_df.host_name) if len(host_names) == 0 else host_names
            if not self.output_path.exists():
                self.output_path.mkdir(parents=True)
            pprint(f'Exporting data to: {self.output_path}')
            self.hosts_df[self.hosts_df.host_name.isin(host_names)].to_excel(Path(self.output_path, 'hosts_df.xlsx'), index=False, engine='openpyxl') if 'hosts_df' in dfs else None
            df_races = self.get_df()
            df_races.entrant_finishtime = [to_tstr(f) for f in df_races.entrant_finishtime]
            df_races.to_excel(Path(self.output_path, 'races_df.xlsx'), index=False, engine='openpyxl') if 'races_df' in dfs else None
            self.race_mode_map_df.to_excel(Path(self.output_path, 'race_mode_map_df.xlsx'), index=False, engine='openpyxl') if 'race_mode_map_df' in dfs else None
            self.race_mode_simple_map_df.to_excel(Path(self.output_path, 'race_mode_simple_map_df.xlsx'), index=False, engine='openpyxl') if 'race_mode_simple_map_df' in dfs else None
            self.race_tournament_map_df.to_excel(Path(self.output_path, 'race_tournament_map_df.xlsx'), index=False, engine='openpyxl') if 'race_tournament_map_df' in dfs else None
            df_metrics = self.metrics_df[['scope', 'forfeits', 'win_filter', 'name', 'aggregation', 'pivoted_by', 'pivot_label', 'metric'] + host_names].dropna(how='all', subset=host_names).astype(str)
            for c in df_metrics.columns:
                df_metrics[c] = [d.replace('NaT', '').replace('nan', '').replace('0 days ', '') for d in df_metrics[c]]
            df_metrics.to_excel(Path(self.output_path, 'metrics_df.xlsx'), index=False, engine='openpyxl') if 'metrics_df' in dfs else None
            export_cols = ['ID', 'Kategorie', 'Template'] if len(host_names) > 1 else ['ID', 'Kategorie']
            df_stats = self.stats_df[export_cols + host_names].astype(str).replace('nan', np.nan).dropna(how='all', subset=host_names).astype(str)
            # check NA status
            df_tmp = df_stats.copy()
            for h in host_names:
                df_tmp[h] = [1 if ('<' in x and '>' in x and '|' in x) or 'NaT' in x or 'NaN' in x or ' nan ' in x else 0 for x in df_tmp[h]]
            df_tmp = df_tmp.replace(0, np.nan)
            try:
                df_na = df_tmp.drop(columns=['Template'])
            except:
                df_na = df_tmp.copy()
            df_na = df_na.drop(columns=['ID']).groupby('Kategorie').sum().reset_index()
            df_na = df_na.replace(0, np.nan).dropna(how='all', subset=host_names).reset_index(drop=True)
            if not(dropna) and df_na.shape[0] > 0:
                pprint('Found unresolved metrics, NaNs and/or NaTs')
                print(df_na)
            # export       
            if dropna:
                n_na = df_stats[df_tmp[host_names[0]].notna()].shape[0]
                df_stats = df_stats[df_tmp[host_names[0]].isna()]
                pprint(f'Removed {n_na} stat(s) that referred to missing metrics. df_stats now has {df_stats.shape[0]} rows.')
            df_stats.to_excel(Path(self.output_path, 'stats_df.xlsx'), index=False, engine='openpyxl') if 'stats_df' in dfs else None
            pprint('done.')
        except Exception as e:
            raise ExportError(f'unable to export: {path=}, {self.output_path=}, {dfs=}, {host_names=}, {dropna=}') from e
    
    def save(self, file_name: str = 'racetime_crawler.pkl') -> None:
        save_path = Path(self.output_path, file_name)
        try:
            if not Path(save_path.parent).exists():
                save_path.parent.mkdir(parents=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
            pprint(f'Crawler object saved to: {save_path}')
        except Exception as e:
            raise SaveError(f'unable to save crawler: {save_path=}') from e
    
    @staticmethod
    def load(path: Union[Path, str]) -> 'RacetimeCrawler':
        try:
            pprint(f'Loading Crawler from: {path}', end='...')
            try:
                with open(path, 'rb') as f:
                    crawler = pickle.load(f)
            except:
                print('not found. Creating vanilla crawler', end='...')
                crawler = RacetimeCrawler()
            print('done.')
            pprint(f'Number of race ids in races_df ({len(crawler._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(crawler.race_ids)})') if len(crawler._list_ids_in_races_df()) != len(crawler.race_ids) else None
            pprint(f'Number of columns in races_df ({len(crawler.races_df.columns)}) does not match number of cols in crawler.races_df_cols_cr + _tf ({len(crawler.races_df_cols_cr) + len(crawler.races_df_cols_tf)})') if len(crawler.races_df_cols_cr) + len(crawler.races_df_cols_tf) != len(crawler.races_df.columns) else None
            return crawler
        except Exception as e:
            raise LoadError(f'unable to load crawler: {path=}') from e
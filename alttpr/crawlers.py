# ALTTPR crawlers
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
from alttpr.scanners import DunkaScanner
from alttpr.utils import pprint, pdidx, pprintdesc, get_list, clean_race_info_str, chop_ms
from alttpr.utils import notna, to_tstr, to_dstr, date_from_str, read_var_from_files

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

class VodCrawlerException(Exception):
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


class VodCrawler:
    def __init__(self,
                 gg_path: Union[Path, str],
                 vod_path: Union[Path, str],
                 point_reference: list, 
                 video_ext: Union[str, List[str]] = ".mp4",
                 scanner_ext: Union[str, List[str]] = ".pkl",
                 config_notes_fn: str = 'config_notes.py',
                 config_filter_td_fn: str = 'config_filter_td.py',
                 config_offset_fn: str = 'config_offset.py',
                 config_timestamps_fn: str = 'config_timestamps.py',
                 config_trackerpoints_fn: str = 'config_trackerpoints.py',
                 points_metadata_path: Union[Path, str] = None,
                 final_frame_point_names = ["70|EPP", "71|DPP", "72|THP", "73|PODP", "74|SPP", "75|SWP", "76|TTP", "77|IPP", "78|MMP",  "79|TRP"],
                 ) -> None:
        self.gg_path = Path(gg_path)
        self.vod_path = Path(vod_path)
        self.point_reference = point_reference
        self.video_ext = [video_ext] if isinstance(video_ext, str) else video_ext
        self.scanner_ext = [scanner_ext] if isinstance(scanner_ext, str) else scanner_ext
        self.config_notes_fn=config_notes_fn
        self.config_filter_td_fn=config_filter_td_fn
        self.config_offset_fn=config_offset_fn
        self.config_timestamps_fn=config_timestamps_fn
        self.config_trackerpoints_fn=config_trackerpoints_fn
        self.stutter_sec = 10
        self.get_points_metadata(points_metadata_path)
        self.final_frame_point_names = final_frame_point_names
        
        self.vod_metadata_df = None
        self.vod_names = pd.DataFrame(columns=['label', 'start_ts', 'end_ts', 'offset_ts'])
        # self.times = pd.DataFrame(columns=['label'])
        # self.tracker_names = pd.DataFrame(columns=['label'])
        self.tracker_points = pd.DataFrame(columns=['label', 'tracker_name'])
        self.tracker_labels = pd.DataFrame(columns=['label'])

        self.weekday_dict_DE = {'Monday': 'Montag', 'Tuesday': 'Dienstag', 'Wednesday': 'Mittwoch', 'Thursday': 'Donnerstag', 'Friday': 'Freitag', 'Saturday': 'Samstag', 'Sunday': 'Sonntag'}
        self.weekday_dict_EN = {'0': 'Monday', '1': 'Tuesday', '2': 'Wednesday', '3': 'Thursday', '4': 'Friday', '5': 'Saturday', '6': 'Sunday'}
        self.lang = 'DE'
        self.last_updated: pd.Timestamp = pd.Timestamp.now()

        self.gg = None
        self.scanner_files = []

    def init_gg(self) -> None:
        """Loads the RacetimeCrawler instance."""
        self.gg = RacetimeCrawler.load(self.gg_path)
        pprint(f'--- Crawler host names: {list(self.gg.hosts_df.host_name)}')
        pprint(f'--- Crawler last updated at: {self.gg.last_updated}')
        pprint(f'--- Crawler # races: {len(self.gg.race_ids)}')

    def get_points_metadata(self, points_metadata_path: Union[Path, str] = None) -> None:
        points_metadata_path = points_metadata_path if points_metadata_path else Path(os.getcwd(), 'input/scanner_points_metadata.xlsx')
        self.points_metadata_df = pd.read_excel(points_metadata_path)
    
    def init_vod(self, whitelist: list = None, blacklist: list = None) -> pd.DataFrame:
        """Lists all files matching any file extension specified in vod_fileext_in and vod_fileext_out, and joins them into a dataframe using each file's parent directory as key."""
        whitelist = [x.replace('.pkl', '') for x in whitelist] if whitelist else whitelist
        blacklist = [x.replace('.pkl', '') for x in blacklist] if blacklist else blacklist
        # Recursively list all files matching input extensions
        input_files = []
        for ext in self.video_ext:
            dot_str = '' if ext[0] == '.' else '.'
            input_files.extend(self.vod_path.rglob(f"*{dot_str}{ext}"))

        # Recursively list all files matching output extensions
        output_files = []
        for ext in self.scanner_ext:
            dot_str = '' if ext[0] == '.' else '.'
            output_files.extend(self.vod_path.rglob(f"*{dot_str}{ext}"))

        # Create dataframes for both input and output files
        input_df = pd.DataFrame({
            'vod_name': [f.stem for f in input_files],
            'video_ext': [f.suffix for f in input_files],
            'video_fp': [f for f in input_files],
            'host_name': [f.parts[-3] for f in input_files],
        }).set_index('vod_name')

        notes_lst = read_var_from_files([Path(f.parent, self.config_notes_fn) for f in output_files], var_name='NOTES')
        timestamps_lst = read_var_from_files([
            Path(f.parent, self.config_timestamps_fn) for f in output_files],
            var_name='START_END_TIMESTAMP')
        offset_lst = read_var_from_files([
            Path(f.parent, self.config_offset_fn) for f in output_files],
            var_name='OFFSET_TIMEDELTA')
        itemtracker_box_lst = read_var_from_files([
            Path(f.parent, self.config_trackerpoints_fn) for f in output_files],
            var_name='ITEMTRACKER_BOX')
        itemtracker_points_lst = read_var_from_files([
            Path(f.parent, self.config_trackerpoints_fn) for f in output_files],
            var_name='ITEMTRACKER_POINTS')
        lightworld_map_box_lst = read_var_from_files([
            Path(f.parent, self.config_trackerpoints_fn) for f in output_files],
            var_name='LIGHTWORLD_MAP_BOX')
        lightworld_map_points_lst = read_var_from_files([
            Path(f.parent, self.config_trackerpoints_fn) for f in output_files],
            var_name='LIGHTWORLD_MAP_POINTS')
        darkworld_map_box_lst = read_var_from_files([
            Path(f.parent, self.config_trackerpoints_fn) for f in output_files],
            var_name='DARKWORLD_MAP_BOX')
        darkworld_map_points_lst = read_var_from_files([
            Path(f.parent, self.config_trackerpoints_fn) for f in output_files],
            var_name='DARKWORLD_MAP_POINTS')
        notes_lst = read_var_from_files([Path(f.parent, self.config_notes_fn) for f in output_files], var_name='NOTES')
        filter_td_lst = read_var_from_files([Path(f.parent, self.config_filter_td_fn) for f in output_files], var_name='FILTER_TD_LST')
        output_df = pd.DataFrame({
            'vod_name': [f.stem for f in output_files],
            'scanner_ext': [f.suffix for f in output_files],
            'scanner_fp': [f for f in output_files],
            'start_ts': [pd.Timedelta(x['START_TS']) if x['START_TS'][2] == ':' else np.nan for x in timestamps_lst],
            'end_ts': [pd.Timedelta(x['END_TS']) if x['END_TS'][2] == ':' else np.nan for x in timestamps_lst],
            'offset_ts': [pd.Timedelta(x) if x and x[2] == ':' else np.nan for x in offset_lst],
            'notes': notes_lst,
            'filter_td_lst': filter_td_lst,
            'itemtracker_box': itemtracker_box_lst,
            'itemtracker_points': itemtracker_points_lst,
            'lightworld_map_box': lightworld_map_box_lst,
            'lightworld_map_points': lightworld_map_points_lst,
            'darkworld_map_box': darkworld_map_box_lst,
            'darkworld_map_points': darkworld_map_points_lst,
        }).set_index('vod_name')

        # Combine input and output dataframes
        combined_df = pd.concat([input_df, output_df], axis=1).reset_index()
        combined_df['vod_date'] = [date_from_str(s) for s in combined_df.vod_name]
        
        combined_df = combined_df[combined_df.vod_name.isin(whitelist)] if whitelist else combined_df
        combined_df = combined_df[~combined_df.vod_name.isin(blacklist)] if blacklist else combined_df

        # set class attribute
        self.vod_metadata_df = combined_df.sort_values('vod_date', ascending=False)[[
            'vod_name', 'vod_date', 'scanner_fp', 
            'start_ts', 'end_ts', 'offset_ts', 'notes', 'filter_td_lst',
            'scanner_ext', 'host_name', 'video_fp', 'video_ext',
            'itemtracker_box', 'itemtracker_points', 'lightworld_map_box', 'darkworld_map_box',
        ]]

        # eval metadata
        self.eval_vod_metadata()
    
    def eval_vod_metadata(self):
        no_date_lst = list(self.vod_metadata_df[self.vod_metadata_df.vod_date.isna()].vod_name)
        pprint('Checking vod_date', end='...\t')
        if len(no_date_lst):
            print(f'videos without date ({len(no_date_lst)}/{self.vod_metadata_df.shape[0]}):\t{no_date_lst[0]} ... {no_date_lst[-1]}')
        else:
            print('ok')
        no_scanner_lst = list(self.vod_metadata_df[self.vod_metadata_df.scanner_fp.isna()].vod_name)
        pprint('Checking completion', end='...\t')
        if len(no_scanner_lst):
            print(f'unprocessed videos ({len(no_scanner_lst)}/{self.vod_metadata_df.shape[0]}):\t{no_scanner_lst[0]} ... {no_scanner_lst[-1]}')
        else:
            print('ok')
        no_start_ts_lst = list(self.vod_metadata_df[self.vod_metadata_df.start_ts.isna()].vod_name)
        pprint('Checking start_ts', end='...\t\t')
        if len(no_start_ts_lst):
            print(f'unprocessed videos ({len(no_start_ts_lst)}/{self.vod_metadata_df.shape[0]}):\t{no_start_ts_lst[0]} ... {no_start_ts_lst[-1]}')
        else:
            print('ok')
        no_end_ts_lst = list(self.vod_metadata_df[self.vod_metadata_df.end_ts.isna()].vod_name)
        pprint('Checking end_ts', end='...\t\t')
        if len(no_end_ts_lst):
            print(f'unprocessed videos ({len(no_end_ts_lst)}/{self.vod_metadata_df.shape[0]}):\t{no_end_ts_lst[0]} ... {no_end_ts_lst[-1]}')
        else:
            print('ok')
        no_offset_ts_lst = list(self.vod_metadata_df[self.vod_metadata_df.offset_ts.isna()].vod_name)
        pprint('Checking offset_ts', end='...\t')
        if len(no_offset_ts_lst):
            print(f'unprocessed videos ({len(no_offset_ts_lst)}/{self.vod_metadata_df.shape[0]}):\t{no_offset_ts_lst[0]} ... {no_offset_ts_lst[-1]}')
        else:
            print('ok')

    def crawl(self, max_races: int = None) -> None:
        cols = ['vod_name', 'time', 'point_name', 'label', 'label_change', 'has_change']  # 'time_change', 
        scanner_lst = list(self.vod_metadata_df.dropna(subset='scanner_fp').scanner_fp)
        scanner_lst = scanner_lst[:max_races] if max_races else scanner_lst
        self.raw_df = pd.DataFrame()
        pprint('Crawling scanners')
        for i, fp in enumerate(scanner_lst):
            df_tmp = self.get_raw_data_from_scanner(fp, cols)
            self.raw_df = pd.concat([self.raw_df, df_tmp])
            pprint(f'Successfully extracted data ({i+1}/{len(scanner_lst)})')
        pprint('Crawl completed.')

    def get_raw_data_from_scanner(self, fp: Path, cols, scanner: Union[DunkaScanner] = DunkaScanner) -> None:
        # if 'ALTTP Rando #22.01.2024' in str(fp):
        #     print('now')
        sc = scanner.load(fp)
        df = sc.color_coord_df.copy()
        try:
            df['offset_ts'] = sc.offset_ts
        except:
            df['offset_ts'] = np.nan
            pprint('Unable to extract offset_ts')
        df['vod_name'] = fp.stem
        df['tracker_name'] = df['tracker_name'].str.strip()
        df['point_name'] = df['point_name'].str.strip()
        df['label'] = df['label'].str.strip()
        df['point_name_label'] = df['point_name']
        df['vod_name'] = self._add_vod_names(df[['vod_name', 'start_ts', 'end_ts', 'offset_ts']])
        df['time'] = self._add_times(df.frame)
        df['point_name'] = self._add_tracker_points(df[['point_name', 'tracker_name']])
        df['label_name'] = df['label']
        df['label'] = self._add_tracker_labels(df.label)
        # apply custom filters - in vod time!
        df_m = self.get_metadata().reset_index()
        df_m = df_m[df_m.vod_name==fp.stem]
        filter_td_lst = list(df_m[df_m.vod_name==fp.stem].filter_td_lst)[0]
        if filter_td_lst:
            for td_start, td_end in filter_td_lst:
                mask = ((df.time < pd.Timedelta(td_start).total_seconds()) | (df.time > pd.Timedelta(td_end).total_seconds()))
                df['mask'] = mask
                df = df[mask]
        # apply last_frame_filters
        df = df.set_index(['point_name_label']).join(df[['point_name_label', 'time']].groupby('point_name_label').max().rename(columns={'time': 'time_max'}), rsuffix='_max').reset_index()
        df['mask'] = [True if (l not in self.final_frame_point_names) or (l in self.final_frame_point_names and t==tm) else False for l,t,tm in zip(df.point_name_label, df.time, df.time_max)]
        df = df[df['mask']]
        # find change points
        df_changes = df.sort_values(['point_name', 'frame'])
        df_changes['time_change'] = df_changes.time.shift(-1)
        df_changes.time = df_changes.time_change
        df_changes['point_name_change'] = df_changes.point_name.shift(-1)
        df_changes['label_change'] = df_changes.label.shift(-1)
        df_changes['label_name_change'] = df_changes.label_name.shift(-1)
        df_changes = df_changes[df_changes.point_name == df_changes.point_name_change]
        df_changes = df_changes[df_changes.label != df_changes.label_change]
        df_changes['label_change'] = df_changes['label_change'].astype('int')
        df_changes['has_change'] = 1
        df_changes = df_changes[df_changes.label_name.str.split('-', expand=True)[0] != df_changes.label_name_change.str.split('-', expand=True)[0]]
        # add points that never change
        point_names_missing = [p for p in self.point_reference if p not in list(set(df_changes.point_name_label))]
        df_no_changes = df[df.point_name_label.isin(point_names_missing)]
        df_valids = df_no_changes[['point_name', 'frame']].groupby(['point_name']).min().reset_index()
        df_no_changes = df_no_changes.set_index(['point_name', 'frame']).join(df_valids.set_index(['point_name', 'frame']), how='inner').reset_index()
        df_no_changes['point_name_change'] = np.nan
        df_no_changes['label_change'] = df_no_changes['label']
        df_no_changes['has_change'] = 0
        
        df_changes = df_changes[cols]
        df_no_changes = df_no_changes[cols]
        df = pd.concat([df_changes, df_no_changes], ignore_index=True).sort_values(['time', 'point_name'])

        del sc
        return df[cols]

    def _add_vod_names(self, df: pd.DataFrame) -> pd.DataFrame:
        df_vod_names = df.drop_duplicates().reset_index(drop=True).rename(columns={'vod_name': 'label'})
        if df_vod_names.shape[0] > 1:
            raise VodCrawlerException(f'Ingest error - too many rows: {df_vod_names}')
        df_tmp = self.vod_names[self.vod_names.label == df_vod_names.label[0]]
        df_tmp = df_tmp[df_tmp.start_ts == df_vod_names.start_ts[0]]
        df_tmp = df_tmp[df_tmp.end_ts == df_vod_names.end_ts[0]]
        # df_tmp = self.vod_names[self.vod_names.offset_ts == df_vod_names.offset_ts[0]]
        if df_tmp.shape[0] == 0:
            self.vod_names = pd.concat([self.vod_names, df_vod_names], ignore_index=True)
        else:
            raise ValueError(f'found duplicate entries in df_tmp: {df_tmp}')
        mapping = self._vod_names_dict(inverse=True)
        return list(df.vod_name.replace(mapping))

    def _add_times(self, col: pd.Series) -> list:
        times_lst = [x.seconds for x in col]
        return times_lst
   
    def _add_tracker_points(self, df: pd.DataFrame) -> list:
        df.tracker_name = df.tracker_name.replace({'IT': '1|IT', 'LW': '2|LW', 'DW': '3|DW'})
        df_tracker_points = df.drop_duplicates().sort_values(['tracker_name', 'point_name']).reset_index(drop=True).rename(columns={'point_name': 'label'})
        for i, row in df_tracker_points.iterrows():
            df_tmp = self.tracker_points[self.tracker_points.label == row.label]
            df_tmp = df_tmp[df_tmp.tracker_name == row.tracker_name]
            if df_tmp.shape[0] == 0:
                self.tracker_points = pd.concat([self.tracker_points, df_tracker_points], ignore_index=True)
        mapping = self._tracker_points_dict(inverse=True)
        return list(df.point_name.replace(mapping))

    def _add_tracker_labels(self, col: pd.Series) -> list:
        labels_lst = list(sorted(set(col)))
        for l in labels_lst:
            if l not in self.tracker_labels.label:
                self.tracker_labels = pd.concat([self.tracker_labels, pd.DataFrame([l], columns=['label'])], ignore_index=True)
        mapping = self.tracker_labels.reset_index().set_index('label').to_dict()['index']
        return list(col.replace(mapping))

    def _vod_names_dict(self, inverse=False):
        if inverse:
            map_dict = self.vod_names.reset_index().set_index('label').to_dict()['index']
        else:
            map_dict = self.vod_names.to_dict()['label']
        return map_dict

    def _tracker_points_dict(self, inverse=False):
        if inverse:
            map_dict = self.tracker_points.reset_index().set_index('label').to_dict()['index']
        else:
            map_dict = self.tracker_points.to_dict()['label']
        return map_dict

    def initialize_from_crawler(self, vc: 'VodCrawler') -> None:
        pass

    def sanity_checks(self,
        df_in: pd.DataFrame = None,
        event_count_thres: int = 225,
        remove_optional_items: bool = True,
        vod_names: list = None,
        point_names: list = None,
        last_event_td_thres: str = '00:03:00',
        events_per_item_thres: int = 1,
        grey_to_x_timer_thres: str = '00:05:00',
        revert_count_thres: int = 0,
        # broken_chain_thres: int = 0,
        ) -> pd.DataFrame:        
        df = df_in if df_in else self.get_df()      
        # df = df.set_index('point_name').join(self.points_metadata_df.set_index('point_name').drop(columns=['point_id', 'tracker_name'])).reset_index()
        df = df[df.is_opt==0] if remove_optional_items else df
        df = df[df.vod_name.isin(vod_names)] if vod_names else df
        df = df[df.point_names.isin(point_names)] if point_names else df
        df_points = df[['vod_id', 'vod_name', 'host_name', 'point_name']].drop_duplicates().set_index(['vod_id', 'vod_name', 'host_name', 'point_name'])
        df = df.set_index('vod_id')
        df['is_forfeit'] = [0 if notna(x) else 1 for x in df.entrant_finishtime]
        # for debugging
        # df_tst  = df[df.vod_date.isin(['2024-01-29'])][[  # '2024-07-29' '2024-08-01' '2024-07-28'
        # 'vod_name', 'timer_ha', 
        # 'point_name', 'point_label', 'point_label_update', 'is_stutter', 'has_change', 'tracker_id',
        # 'time', 'timer', 'time_adj', 'timer_adj',
        # ]].sort_values(['time', 'tracker_id', 'point_name',])
        # df_tst.time = [str(t)[-8:] for t in df_tst.time]
        # df_tst.time_adj = [str(t)[-8:] for t in df_tst.time_adj]
        # df_tst.timer = [str(t)[-8:] for t in df_tst.timer]
        # df_tst.timer_adj = [str(t)[-8:] for t in df_tst.timer_adj]
        # df_tst.timer_ha = [str(t)[-8:] for t in df_tst.timer_ha]
        # event_count
        df_metrics = df.copy()
        df_metrics = df_metrics[['vod_name', 'host_name', 'point_name', 'is_forfeit']].reset_index().groupby(['vod_id', 'vod_name', 'host_name', 'is_forfeit']).count().rename(columns={'point_name':'event_count'})
        df_metrics[f'event_count>{event_count_thres}|w1'] = [1 if x > event_count_thres else 0 for x in df_metrics.event_count]
        # crystal_count
        df_crystals = df.copy()
        df_crystals = df_crystals[['vod_name', 'host_name', 'point_name']][[True if 'CRYSTAL' in x else False for x in df.point_label_update]].reset_index().groupby(['vod_id', 'vod_name', 'host_name']).count().rename(columns={'point_name':'crystal_count'})
        df_crystals['crystal_count<>7|w1'] = [1 if x != 7 else 0 for x in df_crystals.crystal_count]
        # last_event_delta
        df_last_event_ts = df.copy()
        df_last_event_ts = df_last_event_ts[['vod_name', 'host_name', 'timer_ha', 'end_ts_ha']].reset_index().groupby(['vod_id', 'vod_name', 'host_name']).max().rename(columns={'timer_ha':'last_event_ts'})
        df_last_event_ts['last_event_delta'] = df_last_event_ts.end_ts_ha - df_last_event_ts.last_event_ts
        df_last_event_ts[f'last_event>{last_event_td_thres}|w1'] = [1 if x > pd.Timedelta(last_event_td_thres) else 0 for x in df_last_event_ts.last_event_delta]
        df_last_event_ts['last_event<0|e1'] = [1 if x < pd.Timedelta('00:00:00') else 0 for x in df_last_event_ts.last_event_delta]
        # point_count, missing_points
        most_frequent_point_count = df[['vod_name', 'point_name']].drop_duplicates().groupby('vod_name').count().mode().point_name[0]
        df_point_cnt = df.copy()
        df_point_cnt = df_point_cnt[['vod_name', 'host_name', 'point_name']].reset_index().drop_duplicates().groupby(['vod_id', 'vod_name', 'host_name']).count().rename(columns={'point_name':'point_name_count'})
        vod_name_lst, host_name_lst, missing_points_lst, new_points_lst = [], [], [], []
        for i, row in df[['vod_name', 'host_name']].drop_duplicates().iterrows():
            df_tmp = df[df.vod_name == row.vod_name]
            df_tmp = df_tmp[df_tmp.host_name == row.host_name]
            vod_name_lst += [row.vod_name]
            host_name_lst += [row.host_name]
            missing_points_lst += [', '.join([x for x in self.point_reference if x not in list(sorted(set(df_tmp.point_name)))])]
            new_points_lst += [', '.join([x for x in list(sorted(set(df_tmp.point_name))) if x not in self.point_reference])]
        df_point_cnt = df_point_cnt.join(pd.DataFrame({'vod_name': vod_name_lst, 'host_name': host_name_lst, 'points_missing': missing_points_lst, 'points_new': new_points_lst}).set_index(['vod_name', 'host_name']))
        df_point_cnt[f'point_name_count<>{most_frequent_point_count}|w1'] = [1 if x != most_frequent_point_count else 0 for x in df_point_cnt.point_name_count]
        # df_point_cnt['point_name_count>141|w2'] = [1 if x > 141 else 0 for x in df_point_cnt.point_name_count]
        # revert_count, broken_chain_count
        df_rev = df.copy()
        df_rev = df_rev[['vod_name', 'host_name', 'point_name', 'point_label', 'point_label_update', 'has_lightblue']]
        df_rev = df_rev[df_rev.point_label != 'CRYSTAL']
        df_rev = df_rev[df_rev.point_label_update != 'CRYSTAL']
        df_1chains = df_rev[['vod_name', 'host_name', 'point_name', 'point_label', 'has_lightblue']].groupby(['vod_name', 'host_name', 'point_name', 'has_lightblue']).count().rename(columns={'point_label': 'revert_count'})
        df_1chains = df_rev.reset_index().set_index(['vod_id', 'vod_name', 'host_name', 'point_name', 'has_lightblue']).join(df_1chains[df_1chains.revert_count==1], how='inner').reset_index()
        df_1chains = df_1chains.set_index(['vod_id', 'vod_name', 'host_name', 'point_name', 'has_lightblue'])[['revert_count']]
        df_1chains.revert_count = 0
        for c in df_rev.columns:
            df_rev[c + '_1'] = df_rev[c].shift(-1)
        df_rev = df_rev[df_rev.vod_name == df_rev.vod_name_1]
        df_rev = df_rev[df_rev.host_name == df_rev.host_name_1]
        df_rev = df_rev[df_rev.point_name == df_rev.point_name_1]
        df_rev['revert_count'] = [1 if pl==plu1 and plu==pl1 else 0 for pl,pl1,plu,plu1 in zip(df_rev.point_label, df_rev.point_label_1, df_rev.point_label_update, df_rev.point_label_update_1)]
        df_rev = df_rev[['vod_name', 'host_name', 'point_name', 'revert_count', 'has_lightblue']].groupby(['vod_id', 'vod_name', 'host_name', 'point_name', 'has_lightblue']).sum()
        df_rev = pd.concat([df_rev, df_1chains])
        df_rev = df_rev.reset_index().set_index(['vod_id', 'vod_name', 'host_name', 'point_name'])
        df_rev_bmb = df_rev.reset_index()
        df_rev_bmb = df_rev_bmb[df_rev_bmb.point_name=='17|BMB'][['vod_id', 'revert_count']]
        df_rev = df_rev.join(df_rev_bmb.set_index('vod_id'), rsuffix='_bmb')
        df_rev.revert_count = [max(0, r-rb) if lb else r for r, rb, lb in zip(df_rev.revert_count, df_rev.revert_count_bmb, df_rev.has_lightblue)]
        df_rev[f'revert_count>{revert_count_thres}|w1'] = [1 if x > revert_count_thres else 0 for x in df_rev.revert_count]
        df_rev = df_rev[[f'revert_count>{revert_count_thres}|w1']].reset_index().groupby(['vod_id', 'vod_name', 'host_name', 'point_name']).sum().reset_index()
        df_rev_point_names = df_rev[df_rev[f'revert_count>{revert_count_thres}|w1']==1].reset_index()
        df_rev_point_names.point_name += ', '
        df_rev_point_names = df_rev_point_names.groupby(['vod_id', 'vod_name', 'host_name'])[['point_name']].sum().rename(columns={'point_name': 'revert_points'})
        df_rev_point_names['revert_points'] = [str(x)[:-2] for x in df_rev_point_names.revert_points]
        df_rev_join = df_rev.set_index(['vod_id', 'vod_name', 'host_name', 'point_name'])
        df_rev.point_name = df_rev.point_name + ', '
        df_rev = df_rev.groupby(['vod_id', 'vod_name', 'host_name']).sum().join(df_rev_point_names).drop(columns='point_name')
        # chain, chain_len
        df_chain = df.copy()
        # df_chain = df_chain[df_chain.vod_name=='ALTTP Rando #29.01.2024']
        df_chain['point_label'] = df_chain['point_label'] + '-'
        df_chain['point_label_update'] = df_chain['point_label_update'] + '-'
        df_chain['time'] = [str(x)[-8:] + '-' for x in df_chain.time]
        df_chain = pdidx(df_chain[['vod_name', 'host_name', 'point_name', 'point_label', 'point_label_update', 'time', 'has_lightblue']].reset_index().groupby(['vod_id', 'vod_name', 'host_name', 'point_name', 'has_lightblue']).agg({
            'point_label': ['first'],
            'point_label_update': ['sum', 'count', 'last'],
            'time': ['sum'],
            })).rename(columns={'point_label|first': 'chain_start', 'point_label_update|last': 'chain_end', 'point_label_update|sum': 'chain', 'point_label_update|count': 'chain_len', 'time|sum': 'chain_time'}).reset_index()
        df_chain['chain_len'] += 1
        df_chain['chain'] = df_chain['chain_start'] + df_chain['chain']
        df_chain['chain_start'] = [x[:-1] for x in df_chain['chain_start']]
        df_chain['chain'] = [x[:-1] for x in df_chain['chain']]
        df_chain['chain_end'] = [x[:-1] for x in df_chain['chain_end']]
        df_chain['chain_time'] = [x[:-1] for x in df_chain['chain_time']]
        df_chain_bmb = df_chain[df_chain.point_name=='17|BMB'][['vod_id', 'chain_len']]
        df_chain = df_chain.set_index('vod_id').join(df_chain_bmb.set_index('vod_id'), rsuffix='_bmb').reset_index()
        df_chain['chain_len_no_bmb'] = [max(0, r-rb+1) if lb else r for r, rb, lb in zip(df_chain.chain_len, df_chain.chain_len_bmb, df_chain.has_lightblue)]
        df_chain_join = df_chain.set_index(['vod_id', 'vod_name', 'host_name', 'point_name'])
        df_chain = df_chain[['vod_id', 'vod_name', 'host_name']].groupby(['vod_id', 'vod_name', 'host_name']).sum()
        # events_per_item
        df_epi = df.copy().reset_index()
        df_epi = df_epi[['vod_id', 'vod_name', 'host_name', 'point_name', 'point_label', 'events_per_item_max', 'has_lightblue']].groupby(['vod_id', 'vod_name', 'host_name', 'point_name', 'events_per_item_max', 'has_lightblue']).count().rename(columns={'point_label':'events_per_item'})
        df_epi = df_epi.reset_index()
        df_epi['events_per_item_delta'] = [max(0, x-y) for x,y in zip(df_epi.events_per_item, df_epi.events_per_item_max)]
        df_epi_bmb = df_epi[df_epi.point_name=='17|BMB'][['vod_id', 'events_per_item_delta']]
        df_epi = df_epi.set_index('vod_id').join(df_epi_bmb.set_index('vod_id'), rsuffix='_bmb').reset_index()
        df_epi['events_per_item_delta'] = [max(0, r-rb) if lb else r for r, rb, lb in zip(df_epi.events_per_item_delta, df_epi.events_per_item_delta_bmb, df_epi.has_lightblue)]
        df_epi[f'epi_delta>{events_per_item_thres}|w1'] = [1 if x > events_per_item_thres else 0 for x in df_epi.events_per_item_delta]
        df_epi['epi_delta>5|e1'] = [1 if x > 5 else 0 for x in df_epi.events_per_item_delta]
        df_epi = df_epi[['vod_id', 'vod_name', 'host_name', 'point_name', f'epi_delta>{events_per_item_thres}|w1', 'epi_delta>5|e1', 'events_per_item', 'events_per_item_delta', 'events_per_item_max']].groupby(['vod_id', 'vod_name', 'host_name', 'point_name', 'events_per_item_delta', 'events_per_item', 'events_per_item_max']).sum()
        df_epi_join = df_epi.reset_index().set_index(['vod_id', 'vod_name', 'host_name', 'point_name'])
        df_epi = df_epi.reset_index()[['vod_id', 'vod_name', 'host_name', 'events_per_item', f'epi_delta>{events_per_item_thres}|w1', 'epi_delta>5|e1']].groupby(['vod_id', 'vod_name', 'host_name']).agg({
            'events_per_item': ['max'], 
            f'epi_delta>{events_per_item_thres}|w1': ['sum'], 
            'epi_delta>5|e1': ['sum'], 
        }).droplevel(1,axis=1).rename(columns={'events_per_item': 'epi_max'})
        # grey_to_x_count
        df_g2x = df.copy().reset_index()
        df_g2x = df_g2x[df_g2x.point_label == 'GREY']
        df_g2x = df_g2x[df_g2x.point_label_update != 'GREY']
        df_g2x[f'grey_to_x_count<={grey_to_x_timer_thres}|w1'] = [1 if x <= pd.Timedelta(grey_to_x_timer_thres) else 0 for x in df_g2x.timer]
        df_g2x[f'grey_to_x_count>{grey_to_x_timer_thres}|e1'] = [1 if x > pd.Timedelta(grey_to_x_timer_thres) else 0 for x in df_g2x.timer]
        df_g2x = df_g2x[['vod_id', 'vod_name', 'host_name', 'point_name', f'grey_to_x_count<={grey_to_x_timer_thres}|w1', f'grey_to_x_count>{grey_to_x_timer_thres}|e1']].groupby(['vod_id', 'vod_name', 'host_name', 'point_name']).sum()
        df_g2x_join = df_g2x.copy()
        df_g2x = df_g2x.reset_index()[['vod_id', 'vod_name', 'host_name', f'grey_to_x_count<={grey_to_x_timer_thres}|w1', f'grey_to_x_count>{grey_to_x_timer_thres}|e1']].groupby(['vod_id', 'vod_name', 'host_name']).sum()
        # build output dfs
        df_metrics = df_metrics.join(df_crystals).join(df_point_cnt).join(df_rev).join(df_chain).join(df_epi).join(df_g2x).join(df_last_event_ts)
        df_metrics['n_issues'] = list(df_metrics[[c for c in df_metrics.columns if '|w' in c or '|e' in c]].sum(axis=1))
        df_metrics['n_warnings'] = list(df_metrics[[c for c in df_metrics.columns if '|w' in c]].sum(axis=1))
        df_metrics['n_errors'] = list(df_metrics[[c for c in df_metrics.columns if '|e' in c]].sum(axis=1))
        df_metrics = df_metrics.fillna(0)
        df_metrics = df_metrics.sort_values(['n_issues', 'event_count'], ascending=False).reset_index()
        # select output columns
        df_metrics = df_metrics[[
            'vod_id', 'vod_name', 'host_name', 'is_forfeit', 'n_issues', 'event_count', 'epi_max',
            'crystal_count<>7|w1', f'point_name_count<>{most_frequent_point_count}|w1', f'revert_count>{revert_count_thres}|w1',
            # f'broken_chain_count>{broken_chain_thres}|w1',
            f'epi_delta>{events_per_item_thres}|w1', 'epi_delta>5|e1',
            'last_event<0|e1', f'grey_to_x_count<={grey_to_x_timer_thres}|w1', f'grey_to_x_count>{grey_to_x_timer_thres}|e1',
            f'last_event>{last_event_td_thres}|w1', f'event_count>{event_count_thres}|w1', 'revert_points',
            'n_warnings', 'n_errors', 'crystal_count', 'point_name_count', 'points_missing',
            'points_new', 'last_event_ts', 'end_ts_ha', 'last_event_delta', 
            ]]
        # join metrics
        df_metrics_points = df_points.join(df_rev_join).join(df_chain_join).join(df_epi_join).join(df_g2x_join).reset_index()
        df_metrics_points = df_metrics_points.fillna(0)
        df_metrics_points['n_issues'] = list(df_metrics_points[[c for c in df_metrics_points.columns if '|w' in c or '|e' in c]].sum(axis=1))
        df_metrics_points['n_warnings'] = list(df_metrics_points[[c for c in df_metrics_points.columns if '|w' in c]].sum(axis=1))
        df_metrics_points['n_errors'] = list(df_metrics_points[[c for c in df_metrics_points.columns if '|e' in c]].sum(axis=1))
        # df_metrics_points_tmp  = df_metrics_points[df_metrics_points.vod_name.isin(['ALTTP Rando #26.02.2023', 'ALTTP Rando #01.08.2024'])]
        # TODO Check co-occurrence of itemtracker- and map-boss-completions
        # TODO color validity checks
        pprint('Completed sanity checks')
        return (df, df_metrics, df_metrics_points)

    def get_df(self, tracker_tag_filter: list = ['IT', 'LW', 'DW'], as_time='td', remove_stutter=[], last_state_only=False, cols: list = None, adjust_time=False, gg_cols: list = ['race_start', 'entrant_name', 'entrant_place', 'entrant_finishtime', 'entrant_rank', 'race_mode_simple', 'race_group', 'race_tournament', 'race_category', ]) -> pd.DataFrame:
        # TODO add filter for crystal tags and boss tags
        # TODO make sure all int ids (point_id, tracker_id!) are integer
        try:
            # collect data
            df = self.get_rawdata()
            df = df.set_index('point_name').join(self.points_metadata_df.set_index('point_name').drop(columns=['point_id', 'tracker_name'])).reset_index()
            df = df.set_index('vod_name')
            df = df[[False if s and t in remove_stutter else True for s,t in zip(df.is_stutter, df.point_name)]] if remove_stutter else df
            df = df[df.tracker_tag.isin(tracker_tag_filter)]
            df = df.join(self.get_metadata(), rsuffix='_md').reset_index()
            host_ids_lst = self.gg.hosts_df[self.gg.hosts_df.host_name.isin(list(set(df.host_name)))].host_id.to_list()
            df_gg = self.gg.get_df(host_ids=host_ids_lst, cols=gg_cols, unique=True, host_rows_only=True).rename(columns={'entrant_name': 'host_name', 'race_start': 'vod_date'})
            df_gg.vod_date = pd.to_datetime(df_gg.vod_date.dt.date)
            df = df.set_index(['vod_date', 'host_name']).join(df_gg.set_index(['vod_date', 'host_name']), how='left').reset_index()
            if last_state_only:
                df_filter = df[['vod_id', 'time', 'point_name']].groupby(['vod_id', 'point_name']).max()
                df = df.set_index(['vod_id', 'point_name']).join(df_filter, rsuffix='_max')
                df = df[df.time == df.time_max]
            # timestamps
            df['start_ts_ha'] = [chop_ms(t) if notna(t) else chop_ms(tmd) for t, tmd in zip(df.start_ts, df.start_ts_md)]
            df['end_ts_ha'] = [chop_ms(t) if notna(t) else chop_ms(tmd) for t, tmd in zip(df.end_ts, df.end_ts_md)]
            df['duration_ha'] = df['end_ts_ha'] - df['start_ts_ha']
            df['duration_delta'] = df.entrant_finishtime - df.duration_ha                
            df['offset_ts_ha'] = [chop_ms(o) if notna(o) else chop_ms(omd) for o, omd in zip(df.offset_ts, df.offset_ts_md)]
            df['offset_ts_ha'] = df['offset_ts_ha'].fillna(pd.Timedelta('00:00:00'))
            df['timer'] = [t + o.seconds - s.seconds for t,o,s in zip(df.time, df.offset_ts_ha, df.start_ts_ha)] # TODO subtract start_ts
            df['time_adj'] = round(df.time + (df.time / df.end_ts_ha.dt.seconds) * df.duration_delta.dt.seconds, 0)
            df['timer_adj'] = round(df.timer + (df.timer / (df.end_ts_ha.dt.seconds + df.offset_ts_ha.dt.seconds)) * df.duration_delta.dt.seconds, 0)
            df['time_ha'] = [ta if notna(ta) else t for t,ta in zip(df.time, df.time_adj)]
            df['timer_ha'] = [ta if notna(ta) else t for t,ta in zip(df.timer, df.timer_adj)]
            if as_time in ['td', 'dt']:
                df.time = [pd.Timedelta(seconds=x) for x in df.time]
                df.timer = [pd.Timedelta(seconds=x) for x in df.timer]
                df.time_adj = [pd.Timedelta(seconds=x) if notna(x) else np.nan for x in df.time_adj]
                df.timer_adj = [pd.Timedelta(seconds=x) if notna(x) else np.nan for x in df.timer_adj]
                df.time_ha = [pd.Timedelta(seconds=x) if notna(x) else np.nan for x in df.time_ha]
                df.timer_ha = [pd.Timedelta(seconds=x) if notna(x) else np.nan for x in df.timer_ha]
            if as_time in ['dt']:
                df.time = [pd.to_datetime(' '.join(d.strftime('%Y-%m-%d'), str(td)[-8:])) for d,td in zip(df.vod_date, df.time)]
                df.timer = [pd.to_datetime(' '.join(d.strftime('%Y-%m-%d'), str(td)[-8:])) for d,td in zip(df.vod_date, df.timer)]
                df.time_adj = [pd.to_datetime(' '.join(d.strftime('%Y-%m-%d'), str(td)[-8:])) if notna(x) else np.nan for d,td in zip(df.vod_date, df.time_adj)]
                df.timer_adj = [pd.to_datetime(' '.join(d.strftime('%Y-%m-%d'), str(td)[-8:])) if notna(x) else np.nan for d,td in zip(df.vod_date, df.timer_adj)]
                df.time_ha = [pd.to_datetime(' '.join(d.strftime('%Y-%m-%d'), str(td)[-8:])) if notna(x) else np.nan for d,td in zip(df.vod_date, df.time_ha)]
                df.timer_ha = [pd.to_datetime(' '.join(d.strftime('%Y-%m-%d'), str(td)[-8:])) if notna(x) else np.nan for d,td in zip(df.vod_date, df.timer_ha)]
        except Exception as e:
            raise VodCrawlerException(f'unable to retrieve data') from e
        df = df[cols] if cols else df
        return df

    def get_rawdata(self) -> pd.DataFrame:
        df = self.raw_df.rename(columns={'label': 'point_label', 'label_change': 'point_label_update'})
        df = df.set_index('vod_name').join(self.get_vod_names())
        df = df.set_index('point_name').join(self.get_tracker_points())
        df = df.set_index('point_label').join(self.get_tracker_labels())
        df = df.set_index('point_label_update').join(self.get_tracker_labels(), rsuffix='_update')
        df = df.sort_values(['vod_id', 'tracker_name', 'point_name', 'time']).reset_index(drop=True)
        df['is_stutter'] = self.get_tag_stutter(df)
        df.point_label = df.point_label.str.split('-', expand=True)[0]
        df.point_label_update = df.point_label_update.str.split('-', expand=True)[0]
        return df
    
    def get_tag_stutter(self, df: pd.DataFrame) -> pd.DataFrame:
        # TODO add event count per point_name as removal criterion (ALTTP Rando #19.05.2024)
        df_m = df[['time', 'vod_id', 'point_name', 'point_label', 'point_label_update']]
        df_m.point_label = df_m.point_label.str.split('-', expand=True)[0]
        df_m.point_label_update = df_m.point_label_update.str.split('-', expand=True)[0]
        for c in df_m.columns:
            df_m[c+'_pre'] = df_m[c].shift(1)
            df_m[c+'_post'] = df_m[c].shift(-1)
        # df_m = df_m.dropna()
        df_m['no_change'] = df_m.point_label == df_m['point_label_update']
        df_m['time_pre'] = abs(df_m.time - df_m['time_pre']) <= self.stutter_sec
        df_m['time_post'] = abs(df_m.time - df_m['time_post']) <= self.stutter_sec
        df_m['vod_id_pre'] = df_m.vod_id == df_m['vod_id_pre']
        df_m['vod_id_post'] = df_m.vod_id == df_m['vod_id_post']
        df_m['point_name_pre'] = df_m.point_name == df_m['point_name_pre']
        df_m['point_name_post'] = df_m.point_name == df_m['point_name_post']
        df_m['close_to_pre'] = df_m[['time_pre', 'vod_id_pre', 'point_name_pre']].all(axis=1)
        df_m['close_to_post'] = df_m[['time_post', 'vod_id_post', 'point_name_post']].all(axis=1)
        df_m['is_time_paired'] = df_m[['close_to_pre', 'close_to_post']].any(axis=1)
        df_m.point_label = df_m.point_label + ' - ' + df_m.point_label_update
        df_m.point_label_pre = (df_m.point_label_update_pre + ' - ' + df_m.point_label_pre) == df_m.point_label
        df_m.point_label_post = (df_m.point_label_update_post + ' - ' + df_m.point_label_post) == df_m.point_label
        df_m['point_label_pre'] = df_m[['point_label_pre', 'vod_id_pre', 'point_name_pre']].all(axis=1)
        df_m['point_label_post'] = df_m[['point_label_post', 'vod_id_post', 'point_name_post']].all(axis=1)
        df_m['is_label_paired'] = df_m[['point_label_pre', 'point_label_post']].any(axis=1)
        # df_m = df_m[df_m.point_name.isin(['11|SWO', '12|SLD'])][['time', 'is_time_paired', 'vod_id', 'point_name', 'point_label', 'is_label_paired', 'point_label_pre', 'point_label_post']]  # .drop(columns=['time_pre', 'time_post', 'vod_id_pre', 'vod_id_post', 'point_name_pre', 'point_name_post', 'close_to_pre', 'close_to_post'])
        df_m = df_m[['time', 'is_time_paired', 'vod_id', 'point_name', 'point_label', 'is_label_paired', 'point_label_pre', 'point_label_post', 'no_change']]  # .drop(columns=['time_pre', 'time_post', 'vod_id_pre', 'vod_id_post', 'point_name_pre', 'point_name_post', 'close_to_pre', 'close_to_post'])
        df_m['is_stutter'] = df_m[['is_time_paired', 'point_label_pre']].all(axis=1)
        df_m['is_stutter'] = df_m[['is_stutter', 'no_change']].any(axis=1)
        # df_m['is_stutter'] = ~df_m.is_paired

        return list(df_m['is_stutter'])

    def get_vod_names(self) -> pd.DataFrame:
        df = self.vod_names.rename(columns={'label': 'vod_name'})
        df = df.reset_index().rename(columns={'index': 'vod_id'})
        return df
    
    def get_tracker_points(self) -> pd.DataFrame:
        df = self.tracker_points.rename(columns={'label': 'point_name'})
        df['point_id'] = df.point_name.str.split('|', expand=True)[0]
        df['point_tag'] = df.point_name.str.split('|', expand=True)[1]
        df['tracker_id'] = df.tracker_name.str.split('|', expand=True)[0]
        df['tracker_tag'] = df.tracker_name.str.split('|', expand=True)[1]
        df = df[['point_name', 'point_id', 'point_tag', 'tracker_name', 'tracker_id', 'tracker_tag']]
        return df

    def get_tracker_labels(self) -> pd.DataFrame:
        df = self.tracker_labels.rename(columns={'label': 'point_label'})
        return df

    def get_metadata(self, extended=False, gg_cols: list = ['race_start', 'entrant_name', 'entrant_place', 'entrant_finishtime', 'entrant_rank', 'race_mode_simple', 'race_group', 'race_tournament', 'race_category', ]) -> pd.DataFrame:
        df = self.vod_metadata_df.set_index('vod_name')
        if extended:
            df = df.join(self.get_vod_names().set_index('vod_name'), rsuffix='_md').reset_index()
            host_ids_lst = self.gg.hosts_df[self.gg.hosts_df.host_name.isin(list(set(df.host_name)))].host_id.to_list()
            df_gg = self.gg.get_df(host_ids=host_ids_lst, cols=gg_cols, unique=True, host_rows_only=True).rename(columns={'entrant_name': 'host_name', 'race_start': 'vod_date'})
            df_gg.vod_date = pd.to_datetime(df_gg.vod_date.dt.date)
            df = df.set_index(['vod_date', 'host_name']).join(df_gg.set_index(['vod_date', 'host_name']), how='left').reset_index()
            df = df.set_index('vod_name')
        return df
    
    def refresh_transforms(self) -> None:
        pass

    def get_metrics(self):
        try:
            pass
        except:
            raise VodCrawlerException(f'unable to retrieve metrics') from e

    def get_stats(self):
        try:
            pass
        except Exception as e:
            raise VodCrawlerException(f'unable to retrieve stats') from e
                
    def add_races(self):
        '''Add one or more races and pull data'''
        pass
    
    def set_output_path(self, path: Union[Path, str]) -> None:
        path = Path(path)
        if len(path.suffix) > 0:
            raise RacetimeCrawlerException('self.output_path can\'t have file extension.')
        self.output_path = path

    def export(self, path: Union[Path, str] = None, dfs: List[str] = ['hosts_df', 'races_df', 'metrics_df', 'stats_df', 'race_mode_map_df', 'race_mode_simple_map_df', 'race_tournament_map_df'], host_names: Union[str, List[str]] = [], dropna=False) -> None:
        try:
           pass
        except Exception as e:
            raise VodCrawlerException(f'unable to export') from e
    
    def save(self, file_name: str = 'vod_crawler.pkl') -> None:
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
    def load(path: Union[Path, str]) -> 'VodCrawler':
        try:
            pprint(f'Loading Crawler from: {path}', end='...')
            try:
                with open(path, 'rb') as f:
                    crawler = pickle.load(f)
            except:
                print('not found.')
            print('done.')
            # pprint(f'Number of VODs in vod_df ({len(crawler._list_ids_in_races_df())}) does not match number of ids in self.race_ids ({len(crawler.race_ids)})') if len(crawler._list_ids_in_races_df()) != len(crawler.race_ids) else None
            # pprint(f'Number of columns in races_df ({len(crawler.races_df.columns)}) does not match number of cols in crawler.races_df_cols_cr + _tf ({len(crawler.races_df_cols_cr) + len(crawler.races_df_cols_tf)})') if len(crawler.races_df_cols_cr) + len(crawler.races_df_cols_tf) != len(crawler.races_df.columns) else None
            return crawler
        except Exception as e:
            raise LoadError(f'unable to load crawler: {path=}') from e


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
        race_url_blacklist = list(pd.read_excel(self.parse_template_path, sheet_name=['race-blacklist'])['race-blacklist']['race_href'])
        race_url_whitelist = list(pd.read_excel(self.parse_template_path, sheet_name=['race-whitelist'])['race-whitelist']['race_href'])
        race_ids_to_crawl = [i for i in self.race_ids if i not in self._list_ids_in_races_df()]  # /alttpr/banzai-redboom-2002
        race_ids_to_crawl = list(sorted(set(race_ids_to_crawl + race_url_whitelist)))
        race_ids_already_crawled = [i for i in self.race_ids if i in self._list_ids_in_races_df()]
        pprint(f'{len(race_ids_already_crawled)} races were already parsed.')
        if len(race_ids_to_crawl) > 0:
            pprint(f'Crawling {len(race_ids_to_crawl)} new races (max 30 shown): {race_ids_to_crawl[:30]}')
            race_data_lst = []
            for r in tqdm(race_ids_to_crawl, desc=pprintdesc('Crawling races')):
                if r in race_url_blacklist:
                    warn_str = f"WARNING! Race-URL {r} is blacklisted. Skipped. Check {self.parse_template_path}"
                    tqdm.write(warn_str)
                else:
                    race_data_lst += [self._get_race_data(r)]
            new_races_df = pd.concat(race_data_lst, ignore_index=True)
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
            # if url == '/smr/fearless-threemuskateers-0765':
            #     print('now')
            dfs = pd.read_excel(self.parse_template_path, sheet_name=['racetime-user-class-options'])
            race_url = self.base_url + url
            response = self._scrape(race_url)
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
                    if e.find('span', {"class": "name"}).text == '(deleted user)':
                        href_user = '/user/deleteduser/deleteduser'
                    else:
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
            raise CrawlError(f'Failed to execute _get_race_data() for entrant \'{e}\' on: "{race_url}"') from e

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
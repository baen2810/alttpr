# Schedulers - Schedulers for alttpr
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

"""Schedulers"""

# TODO create tests

from pathlib import Path
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import pprint
from logging.handlers import RotatingFileHandler
from typing import List, Union

import os
import schedule
import time
import logging
import warnings

DEBUG = True  # bool(os.environ.get('ALTTPR_DEBUG'))  # some of the crawlers can print debug info
pprint('DEBUG mode active') if DEBUG else None
if DEBUG:
    pass

class SchedulerException(Exception):
    """Base class for exceptions."""


class SchedulerError(SchedulerException):
    """Parsing a url failed."""


class UnsupportedFormatError(SchedulerException):
    """File format is not supported."""



class DailyScheduler:
    def __init__(self, runtime: str, host_ids: Union[str, List[str]], private_folder: str, public_folder: str, private_dfs: List[str], crawler_file: str, max_retries: int = 3):
        self.runtime = runtime
        self.host_ids = [host_ids] if isinstance(host_ids, str) else host_ids
        self.private_folder = private_folder
        self.public_folder = public_folder
        self.private_dfs = private_dfs
        self.crawler_file = crawler_file
        self.max_retries = max_retries
        self.setup_logging()

    def setup_logging(self):
        logs_dir = Path(os.getcwd(), 'logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / 'daily_scheduler.log'
        
        logging.basicConfig(
            handlers=[RotatingFileHandler(log_file, maxBytes=10000000, backupCount=5)],
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        self.logger = logging.getLogger('DailyScheduler')

    def run_crawler(self):
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f'Starting crawl attempt {attempt + 1}')
                gg = RacetimeCrawler.load(Path(self.crawler_file))
                pprint(f'--- Crawler host names: {list(gg.hosts_df.host_name)}')
                pprint(f'--- Crawler last updated at: {gg.last_updated}')
                pprint(f'--- Crawler # races: {len(gg.race_ids)}')
                
                gg.crawl(host_ids=self.host_ids, n_pages=1)
                
                pprint('--- Saving crawler data')
                gg.set_output_path(Path(self.private_folder))
                gg.export()
                gg.save()
                
                pprint('--- Exporting racer datasets')
                for host_name in gg.hosts_df.host_name:
                    gg.set_output_path(Path(self.public_folder, host_name))
                    gg.export(dfs=self.private_dfs, host_names=host_name)
                
                pprint('Finished.')
                self.logger.info('Crawl completed successfully')
                break
            except Exception as e:
                self.logger.error(f'Error during crawl attempt {attempt + 1}: {e}', exc_info=True)
                if attempt + 1 == self.max_retries:
                    self.logger.error('Max retries reached. Giving up.')
                else:
                    time.sleep(60)  # Wait for a minute before retrying

    def run(self):
        schedule.every().day.at(self.runtime).do(self.run_crawler)
        while True:
            schedule.run_pending()
            time.sleep(1)

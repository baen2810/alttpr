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
# add prints to retry attempts including error msg for faster analysis

from pathlib import Path
from alttpr.crawlers import RacetimeCrawler
from alttpr.utils import pprint, pprintdesc
from logging.handlers import RotatingFileHandler
from typing import List, Union
from datetime import datetime, timedelta

import os
import schedule
import time
import logging
import traceback
import keyboard

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
    def __init__(self, name: str, runtimes: Union[str, List[str]], crawler: RacetimeCrawler, private_folder: Union[str, Path],
                 public_folder: Union[str, Path], private_dfs: List[str], max_retries: int = 3, n_pages: int = None):
        self.name = name
        self.runtimes = [runtimes] if isinstance(runtimes, str) else runtimes
        self.private_folder = Path(private_folder)
        self.public_folder = Path(public_folder)
        self.private_dfs = private_dfs
        self.max_retries = max_retries
        self.running = True
        self.n_pages=n_pages

        # init crawler
        self.crawler = crawler
        self.crawler.set_output_path(self.private_folder)
        
        # log
        self.setup_logging()

    def setup_logging(self):
        logs_dir = Path(os.getcwd(), 'logs', self.name)
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = logs_dir / 'daily_scheduler.log'
        pprint(f'Logging to: {log_file}')
        
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
                try:
                    pprint(f'--- Crawler host names: {list(self.crawler.hosts_df.host_name)}', start='\n')
                    pprint(f'--- Crawler last updated at: {self.crawler.last_updated}')
                    pprint(f'--- Crawler # races: {len(self.crawler.race_ids)}')
                except:
                    pprint(f'--- Found vanilla crawler')
                self.crawler.crawl(self.n_pages)
                
                pprint('--- Saving crawler data')
                self.crawler.export()
                self.crawler.save()
                
                pprint('--- Exporting racer datasets')
                for host_name in self.crawler.hosts_df.host_name:
                    self.crawler.set_output_path(self.public_folder / host_name)
                    self.crawler.export(dfs=self.private_dfs, host_names=host_name, dropna=True)
                
                pprint('Finished.')
                self.logger.info('Crawl completed successfully')
                break
            except Exception as e:
                self.logger.error(f'Error during crawl attempt {attempt + 1}: {e}', exc_info=True)
                print(f'Error during crawl attempt {attempt + 1}: {e}')
                print(traceback.format_exc())
                if attempt + 1 == self.max_retries:
                    self.logger.error('Max retries reached. Giving up.')
                else:
                    time.sleep(60)  # Wait for a minute before retrying

    def run(self):
        self.quit_flag = False

        def on_quit():
            self.quit_flag = True

        keyboard.on_press_key("q", lambda _: on_quit())  # Register event listener for 'q' key

        for runtime in self.runtimes:
            schedule.every().day.at(runtime).do(self.run_crawler)
        
        pprint("Scheduler started. Scheduled times: {}".format(", ".join(self.runtimes)))

        try:
            while self.running:
                if self.quit_flag:
                    print("\nScheduler aborted by user.")
                    self.running = False
                    break

                next_run = schedule.next_run()
                time_left = next_run - datetime.now()
                # Remove milliseconds
                time_left = str(time_left).split('.')[0]
                print(pprintdesc(f"\rTime left until next run (press 'q' to abort): {time_left}"), end='', flush=True)
                schedule.run_pending()
                time.sleep(1)
        finally:
            logging.shutdown()
            print("Logger has been shut down.")

        # Unhook the 'q' key event when done
        keyboard.unhook_all()

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
<<<<<<< HEAD
from alttpr.utils import pprint, pprintdesc, get_workspace_vars
=======
from alttpr.utils import pprint, pprintdesc
>>>>>>> 1fb60a3 (Merge branch 'feat/add-parametrized-metrics')
from logging.handlers import RotatingFileHandler
from typing import List, Union
from datetime import datetime, timedelta

import os
import schedule
import time
import logging
import traceback
import keyboard
import requests

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
<<<<<<< HEAD
                 public_folder: Union[str, Path], private_dfs: List[str], max_retries: int = 3, n_pages: int = None, mailer=None, workspace: str = 'not available', logs_dir:Union[str, Path]=None):
=======
                 public_folder: Union[str, Path], private_dfs: List[str], max_retries: int = 3, n_pages: int = None):
>>>>>>> 1fb60a3 (Merge branch 'feat/add-parametrized-metrics')
        self.name = name
        self.runtimes = [runtimes] if isinstance(runtimes, str) else runtimes
        self.private_folder = Path(private_folder)
        self.public_folder = Path(public_folder)
        self.private_dfs = private_dfs
        self.max_retries = max_retries
        self.running = True
        self.n_pages=n_pages
<<<<<<< HEAD
        self.mailer=mailer
        self.workspace=workspace
        self.logs_dir=Path(os.getcwd(), 'logs', self.name) if logs_dir is None else Path(logs_dir)
=======
>>>>>>> 1fb60a3 (Merge branch 'feat/add-parametrized-metrics')

        # init crawler
        self.crawler = crawler
        self.crawler.set_output_path(self.private_folder)
        
        # log
        self.setup_logging()

    def setup_logging(self):
        # logs_dir = Path(os.getcwd(), 'logs', self.name)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
<<<<<<< HEAD
        log_file = self.logs_dir / 'daily_scheduler.log'
=======
        log_file = logs_dir / 'daily_scheduler.log'
>>>>>>> 1fb60a3 (Merge branch 'feat/add-parametrized-metrics')
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
<<<<<<< HEAD
                # raise ValueError()
                msg = f'Starting crawl attempt {attempt + 1}'
                self.logger.info(msg)
                pprint(msg, start='\n')
=======
                self.logger.info(f'Starting crawl attempt {attempt + 1}')
>>>>>>> 1fb60a3 (Merge branch 'feat/add-parametrized-metrics')
                try:
                    pprint(f'--- Crawler host names: {list(self.crawler.hosts_df.host_name)}', start='\n')
                    pprint(f'--- Crawler last updated at: {self.crawler.last_updated}')
                    pprint(f'--- Crawler # races: {len(self.crawler.race_ids)}')
                except:
                    pprint(f'--- Found vanilla crawler')
                self.crawler.crawl(self.n_pages)
                
                pprint('--- Saving crawler data')
                self.crawler.save()
                self.crawler.export()
                
                msg = '--- Exporting racer datasets'
                self.logger.info(msg)
                pprint(msg)                
                for host_name in self.crawler.hosts_df.host_name:
                    self.logger.info(self.public_folder / host_name)
                    self.crawler.set_output_path(self.public_folder / host_name)
                    self.crawler.export(dfs=self.private_dfs, host_names=host_name, dropna=True)
                
                msg = 'Successfully completed crawl'
                self.logger.info(msg)
                pprint(msg)
                self.mailer.send(subject=msg, msg=f'Workspace:\n{self.workspace}')
                break
            except Exception as e:
<<<<<<< HEAD
                traceback_msg = traceback.format_exc()
                error_msg = f'Error during crawl attempt {attempt + 1}: {e}'
                self.logger.error(error_msg + f'\n\nWorkspace:{self.workspace}\n', exc_info=True)
                print(error_msg)
                print(traceback_msg)
                print(self.workspace)
=======
                self.logger.error(f'Error during crawl attempt {attempt + 1}: {e}', exc_info=True)
                print(f'Error during crawl attempt {attempt + 1}: {e}')
                print(traceback.format_exc())
>>>>>>> 1fb60a3 (Merge branch 'feat/add-parametrized-metrics')
                if attempt + 1 == self.max_retries:
                    msg = 'Max retries reached. Giving up.'
                    self.logger.error(msg)
                    print(msg)
                    if self.mailer:
                        subject = f'{e.__class__.__name__} during crawl attempt {attempt + 1}'
                        msg = f'The following error occurred:\n{e}\nTraceback:\n{traceback_msg}\nWorkspace:\n{self.workspace}'
                        self.mailer.send(subject, msg)
                else:
                    time.sleep(60*attempt + 1)  # Wait for a minute before retrying

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
<<<<<<< HEAD

class MailgunMailer:
    def __init__(self, domain: str, api_key: str, sender: str, recipients: Union[str, list]):
        self.domain = domain
        self.api_key = api_key
        self.sender = sender
        self.recipients = [recipients] if isinstance(recipients, str) else recipients

    def send(self, subject: str, msg: str):
        response = requests.post(
            f"https://api.mailgun.net/v3/{self.domain}/messages",
            auth=("api", self.api_key),
            data={
                "from": self.sender,
                "to": self.recipients,
                "subject": subject,
                "text": msg
                })
        if response.status_code == 200:
            pprint(f'Successfully delivered Mailgun-Mail to "{self.recipients}"')
        else:
            pprint(f'Unable to send Mailgun-Mail. Error code "{response.status_code}"')
        return response
=======
>>>>>>> 1fb60a3 (Merge branch 'feat/add-parametrized-metrics')

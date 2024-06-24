# DunkaScanner - a Scanner for DunkaTracker in VOD
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

"""Dunka Scanner"""

# TODO tbd

# from __future__ import annotations
# from collections.abc import Callable, Iterator
# from functools import reduce
# from sys import stderr
from datetime import datetime as dt
from typing import Any
from warnings import warn
from typing import Union, Optional, Tuple, Dict, List
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
import cv2
import pickle
import warnings
import shutil
import zipfile

DEBUG = True  # bool(os.environ.get('ALTTPR_DEBUG'))  # some of the crawlers can print debug info
pprint('DEBUG mode active') if DEBUG else None
NAN_VALUE = np.nan
if DEBUG:
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=FutureWarning)
    pd.options.mode.chained_assignment = None
class DunkaScannerException(Exception):
    """Base class for exceptions."""


class DunkaScannerError(DunkaScannerException):
    """Parsing a url failed."""


class UnsupportedFormatError(DunkaScannerException):
    """File format is not supported."""


import os
from pathlib import Path
from typing import Union, Optional
import cv2
import pandas as pd
from tqdm import tqdm
from alttpr.utils import pprintdesc

# default coordinates for the itemtracker box
DEFAULT_ITEMTRACKER_BOX = (1418, 0, 245, 224)

# default coordinates for the lightworld map tracker box
DEFAULT_LIGHTWORLD_MAP_BOX = (1413, 229, 251, 261)

# default coordinates for the darkworld map tracker box
DEFAULT_DARKWORLD_MAP_BOX = (1670, 229, 249, 260)

# default tracking point coordinates for the itemtracker box
# absolute pixels relative to top left corner of box
DEFAULT_ITEMTRACKER_POINTS = {
    # itemtracker row 1
    # '01|SWO': (200, 200),
    # '02|BMR': (124, 16),
    # '03|BOW': (88, 16),
    # '04|HKS': (156, 25),
    # '05|BMB': (193, 20),
    # '06|MSR': (218, 6),
    # '07|POW': (237, 22),
    # # itemtracker row 2
    # '11|MNP': (55, 50),
    # '12|FIR': (86, 53),
    # '13|ICR': (121, 53),
    # '14|BBS': (148, 46),
    # '15|ETH': (187, 55),
    # '16|QUK': (228, 55),
    # # itemtracker row 3
    # '21|EP': (15, 72),
    # '22|LMP': (89, 85),
    # '23|HMR': (122, 78),
    # '24|SVL': (148, 70),
    # '25|FLU': (164, 89),
    # '26|BUG': (190, 73),
    # '27|BOK': (229, 92),
    # # itemtracker row 4
    # '31|DP': (7, 119),
    # '32|BTL': (89, 113),
    # '33|SOM': (122, 120),
    # '34|BYR': (157, 120),
    # '35|CAP': (188, 118),
    # '36|MIR': (227, 106),
    # # itemtracker row 5
    '41|TH': (9, 156),
    '42|BTS': (80, 154),
    '43|GLV': (114, 150),
    '44|FLP': (163, 143),
    '45|MAG': (183, 158),
    '46|AG1': (219, 132),
    '47|AG2': (236, 151),
    }

# default tracking point coordinates for the lightworld map tracker box
# absolute pixels relative to top left corner of box
DEFAULT_LIGHTWORLD_MAP_POINTS = {
    'P1': (13, 7),
    'P2': (31, 22),
    'P3': (47, 33)
    }

# default tracking point coordinates for the darkworld map tracker box
# absolute pixels relative to top left corner of box
DEFAULT_DARKWORLD_MAP_POINTS = {
    'P1': (6, 6),
    'P2': (83, 39),
    'P3': (142, 39)
    }

# Define the predefined RGB values for each color label
DEFAULT_COLOR_LABELS_MAP_TRACKERS = {
    'DEFAULT': {
        'RED': (230, 0, 0),
        'GREEN': (20, 255, 20),
        'LIGHTBLUE': (40, 180, 240),
        'DARKBLUE': (0, 0, 240),
        'ORANGE': (240, 160, 20),
        'YELLOW': (245, 255, 15),
        'GREY': (128, 128, 128)
    },
}

# Define the predefined RGB values for each item tracker label
DEFAULT_COLOR_LABELS_ITEM_TRACKER = {
    'HS|NONE': (230, 0, 0),
    'HS|ACTIVE': (20, 255, 20),
    'SWORD|NONE': (40, 180, 240),
    'SWORD|FIGHTER': (0, 0, 240),
    'SWORD|MASTER': (240, 160, 20),
    'SWORD|TEMPERED': (245, 255, 15),
    'BOW': {
        'FALSE': (0, 0, 0),
        'SIMPLE': (255, 0, 0),
        'SILVERS': (0, 255, 0),
    },
    'HKS': {
        'FALSE': (0, 0, 0),  # black
        'TRUE': (230, 0, 0),  # red
    },
    'DEFAULT': {
        'CLF_ERR': (0, 0, 0),
    }
}

class DunkaScanner:
    def __init__(self, input_video_path: Union[str, Path], output_path: Union[str, Path], 
                 start_ts: Union[int, str, pd.Timestamp, pd.Timedelta] = 0, 
                 end_ts: Optional[Union[int, str, pd.Timestamp, pd.Timedelta]] = None, 
                 frames_per_second: int = 1, 
                 itemtracker_box: Tuple[int, int, int, int] = None,
                 lightworld_map_box: Tuple[int, int, int, int] = None, 
                 darkworld_map_box: Tuple[int, int, int, int] = None,
                 itemtracker_points: dict = None,
                 lightworld_map_tracker_points: dict = None,
                 darkworld_map_tracker_points: dict = None) -> None:
        """
        Initialize the DunkaScanner with video and extraction parameters.

        Parameters:
        - input_video_path: Path to the input video file.
        - output_path: Path to the folder to save the extracted frames.
        - start_ts: Start time or frame (can be int, str in 'HH:MM:SS:FF' or 'HH:MM:SS' format, pd.Timestamp, or pd.Timedelta).
        - end_ts: End time or frame (can be int, str in 'HH:MM:SS:FF' or 'HH:MM:SS' format, pd.Timestamp, or pd.Timedelta).
        - frames_per_second: Number of frames to extract per second of video (default is 1).
        - itemtracker_box: Coordinates of the itemtracker box (x, y, width, height).
        - lightworld_map_box: Coordinates of the lightworld map box (x, y, width, height).
        - darkworld_map_box: Coordinates of the darkworld map box (x, y, width, height).
        - itemtracker_points: Dictionary of points within the itemtracker box.
        - lightworld_map_tracker_points: Dictionary of points within the lightworld map box.
        - darkworld_map_tracker_points: Dictionary of points within the darkworld map box.
        """
        self.input_video_path = Path(input_video_path)
        self.output_path = Path(output_path)
        self.frames_per_second = frames_per_second
        self.itemtracker_box = itemtracker_box
        self.lightworld_map_box = lightworld_map_box
        self.darkworld_map_box = darkworld_map_box
        self.itemtracker_points = itemtracker_points
        self.lightworld_map_tracker_points = lightworld_map_tracker_points
        self.darkworld_map_tracker_points = darkworld_map_tracker_points
        self.color_coord_df = pd.DataFrame()
        
        self.frames = []  # Initialize the frames attribute

        # Open the video file to read its properties
        cap = cv2.VideoCapture(str(self.input_video_path))
        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        # Get the video's frames per second (fps)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:  # Handle cases where FPS might be zero or unavailable
            raise ValueError("Error: Could not get video FPS.")

        # Get the total number of frames in the video
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Calculate the video length in seconds
        self.video_length = pd.to_timedelta(self.total_frames / self.fps, unit='s')

        # Raise an error if the video is longer than 24 hours
        if self.video_length > pd.Timedelta(hours=24):
            raise UnsupportedFormatError("Video length exceeds 24 hours and is not supported.")

        # Get the video width and height
        self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate the file size in KB
        self.video_size = self.input_video_path.stat().st_size / 1024  # in KB

        # Convert start_ts and end_ts to timedelta
        self.start_ts = self._convert_to_timedelta(start_ts)
        self.end_ts = self._convert_to_timedelta(end_ts) if end_ts is not None else self.video_length

        # Ensure end_ts does not exceed the video length
        if self.end_ts > self.video_length:
            self.end_ts = self.video_length

        # Release the video capture object
        cap.release()

        pprint(f'Video: {self.input_video_path}')
        pprint(f'FPS: {self.fps}')
        pprint(f'Size (kB): {self.video_size}')
        pprint(f'Video length: {self.video_length.components.hours:02}:{self.video_length.components.minutes:02}:{self.video_length.components.seconds:02}')
        pprint(f'Video resolution: {self.video_width}x{self.video_height}')
        pprint(f'Number of item tracker points: {len(self.itemtracker_points)}')
        pprint(f'Number of lightworld map tracker points: {len(self.lightworld_map_tracker_points)}')
        pprint(f'Number of darkworld map tracker points: {len(self.darkworld_map_tracker_points)}')

    @staticmethod
    def select_timestamp(video_path: Union[str, Path], title: str = 'Select Timestamp', default_time: str = '00:04:30') -> Optional[pd.Timedelta]:
        """
        Allow the user to visually select a timestamp in the video.
        Returns the selected timestamp as pd.Timedelta or None if aborted.

        Parameters:
        - video_path: Path to the input video file.
        - title: Title of the window.
        - default_time: Default timestamp to seek initially.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        default_frame = int(pd.to_timedelta(default_time).total_seconds() * fps)
        current_frame = default_frame

        jump_to_frame = min(int(1.5 * 3600 * fps), total_frames - 1)  # Jump to 01:30:00 or end of video

        window_title = title + " (e: select, w: forward 30s, s: back 30s, d: forward 1s, a: back 1s, f: jump to 01:30:00, y: back 1 min, c: forward 1 min, r: back 1 frame, t: forward 1 frame, q: quit)"

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            # Calculate the current timestamp
            current_time = pd.to_timedelta(current_frame / fps, unit='s')
            timestamp_str = f"{current_time.components.hours:02}:{current_time.components.minutes:02}:{current_time.components.seconds:02}"

            # Display the timestamp on the frame
            cv2.putText(frame, timestamp_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_title, frame)

            # Ensure you click on the OpenCV window to bring it into focus
            key = cv2.waitKey(0)

            if key == ord('e'):
                break
            elif key == ord('w'):
                current_frame = min(current_frame + int(30 * fps), total_frames - 1)
            elif key == ord('s'):
                current_frame = max(0, current_frame - int(30 * fps))
            elif key == ord('d'):
                current_frame = min(current_frame + int(fps), total_frames - 1)
            elif key == ord('a'):
                current_frame = max(0, current_frame - int(fps))
            elif key == ord('f'):
                current_frame = jump_to_frame
            elif key == ord('y'):
                current_frame = max(0, current_frame - int(60 * fps))
            elif key == ord('c'):
                current_frame = min(current_frame + int(60 * fps), total_frames - 1)
            elif key == ord('r'):
                current_frame = max(0, current_frame - 1)
            elif key == ord('t'):
                current_frame = min(current_frame + 1, total_frames - 1)
            elif key == ord('q'):  # 'q' to quit
                cap.release()
                cv2.destroyAllWindows()
                pprint(f"Selection aborted.")
                return None

        cap.release()
        cv2.destroyAllWindows()

        selected_time = pd.to_timedelta(current_frame / fps, unit='s')
        pprint(f'Selected {title.lower()}: {selected_time.components.hours:02}:{selected_time.components.minutes:02}:{selected_time.components.seconds:02}')
        return selected_time

    @staticmethod
    def select_box(video_path: Union[str, Path], timestamp: pd.Timedelta, title: str = 'Select Box',
                step_size_horizontal: float = 0.01, step_size_vertical: float = 0.01,
                default_box: Tuple[int, int, int, int] = (800, 100, 200, 200),
                tracking_points: Dict[str, Tuple[int, int]] = None) -> Optional[Tuple[Tuple[int, int, int, int], Dict[str, Tuple[int, int]]]]:
        """
        Allow the user to select a rectangular area on a reference frame from the video.
        Returns the selected rectangle as (x, y, width, height) or None if aborted.

        Parameters:
        - video_path: Path to the input video file.
        - timestamp: Timestamp to seek the reference frame.
        - title: Title of the window.
        - step_size_horizontal: Step size for horizontal movement as a percentage of rectangle width.
        - step_size_vertical: Step size for vertical movement as a percentage of rectangle height.
        - default_box: Default box coordinates (x, y, width, height).
        - tracking_points: Dictionary of point names and their coordinates within the box.
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise ValueError("Error: Could not open video.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(timestamp.total_seconds() * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Error: Could not read frame.")

        frame_height, frame_width = frame.shape[:2]
        x, y, w, h = default_box
        points = tracking_points or {}

        step_size_horizontal = 0.0  # Default to 0%, which is one pixel
        step_size_vertical = 0.0  # Default to 0%, which is one pixel

        selected_point_name = None
        point_names = list(points.keys())
        point_index = 0 if point_names else -1

        window_title = title + " (e: confirm, q: quit, w: up, s: down, a: left, d: right, r: increase height, f: decrease height, c: increase width, x: decrease width, t: increase step, g: decrease step, i: point up, k: point down, j: point left, l: point right, u: previous point, o: next point)"

        while True:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

            if point_index >= 0:
                selected_point_name = point_names[point_index]
                cv2.putText(temp_frame, f"Selected Point: {selected_point_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            step_size_text = f"Step size: {int(step_size_horizontal * 100)}%"
            cv2.putText(temp_frame, step_size_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            for point_name, (px, py) in points.items():
                point_x = int(x + px)
                point_y = int(y + py)
                cv2.circle(temp_frame, (point_x, point_y), 2, (255, 255, 255), -1)
                cv2.putText(temp_frame, point_name, (point_x + 5, point_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window_title, temp_frame)

            key = cv2.waitKey(0) & 0xFF

            horizontal_step = max(1, int(w * step_size_horizontal)) if step_size_horizontal > 0 else 1
            vertical_step = max(1, int(h * step_size_vertical)) if step_size_vertical > 0 else 1

            if key == ord('e'):
                break
            elif key == ord('w'):
                y = max(0, y - vertical_step)
            elif key == ord('s'):
                y = min(frame_height - h, y + vertical_step)
            elif key == ord('a'):
                x = max(0, x - horizontal_step)
            elif key == ord('d'):
                x = min(frame_width - w, x + horizontal_step)
            elif key == ord('r'):
                scale_factor = (h + vertical_step) / h
                h = min(frame_height - y, h + vertical_step)
                points = {k: (px, py * scale_factor) for k, (px, py) in points.items()}
            elif key == ord('f'):
                scale_factor = (h - vertical_step) / h
                h = max(1, h - vertical_step)
                points = {k: (px, py * scale_factor) for k, (px, py) in points.items()}
            elif key == ord('c'):
                scale_factor = (w + horizontal_step) / w
                w = min(frame_width - x, w + horizontal_step)
                points = {k: (px * scale_factor, py) for k, (px, py) in points.items()}
            elif key == ord('x'):
                scale_factor = (w - horizontal_step) / w
                w = max(1, w - horizontal_step)
                points = {k: (px * scale_factor, py) for k, (px, py) in points.items()}
            elif key == ord('t'):
                step_size_horizontal = min(1.0, step_size_horizontal + 0.1)
                step_size_vertical = min(1.0, step_size_vertical + 0.1)
            elif key == ord('g'):
                step_size_horizontal = max(0.0, step_size_horizontal - 0.1)
                step_size_vertical = max(0.0, step_size_vertical - 0.1)
            elif key == ord('i') and point_index >= 0:
                points[selected_point_name] = (points[selected_point_name][0], max(0, points[selected_point_name][1] - vertical_step))
            elif key == ord('k') and point_index >= 0:
                points[selected_point_name] = (points[selected_point_name][0], min(h, points[selected_point_name][1] + vertical_step))
            elif key == ord('j') and point_index >= 0:
                points[selected_point_name] = (max(0, points[selected_point_name][0] - horizontal_step), points[selected_point_name][1])
            elif key == ord('l') and point_index >= 0:
                points[selected_point_name] = (min(w, points[selected_point_name][0] + horizontal_step), points[selected_point_name][1])
            elif key == ord('u') and point_names:
                point_index = (point_index - 1) % len(point_names)
            elif key == ord('o') and point_names:
                point_index = (point_index + 1) % len(point_names)
            elif key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                pprint(f"Selection aborted.")
                return None, None

        cap.release()
        cv2.destroyAllWindows()

        selected_box = (x, y, w, h)
        pprint(f'{title} box: {selected_box}')
        pprint(f'{title} points: {points}')
        return selected_box, points

    def _convert_to_timedelta(self, time: Union[int, str, pd.Timestamp, pd.Timedelta]) -> pd.Timedelta:
        if isinstance(time, int):
            return pd.to_timedelta(time / self.fps, unit='s')
        elif isinstance(time, pd.Timestamp):
            return pd.to_timedelta(time - time.normalize())
        elif isinstance(time, pd.Timedelta):
            return time
        elif isinstance(time, str):
            parts = time.split(':')
            if len(parts) == 4:  # HH:MM:SS:FF format
                hours, minutes, seconds, frames = map(int, parts)
                return pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m') + pd.to_timedelta(seconds, unit='s') + pd.to_timedelta(frames / self.fps, unit='s')
            elif len(parts) == 3:  # HH:MM:SS format
                return pd.to_timedelta(time)
            else:
                raise ValueError("String time format not recognized.")
        else:
            raise ValueError("Unsupported time format")
    
    def scan(self, table_position: Optional[str] = 'upper left', save_nth_frame: int = 10) -> None:
        """
        Extract frames from the video, save them as image files, and collect RGB and HSI values for specified points.
        The frames are extracted based on the set parameters and saved in the output folder.

        Parameters:
        - table_position: Optional; position of the table with point names and RGB values. 
                          Choices: 'lower left', 'lower center', 'lower right', 'upper left', 'upper center', 'upper right', None.
                          Default is 'upper left'. If None, the table will not be drawn.
        - save_nth_frame: Optional; save every nth frame. Default is 10. If 1, save every frame.
        """
        video_name = self.input_video_path.stem
        frames_output_path = self.output_path / video_name / 'frames'
        
        self._check_and_delete_existing_output(frames_output_path)

        if not frames_output_path.exists():
            frames_output_path.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(self.input_video_path))

        interval, frame_indices = self._calculate_frame_extraction_intervals()

        data = []

        for i, frame_index in enumerate(tqdm(frame_indices, desc=pprintdesc('Extracting frames'))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break

            frame_time, frame_timestamp = self._calculate_frame_timestamp(frame_index)
            frame_data = self._extract_rgb_values(frame, frame_time)

            self._overlay_boxes_and_points(frame)

            if table_position:
                self._draw_table(frame, frame_data, table_position)

            if save_nth_frame > 0 and i % save_nth_frame == 0:
                frame_filename = frames_output_path / f"{frame_timestamp}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                self.frames.append(str(frame_filename))

            data.extend(frame_data)

        cap.release()
        
        self.color_coord_df = pd.DataFrame(data)
        self.color_coord_df.frame = [str(f)[-8:] for f in self.color_coord_df.frame]
        self.color_coord_df['start_ts'] = str(self.start_ts)[-8:]
        self.color_coord_df['end_ts'] = str(self.end_ts)[-8:]

        self._zip_output_folder(frames_output_path)

    def _calculate_frame_extraction_intervals(self) -> Tuple[int, range]:
        interval = int(self.fps // self.frames_per_second)
        if interval == 0:
            interval = 1
        start_frame = int(self.start_ts.total_seconds() * self.fps)
        end_frame = int(self.end_ts.total_seconds() * self.fps)
        frame_indices = range(start_frame, min(end_frame + 1, self.total_frames), interval)
        return interval, frame_indices

    def _calculate_frame_timestamp(self, frame_index: int) -> Tuple[pd.Timedelta, str]:
        frame_time = pd.to_timedelta(frame_index / self.fps, unit='s')
        hours, remainder = divmod(frame_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        frames = int((frame_time.microseconds / 1e6) * self.fps)
        frame_timestamp = f"{hours:02d}{minutes:02d}{seconds:02d}{frames:02d}"
        return frame_time, frame_timestamp

    def _extract_rgb_values(self, frame: np.ndarray, frame_time: pd.Timedelta) -> List[Dict]:
        data = []
        for box, points, tracker_name, color_labels in [
            (self.itemtracker_box, self.itemtracker_points, 'IT', DEFAULT_COLOR_LABELS_ITEM_TRACKER),
            (self.lightworld_map_box, self.lightworld_map_tracker_points, 'LW', DEFAULT_COLOR_LABELS_MAP_TRACKERS),
            (self.darkworld_map_box, self.darkworld_map_tracker_points, 'DW', DEFAULT_COLOR_LABELS_MAP_TRACKERS)
        ]:
            if box:
                x, y, w, h = box
                for point_name, (px, py) in points.items():
                    point_x = int(x + px)
                    point_y = int(y + py)
                    B, G, R = frame[point_y, point_x].astype(float)
                    I = (R + G + B) / 3
                    S = 1 - min(R, G, B) / I if I > 0 else 0
                    H = 0
                    if S != 0:
                        R_prime = R / (R + G + B)
                        G_prime = G / (R + G + B)
                        B_prime = B / (R + G + B)
                        H = 0.5 * ((R_prime - G_prime) + (R_prime - B_prime)) / ((R_prime - G_prime)**2 + (R_prime - B_prime) * (G_prime - B_prime))**0.5
                        H = np.degrees(np.arccos(H))
                        if B > G:
                            H = 360 - H
                    color_label = self._classify_RGB_into_color_labels(point_name, R, G, B, H, S, I, color_labels)
                    data.append({
                        'frame': frame_time,
                        'point_name': point_name,
                        'R': R,
                        'G': G,
                        'B': B,
                        'H': H,
                        'S': S,
                        'I': I,
                        'tracker_name': tracker_name,
                        'color_label': color_label
                    })
        return data

    def _overlay_boxes_and_points(self, frame: np.ndarray) -> None:
        for box, points, label in [
            (self.itemtracker_box, self.itemtracker_points, "IT"),
            (self.lightworld_map_box, self.lightworld_map_tracker_points, "LW"),
            (self.darkworld_map_box, self.darkworld_map_tracker_points, "DW")
        ]:
            if box:
                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                if label != "items":
                    cv2.putText(frame, label, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                for point_name, (px, py) in points.items():
                    point_x = int(x + px)
                    point_y = int(y + py)
                    cv2.circle(frame, (point_x, point_y), 2, (255, 255, 255), -1)
                    cv2.putText(frame, point_name, (point_x + 5, point_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_table(self, frame: np.ndarray, frame_data: List[Dict], table_position: str) -> None:
        table_positions = {
            'lower left': (10, frame.shape[0] - 10),
            'lower center': ((frame.shape[1] - 200) // 2, frame.shape[0] - 10),
            'lower right': (frame.shape[1] - 210, frame.shape[0] - 10),
            'upper left': (10, 10),
            'upper center': ((frame.shape[1] - 200) // 2, 10),
            'upper right': (frame.shape[1] - 210, 10)
        }
        x_base, y_base = table_positions.get(table_position, (10, 10))

        table_text = "Tracker    Point    R    G    B    Color\n"
        table_text += "\n".join([f"{entry['tracker_name']}  {entry['point_name']}  {int(entry['R']):3d}  {int(entry['G']):3d}  {int(entry['B']):3d}  {entry['color_label']}" for entry in frame_data])
        lines = table_text.split('\n')

        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x_base, y_base + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def _convert_to_timedelta(self, time: Union[int, str, pd.Timestamp, pd.Timedelta]) -> pd.Timedelta:
        if isinstance(time, int):
            return pd.to_timedelta(time / self.fps, unit='s')
        elif isinstance(time, pd.Timestamp):
            return pd.to_timedelta(time - time.normalize())
        elif isinstance(time, pd.Timedelta):
            return time
        elif isinstance(time, str):
            parts = time.split(':')
            if len(parts) == 4:
                hours, minutes, seconds, frames = map(int, parts)
                return pd.to_timedelta(hours, unit='h') + pd.to_timedelta(minutes, unit='m') + pd.to_timedelta(seconds, unit='s') + pd.to_timedelta(frames / self.fps, unit='s')
            elif len(parts) == 3:
                return pd.to_timedelta(time)
            else:
                raise ValueError("String time format not recognized.")
        else:
            raise ValueError("Unsupported time format")

    def _classify_RGB_into_color_labels(self, point_name: str, R, G, B, H, S, I, color_labels):
        """
        Classify the RGB and/or HSI values of a point into predefined color labels.

        Parameters:
        - point_name: Name of the point.
        - R, G, B: Red, Green, Blue color values of the point.
        - H, S, I: Hue, Saturation, Intensity values of the point.
        - color_labels: Dictionary of predefined color labels and their RGB values.

        Returns:
        - label: The classified color label.
        """
        point_color_labels = color_labels.get(point_name, color_labels['DEFAULT'])
        min_distance = float('inf')
        label = None

        for color, (R_ref, G_ref, B_ref) in point_color_labels.items():
            distance = np.sqrt((R - R_ref) ** 2 + (G - G_ref) ** 2 + (B - B_ref) ** 2)
            if distance < min_distance:
                min_distance = distance
                label = color

        return label
    
    def _check_and_delete_existing_output(self, frames_output_path: Path) -> None:
        if frames_output_path.exists() and frames_output_path.is_dir():
            shutil.rmtree(frames_output_path)
        zip_output_path = frames_output_path.with_suffix('.zip')
        if zip_output_path.exists() and zip_output_path.is_file():
            zip_output_path.unlink()

    def _zip_output_folder(self, frames_output_path: Path) -> None:
        zip_output_path = frames_output_path.with_suffix('.zip')
        with zipfile.ZipFile(zip_output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(frames_output_path):
                for file in files:
                    file_path = Path(root) / file
                    zipf.write(file_path, file_path.relative_to(frames_output_path.parent))
        shutil.rmtree(frames_output_path)

    def export(self) -> None:
        """
        Export the color_coord_df DataFrame to the first sheet 'data' of an .xlsx file.
        On the second sheet 'params', save all other class attributes.
        """
        output_file = self.output_path / self.input_video_path.stem / f"{self.input_video_path.stem}.xlsx"
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            self.color_coord_df.to_excel(writer, sheet_name='data', index=False)

            params = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
            params_df = pd.DataFrame(list(params.items()), columns=['Attribute', 'Value'])
            params_df.to_excel(writer, sheet_name='params', index=False)

        pprint(f"Data exported to {output_file}")
    
    def save(self) -> None:
        """
        Save the DunkaScanner object to the output directory as a .pkl file.
        """
        with open(self.output_path / self.input_video_path.stem / f"{self.input_video_path.stem}.pkl", 'wb') as file:
            pickle.dump(self, file)
        pprint(f"Scanner object saved to {self.output_path / f'{self.input_video_path.stem}.pkl'}")

    @staticmethod
    def load(file_path: Union[str, Path]) -> 'DunkaScanner':
        """
        Load an existing DunkaScanner object from a .pkl file.

        Parameters:
        - file_path: Path to the .pkl file.

        Returns:
        - DunkaScanner object.
        """
        with open(file_path, 'rb') as file:
            scanner = pickle.load(file)
        pprint(f"Scanner object loaded from {file_path}")
        return scanner

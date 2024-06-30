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

# TODO add internal consistency for output data, e.g. check consistency of boss status from map and item tracker
# TODO handle reference count overflow error when picking the very first frame as starting point (ALTTP Rando #09.01.2024.mp4)
# TODO fix issue when selecting end time
# TODO add third scanner points for boomerang 
# TODO ensure DEBUG is read from env var

from typing import Union, Optional, Tuple, Dict, List
from pathlib import Path
from tqdm import tqdm
from alttpr.utils import pprint, pprintdesc
from alttpr.config_color_labels_default import COLOR_LABELS_MAP_TRACKERS, COLOR_LABELS_ITEM_TRACKER
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
        self.fps = int(cap.get(cv2.CAP_PROP_FPS) + 1)
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
        self.video_size = round((self.input_video_path.stat().st_size / 1024) / 1024, 1)  # in MB

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
        pprint(f'Size (MB): {self.video_size}')
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

        fps = int(cap.get(cv2.CAP_PROP_FPS) + 1)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        default_frame = int(pd.to_timedelta(default_time).total_seconds() * fps)
        current_frame = default_frame

        jump_to_frame = min(int(1.5 * 3600 * fps), total_frames - 2*fps)  # Jump to 01:30:00 or end of video

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

        step_size_horizontal = 0.01  # Default to 1%
        step_size_vertical = 0.01  # Default to 1%

        selected_point_name = None
        point_names = list(points.keys())
        point_index = 0 if point_names else -1
        label_display_mode = 'ID'

        window_title = title + " (e: confirm, q: quit, w: up, s: down, a: left, d: right, r: increase height, f: decrease height, c: increase width, x: decrease width, t: increase step, g: decrease step, i: point up, k: point down, j: point left, l: point right, u: previous point, o: next point, b: forward 15 min, v: back 15 min, p: toggle label, y: save to file)"
        keybindings_text = ["Keybindings: e: confirm, q: quit, w: up, s: down, a: left, d: right, r: increase height, f: decrease height",
                            "c: increase width, x: decrease width, t: increase step, g: decrease step, i: point up, k: point down",
                            "j: point left, l: point right, u: previous point, o: next point, b: forward 15 min, v: back 15 min, p: toggle label, y: save to file"]

        while True:
            temp_frame = frame.copy()
            cv2.rectangle(temp_frame, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

            if point_index >= 0:
                selected_point_name = point_names[point_index]
                cv2.putText(temp_frame, f"Selected Point: {selected_point_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            step_size_text = f"Step size: {int(step_size_horizontal * 100)}%"
            cv2.putText(temp_frame, step_size_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            frame_data = []
            for point_name, (px, py) in points.items():
                point_x = int(x + px)
                point_y = int(y + py)
                cv2.circle(temp_frame, (point_x, point_y), 2, (255, 255, 255), -1)

                # Determine the label to display based on the display mode
                if label_display_mode == 'None':
                    label = ''
                elif label_display_mode == 'ID':
                    label = point_name.split('|')[0]
                else:
                    label = point_name

                cv2.putText(temp_frame, label, (point_x + 5, point_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

                B, G, R = frame[point_y, point_x]
                if point_name in COLOR_LABELS_ITEM_TRACKER.keys():  # TODO this may break when not all items are explicit keys but use default instead
                    color_labels = COLOR_LABELS_ITEM_TRACKER
                else:
                    color_labels = COLOR_LABELS_MAP_TRACKERS
                color_label = DunkaScanner._classify_RGB_into_color_labels(point_name, R, G, B, color_labels)
                frame_data.append({
                    "tracker_name": "Tracker",
                    "R": str(R).zfill(3),
                    "G": str(G).zfill(3),
                    "B": str(B).zfill(3),
                    "label": color_label.ljust(7),
                    "point_name": point_name.ljust(7),
                })

            DunkaScanner._draw_table(temp_frame, frame_data, 'center_left')

            # Draw the keybindings text in a white box with black text
            y_offset = frame_height - 50
            for keybinding_text in keybindings_text:
                (text_width, text_height), _ = cv2.getTextSize(keybinding_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                text_x, text_y = 10, y_offset
                cv2.rectangle(temp_frame, (text_x - 5, text_y - text_height - 5), (text_x + text_width + 5, text_y + 5), (255, 255, 255), cv2.FILLED)
                cv2.putText(temp_frame, keybinding_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                y_offset += text_height + 10

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
            elif key == ord('b'):
                frame_count = min(frame_count + int(15 * 60 * fps), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break
            elif key == ord('v'):
                frame_count = max(0, frame_count - int(15 * 60 * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                ret, frame = cap.read()
                if not ret:
                    break
            elif key == ord('p'):
                if label_display_mode == 'Full':
                    label_display_mode = 'ID'
                elif label_display_mode == 'ID':
                    label_display_mode = 'None'
                else:
                    label_display_mode = 'Full'
            elif key == ord('y'):
                save_path = os.path.join(os.getcwd(), 'tracker_box_and_points.txt')
                with open(save_path, 'w') as f:
                    f.write(f'{title.replace(" ", "_").upper()}_BOX = {{\n')
                    f.write(f'    "x": {x},\n    "y": {y},\n    "w": {w},\n    "h": {h}\n}}\n')
                    f.write(f'{title.replace(" ", "_").upper()}_POINTS = {{\n')
                    for point_name, coords in points.items():
                        f.write(f'    "{point_name}": {coords},\n')
                    f.write('}\n')
                pprint(f"Saved box and points to {save_path}")
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
    
    def scan(self, table_position: str = 'upper_left', save_nth_frame: int = 10) -> None:
        """
        Extract frames from the video and save them as image files.
        The frames are extracted based on the set parameters and saved in the output folder.

        Parameters:
        - table_position: Position of the table (default is 'upper_left').
        - save_nth_frame: Save every nth frame (default is 10).
        """
        # Check and clear output directory
        save_nth_frame = save_nth_frame * self.fps
        video_name = self.input_video_path.stem  # Get the video file name without extension
        frames_output_path = self.output_path / video_name / 'frames'
        self._check_and_delete_existing_output(frames_output_path)

        # Create the output subdirectory for frames
        if not frames_output_path.exists():
            frames_output_path.mkdir(parents=True, exist_ok=True)

        # Open the video file
        cap = cv2.VideoCapture(str(self.input_video_path))

        # Calculate the interval for frame extraction
        interval, frame_indices = self._calculate_frame_extraction_intervals()

        data = []

        for frame_index in tqdm(frame_indices, desc=pprintdesc('Extracting frames')):
            # Set the position of the next frame to read
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                break

            # Calculate the timestamp for the frame
            frame_time, frame_timestamp = self._calculate_frame_timestamp(frame_index)

            # Extract RGB values for points before overlaying
            frame_data = self._extract_rgb_values(frame, frame_time)

            # Overlay the boxes and points
            self._overlay_boxes_and_points(frame)

            # Draw the tables with RGB values
            
            self._draw_table(frame, frame_data, table_position,
                             font_scale=0.25, line_height=8)

            # Append data for dataframe
            data.extend(frame_data)

            # Construct the output file path
            frame_filename = frames_output_path / f"{frame_timestamp}.jpg"
            if len(data) % save_nth_frame == 0:
                cv2.imwrite(str(frame_filename), frame)

            # Add the filename to the list of extracted frames
            self.frames.append(str(frame_filename))

        # Release the video capture object
        cap.release()

        # Create the dataframe
        self.color_coord_df = pd.DataFrame(data, columns=['frame', 'tracker_name', 'point_name', 'R', 'G', 'B', 'label'])
        self.color_coord_df['R'] = self.color_coord_df['R'].astype(float).astype(int)
        self.color_coord_df['G'] = self.color_coord_df['G'].astype(float).astype(int)
        self.color_coord_df['B'] = self.color_coord_df['B'].astype(float).astype(int)
        self.color_coord_df['point_name'] = self.color_coord_df['point_name'].str.strip()
        self.color_coord_df['label'] = self.color_coord_df['label'].str.strip()
        self.color_coord_df['start_ts'] = self.start_ts
        self.color_coord_df['end_ts'] = self.end_ts

        # Zip the output directory and clean up
        pprint('Zipping frames')
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
            (self.itemtracker_box, self.itemtracker_points, 'IT', COLOR_LABELS_ITEM_TRACKER),
            (self.lightworld_map_box, self.lightworld_map_tracker_points, 'LW', COLOR_LABELS_MAP_TRACKERS),
            (self.darkworld_map_box, self.darkworld_map_tracker_points, 'DW', COLOR_LABELS_MAP_TRACKERS)
        ]:
            if box:
                x, y, w, h = box
                for point_name, (px, py) in points.items():
                    point_x = int(x + px)
                    point_y = int(y + py)
                    B, G, R = frame[point_y, point_x].astype(float)
                    B, G, R = np.clip([B, G, R], 0, 255)  # Ensure values are within [0, 255]
                    color_label = self._classify_RGB_into_color_labels(point_name, R, G, B, color_labels)
                    data.append({
                        'frame': frame_time,
                        'point_name': point_name.ljust(7),
                        'R': str(R).zfill(3),
                        'G': str(G).zfill(3),
                        'B': str(B).zfill(3),
                        'tracker_name': tracker_name.ljust(7),
                        'label': color_label.ljust(7)
                    })
        return data

    def _overlay_boxes_and_points(self, frame: np.ndarray, labels: bool = False) -> None:
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
                    if labels:
                        cv2.putText(frame, point_name, (point_x + 5, point_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def _draw_table(frame, frame_data, table_position: str = 'upper_left',
                    font_scale: float = 0.5, line_height: int = 15, font_thickness: int = 1):
        """
        Draw a table with RGB values on the frame.
        """
        # Define the desired order of columns
        column_order = ["tracker_name", "R", "G", "B", "label", "point_name"]

        # Create the table header
        table_text = " ".join([col.capitalize() for col in column_order]) + "\n"

        # Create the table rows based on the column order
        for item in frame_data:
            row_text = " ".join([str(item[col]) for col in column_order])
            table_text += row_text + "\n"

        font = cv2.FONT_HERSHEY_SIMPLEX

        lines = table_text.split('\n')

        # Determine the position of the table
        if table_position == 'upper_left':
            x, y = 10, 10
        elif table_position == 'upper_center':
            x, y = frame.shape[1] // 2 - 200, 10
        elif table_position == 'upper_right':
            x, y = frame.shape[1] - 400, 10
        elif table_position == 'lower_left':
            x, y = 10, frame.shape[0] - (line_height * len(lines)) - 10
        elif table_position == 'lower_center':
            x, y = frame.shape[1] // 2 - 200, frame.shape[0] - (line_height * len(lines)) - 10
        elif table_position == 'lower_right':
            x, y = frame.shape[1] - 400, frame.shape[0] - (line_height * len(lines)) - 10
        elif table_position == 'center_left':
            x, y = 10, frame.shape[0] // 2 - (line_height * len(lines)) // 2
        elif table_position == 'center':
            x, y = frame.shape[1] // 2 - 200, frame.shape[0] // 2 - (line_height * len(lines)) // 2
        elif table_position == 'center_right':
            x, y = frame.shape[1] - 400, frame.shape[0] // 2 - (line_height * len(lines)) // 2
        else:
            return  # Do not draw the table if the position is None

        # Draw the white box
        box_width = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines]) + 10
        box_height = line_height * len(lines) + 10
        cv2.rectangle(frame, (x - 5, y - 15), (x + box_width, y + box_height - 15), (255, 255, 255), cv2.FILLED)

        # Draw the table text on the frame
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (x, y + i * line_height), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


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

    @classmethod
    def _classify_RGB_into_color_labels(cls, point_name: str, R, G, B, color_labels):
        """
        Classify the RGB and/or HSI values of a point into predefined color labels.

        Parameters:
        - point_name: Name of the point.
        - R, G, B: Red, Green, Blue color values of the point.
        - color_labels: Dictionary of predefined color labels and their RGB values.

        Returns:
        - label: The classified color label.
        """
        point_color_labels = color_labels.get(point_name, color_labels['DEFAULT'])
        min_distance = float('inf')
        label = None

        for color, (R_ref, G_ref, B_ref) in point_color_labels.items():
            distance = int(np.sqrt((int(R) - int(R_ref)) ** 2 + (int(G) - int(G_ref)) ** 2 + (int(B) - int(B_ref)) ** 2))
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
        # Define a function to export a dictionary to a text file
        def export_dict_to_txt(dict_name, dictionary, delete=False, fn="config_trackerpoints.py"):
            txt_file = self.output_path / self.input_video_path.stem / fn
            # Ensure the directory exists
            txt_file.parent.mkdir(parents=True, exist_ok=True)

            # Delete the file if it exists
            if txt_file.exists() and delete:
                txt_file.unlink()
            # Write the dictionary to the file, appending if the file exists
            with open(txt_file, 'a') as f:
                f.write(f"{dict_name} = {{\n")
                for key, value in dictionary.items():
                    f.write(f'    "{key}": {value},\n')
                f.write('}\n')
            
            pprint(f"{dict_name} exported to {txt_file}")
        
        # Define a function to export a tuple to a text file
        def export_tuple_to_txt(tuple_name, tuple_value, delete=False, fn="config_trackerpoints.py"):
            txt_file = self.output_path / self.input_video_path.stem / fn
            # Ensure the directory exists
            txt_file.parent.mkdir(parents=True, exist_ok=True)
            # Delete the file if it exists
            if txt_file.exists() and delete:
                txt_file.unlink()
            # Write the tuple to the file, appending if the file exists
            with open(txt_file, 'a') as f:
                f.write(f"{tuple_name} = {tuple_value}\n")
            
            pprint(f"{tuple_name} exported to {txt_file}")

        export_tuple_to_txt('ITEMTRACKER_BOX', self.itemtracker_box, delete=True)
        export_tuple_to_txt('LIGHTWORLD_MAP_BOX', self.lightworld_map_box)
        export_tuple_to_txt('DARKWORLD_MAP_BOX', self.darkworld_map_box)
        export_dict_to_txt('ITEMTRACKER_POINTS', self.itemtracker_points)
        export_dict_to_txt('LIGHTWORLD_MAP_POINTS', self.lightworld_map_tracker_points)
        export_dict_to_txt('DARKWORLD_MAP_POINTS', self.darkworld_map_tracker_points)
        export_dict_to_txt('START_END_TIMESTAMP', {'START_TS': '\''+ str(self.start_ts)[-8:] + '\'', 'END_TS': '\'' + str(self.end_ts)[-8:] + '\''}, fn="config_timestamps.py", delete=True)

        pprint('Exporting raw data')
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_color_coord = self.color_coord_df.copy()
            df_color_coord.frame = [str(s)[-8:] for s in df_color_coord.frame]
            df_color_coord.start_ts = [str(s)[-8:] for s in df_color_coord.start_ts]
            df_color_coord.end_ts = [str(s)[-8:] for s in df_color_coord.end_ts]
            df_color_coord.to_excel(writer, sheet_name='data', index=False)

            params = {attr: getattr(self, attr) for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")}
            params_df = pd.DataFrame(list(params.items()), columns=['Attribute', 'Value'])
            params_df['Value'] = params_df['Value'].astype(str)
            params_df.to_excel(writer, sheet_name='params', index=False)

        pprint(f"Data exported to {output_file}")
    
    def save(self) -> None:
        """
        Save the DunkaScanner object to the output directory as a .pkl file.
        """
        pprint('Saving Scanner object', end='...')
        with open(self.output_path / self.input_video_path.stem / f"{self.input_video_path.stem}.pkl", 'wb') as file:
            pickle.dump(self, file)
        print(f"saved to {self.output_path / self.input_video_path.stem / f'{self.input_video_path.stem}.pkl'}")

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

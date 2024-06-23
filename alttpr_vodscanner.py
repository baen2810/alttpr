from pathlib import Path
from alttpr.scanners import DunkaScanner
from alttpr.scanners import DEFAULT_ITEMTRACKER_POINTS, DEFAULT_LIGHTWORLD_MAP_POINTS, DEFAULT_DARKWORLD_MAP_POINTS
from alttpr.scanners import DEFAULT_ITEMTRACKER_BOX, DEFAULT_LIGHTWORLD_MAP_BOX, DEFAULT_DARKWORLD_MAP_BOX
from alttpr.utils import to_tstr, pprint

import os

def main():
    # Example usage
    video_path = os.path.join(os.getcwd(), "input", "ALTTP Randomizer 1.mp4")

    # Allow the user to select the starting point with default
    selected_start_time = DunkaScanner.select_timestamp(
        video_path, 
        title=f'Select Race Start', 
        default_time='00:04:30'
    )
    if selected_start_time is None:
        exit()

    # Allow the user to select the ending point with default
    selected_end_time = DunkaScanner.select_timestamp(
        video_path, 
        title=f'Select Race End', 
        default_time='00:05:00'
    )
    if selected_end_time is None:
        exit()

    # Select the itemtracker box with default
    itemtracker_box, itemtracker_points = DunkaScanner.select_box(
        video_path, 
        selected_start_time, 
        title=f'Select Item Tracker',
        default_box=DEFAULT_ITEMTRACKER_BOX,
        tracking_points=DEFAULT_ITEMTRACKER_POINTS,
    )
    if itemtracker_box is None:
        exit()

    # Select the lightworld map box with default
    lightworld_map_box, lightworld_map_tracker_points = DunkaScanner.select_box(
        video_path, 
        selected_start_time, 
        title=f'Select Light World Map',
        default_box=DEFAULT_LIGHTWORLD_MAP_BOX,
        tracking_points=DEFAULT_LIGHTWORLD_MAP_POINTS,
    )
    if lightworld_map_box is None:
        exit()

    # Select the lightworld map box with default
    darkworld_map_box, darkworld_map_tracker_points = DunkaScanner.select_box(
        video_path, 
        selected_start_time, 
        title=f'Select Dark World Map',
        default_box=DEFAULT_DARKWORLD_MAP_BOX,
        tracking_points=DEFAULT_DARKWORLD_MAP_POINTS,
    )
    if darkworld_map_box is None:
        exit()

    scanner = DunkaScanner(
        input_video_path=video_path,
        output_path=os.path.join(os.getcwd(), "input"),
        start_ts=selected_start_time,  # Use the selected start time
        end_ts=selected_end_time,      # Use the selected end time
        frames_per_second=1,
        itemtracker_box=itemtracker_box,
        lightworld_map_box=lightworld_map_box,
        darkworld_map_box=darkworld_map_box,
        itemtracker_points=itemtracker_points,
        lightworld_map_tracker_points=lightworld_map_tracker_points,
        darkworld_map_tracker_points=darkworld_map_tracker_points,
    )

    # Run the scan
    scanner.scan()
    scanner.save()
    scanner.export()

    pprint('Finished scanning.')

if __name__ == "__main__":
    main()

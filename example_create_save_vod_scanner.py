from alttpr.scanners import DunkaScanner
from alttpr.config_dunka_compact_sensor_points import ITEMTRACKER_POINTS, LIGHTWORLD_MAP_POINTS, DARKWORLD_MAP_POINTS
from alttpr.config_dunka_compact_sensor_points import ITEMTRACKER_BOX, LIGHTWORLD_MAP_BOX, DARKWORLD_MAP_BOX
from alttpr.utils import pprint

import os

def main():
    # Example usage
    video_path = os.path.join(os.getcwd(), "tests", "data", "alttpr_bow.mp4")
    output_folder = os.path.join(os.getcwd(), "tests", "data")

    # Allow the user to select the starting point with defaults
    selected_start_time = DunkaScanner.select_timestamp(
        video_path, 
        title=f'Select Race Start', 
        default_time='00:00:00'
    )
    if selected_start_time is None:
        exit()

    # Allow the user to select the ending point with defaults
    selected_end_time = DunkaScanner.select_timestamp(
        video_path, 
        title=f'Select Race End', 
        default_time='00:00:05'
    )
    if selected_end_time is None:
        exit()

    # Select the itemtracker box with defaults
    itemtracker_box, itemtracker_points = DunkaScanner.select_box(
        video_path, 
        selected_start_time, 
        title=f'Select Item Tracker',
        default_box=ITEMTRACKER_BOX,
        tracking_points=ITEMTRACKER_POINTS,
    )
    if itemtracker_box is None:
        exit()

    # Select the lightworld map box with default
    lightworld_map_box, lightworld_map_tracker_points = DunkaScanner.select_box(
        video_path, 
        selected_start_time, 
        title=f'Select Light World Map',
        default_box=LIGHTWORLD_MAP_BOX,
        tracking_points=LIGHTWORLD_MAP_POINTS,
    )
    if lightworld_map_box is None:
        exit()

    # Select the lightworld map box with default
    darkworld_map_box, darkworld_map_tracker_points = DunkaScanner.select_box(
        video_path, 
        selected_start_time, 
        title=f'Select Dark World Map',
        default_box=DARKWORLD_MAP_BOX,
        tracking_points=DARKWORLD_MAP_POINTS,
    )
    if darkworld_map_box is None:
        exit()

    scanner = DunkaScanner(
        input_video_path=video_path,
        output_path=output_folder,
        start_ts=selected_start_time,
        end_ts=selected_end_time,
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

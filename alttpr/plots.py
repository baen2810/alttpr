"""
Collection of plots.

"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import min2tstr

def format_df_table(df, int_cols=[], min2tstr_cols=[], pct_cols=[], round_cols=[], combine_cols=[], round_digits=2):
    df_out = df.copy()
    for c in int_cols:
        df_out[c] = round(df_out[c], 0).astype(int)
    for c in min2tstr_cols:
        df_out[c] = [min2tstr(x) for x in df_out[c]]
    for c in round_cols:
        df_out[c] = [round(x, round_digits) for x in df_out[c]]
    for c in pct_cols:
        df_out[c] = [str(int(x*100)) + '%' for x in df_out[c]]
    for c in combine_cols:
        colname, c1, c2 = c
        df_out[colname] = [str(x) + ' - ' + str(y) for x, y in zip(df_out[c1], df_out[c2])]
        df_out = df_out.drop(columns=[c1, c2])
    return df_out


def plot_table(df, title=None, legend=None, text_size=8, legend_x=0.125, legend_y=0.04, legend_font_size_factor=0.45):
    # Add a dummy row to the DataFrame to be able to set its top border to the appropriate style
    dummy_row = pd.DataFrame([['' for _ in range(len(df.columns))]], columns=df.columns)
    df = pd.concat([df, dummy_row], ignore_index=True)
    
    # Calculate dynamic figure size based on the DataFrame dimensions
    num_rows, num_cols = df.shape
    figsize = (num_cols * 1.5, num_rows * 0.1 + 0.5)  # Adjust multipliers as needed for padding

    # Creating a table with matplotlib
    fig, ax = plt.subplots(figsize=figsize)  # Set dynamic figure size
    ax.axis('tight')
    ax.axis('off')
    table_data = df.values
    columns = df.columns

    # Set font properties
    plt.rcParams['font.family'] = 'Georgia'

    # Create the table
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

    # Set font size for the table cells
    table.auto_set_font_size(False)
    table.set_fontsize(text_size)

    # Remove all borders first
    for i in range(len(table_data) + 1):  # +1 to include the header row
        for j in range(len(columns)):
            cell = table[i, j]
            cell.set_edgecolor('white')  # Remove border by setting to white or setting width to 0
            cell.set_linewidth(0)

    # Set only the bottom border for each cell in the specified rows
    def set_bold_bottom_border(table, row):
        for col in range(len(columns)):
            cell = table[row, col]
            cell.set_edgecolor('black')
            cell.set_linewidth(2)  # Set bold bottom border
            cell.visible_edges = "B"  # Only display the bottom edge
            if row == 0:  # Bold text for header row
                cell.set_text_props(weight='bold')

    # Apply the bold bottom border and bold text to the first and last rows
    set_bold_bottom_border(table, 0)  # Header row
    set_bold_bottom_border(table, len(table_data) - 1)  # Last row

    # Center all column values
    for i in range(1, len(table_data) + 1):  # Start from 1 to skip the header row
        for j in range(len(columns)):
            cell = table[i, j]
            cell.set_text_props(ha='center')

    # Adjust table position to leave space for the legend
    ax.set_position([ax.get_position().x0, ax.get_position().y0 + 0.05, ax.get_position().width, ax.get_position().height - 0.1])

    # Add the title above the table if provided
    if title:
        plt.figtext(0.5, 1, title, ha="center", fontsize=text_size, weight='bold')
        
    # Add legend text below the table if provided, aligned to the left of the table
    if legend:
        plt.figtext(legend_x, 0.05-legend_y*(df.shape[0]/5), legend, ha="left", fontsize=text_size * legend_font_size_factor)

    return fig

def plot_race_bars_and_points(
        df: pd.DataFrame, host_name: str, min_races: int = 10, min_n_host: int = 5, highlighted_lst: Optional[List[str]] = None,
        main_bar_color: str = 'steelblue', highlight_color: str = 'teal', host_point_color: str = 'midnightblue',
        overlay_host_points: bool = True, show_error_bars: bool = True, filter_race_modes: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6), text_sizes: Tuple[int, int, int, int] = (9, 7, 8, 10), 
        chart_title: str = 'Median Racetime by Race-Modus with. Standard Error',
        x_axis_title: str = 'Median Racetime', y_axis_title: str = 'Race Mode',
        group_by_col: str = 'race_mode_simple', bar_col: str = 'entrant_finishtime|median', point_col: str = 'entrant_finishtime|median_host',
        footnote_str: str = '',
        show_legend: bool = True,
        agg_func: str = 'median',
        tight_layout: bool = False,
        unknown_label: str ='Unbekannt',
        sort: str = 'descending'
        ) -> plt:
    """
    Plot race statistics with median finish times, including optional host median finish times and error bars.

    Parameters:
    - df: DataFrame containing race data with median finish times and host median finish times.
    - host_name: Name of the host for which to plot the median finish times.
    - min_races: Minimum number of races for a race mode to be included in the plot (default is 10).
    - min_n_host: Minimum number of races for the host to be included in the plot (default is 5).
    - highlighted_lst: List of race modes to highlight (default is []).
    - main_bar_color: Color of the main bars (default is 'steelblue').
    - highlight_color: Color of the highlighted bars (default is 'teal').
    - host_point_color: Color of the host median finish time points (default is 'midnightblue').
    - overlay_host_points: Boolean to control overlay of host median time points (default is True).
    - show_error_bars: Boolean to control display of error bars (default is True).
    - filter_race_modes: List of race modes to filter (default is None).
    - figsize: Tuple specifying the figure size (default is (10, 6)).
    - text_sizes: Tuple specifying text sizes for bar annotations, host point labels, footnote, axis tick labels, and axis titles
                  (default is (9, 7, 8, 10)).
    - chart_title: Title of the chart (default is 'Medianzeiten nach Race-Modus inkl. Standardfehler').
    - x_axis_title: Title of the x-axis (default is 'Medianzeiten').
    - y_axis_title: Title of the y-axis (default is 'Race Mode').
    - group_by_col: Column to group by (default is 'race_mode_simple').
    - bar_col: Column to compute the bars (default is 'entrant_finishtime|median').
    - point_col: Column to compute the points (default is 'entrant_finishtime|median_host').
    - footnote_str: String to display as footnote at the bottom left of the plot.
    - show_legend: Boolean to show/hide the point legend (host_name).
    - agg_func: Aggregation function (default is 'median').
    - tight_layout: Boolean to adjust layout to ensure labels are not cut off (default is False).
    - unknown_label: Label for unknown categories (default is 'Unbekannt').
    - sort: Parameter to specify sorting ('ascending', 'descending', 'cat_ascending', 'cat_descending').

    Returns:
    - plt: The matplotlib.pyplot object to allow further customization or saving.
    """

    # Function to compute standard error of the median using bootstrapping
    def bootstrap_median_std_error(data, n_bootstraps=1000):
        medians = []
        for _ in range(n_bootstraps):
            sample = np.random.choice(data, size=len(data), replace=True)
            medians.append(np.median(sample))
        return np.std(medians)
    
    # Determine if bar_col and point_col are timedelta or numeric
    if np.issubdtype(df[bar_col].dtype, np.timedelta64):
        is_timedelta = True
        df[bar_col] = df[bar_col].dt.total_seconds()
        df[point_col] = df[point_col].dt.total_seconds()
        df['finishtime_seconds'] = df[bar_col]
        df['finishtime_seconds_host'] = df[point_col]
    else:
        is_timedelta = False
        df['finishtime_seconds'] = df[bar_col]
        df['finishtime_seconds_host'] = df[point_col]

    # Group by the specified column, calculate the median and count
    grouped_df = df.groupby(group_by_col).agg(
        finishtime_seconds=('finishtime_seconds', agg_func),
        finishtime_seconds_host=('finishtime_seconds_host', agg_func),
        race_count=('finishtime_seconds', 'count'),
        host_race_count=('finishtime_seconds_host', 'count')
    ).reset_index()

    # Compute standard error for each group
    grouped_df['std_error'] = df.groupby(group_by_col)['finishtime_seconds'].apply(bootstrap_median_std_error).reset_index(drop=True)
    grouped_df['std_error_host'] = df.groupby(group_by_col)['finishtime_seconds_host'].apply(bootstrap_median_std_error).reset_index(drop=True)

    # Filter out labels
    filtered_df = grouped_df[grouped_df['race_count'] >= min_races]
    filtered_df = filtered_df[filtered_df['host_race_count'] >= min_n_host]

    if filter_race_modes:
        filtered_df = filtered_df[filtered_df[group_by_col].isin(filter_race_modes)]
    
    # Separate 'Unknown' category
    unknown_df = filtered_df[filtered_df[group_by_col] == unknown_label]
    filtered_df = filtered_df[filtered_df[group_by_col] != unknown_label]

    # Sorting logic
    if sort == 'ascending':
        sorted_df = filtered_df.sort_values(by='finishtime_seconds', ascending=True)
    elif sort == 'descending':
        sorted_df = filtered_df.sort_values(by='finishtime_seconds', ascending=False)
    elif sort == 'cat_ascending':
        sorted_df = filtered_df.sort_values(by=group_by_col, ascending=True)
    elif sort == 'cat_descending':
        sorted_df = filtered_df.sort_values(by=group_by_col, ascending=False)
    else:
        raise ValueError("Invalid sort parameter. Choose from 'ascending', 'descending', 'cat_ascending', or 'cat_descending'.")

    # Prepend 'Unknown' category to the beginning if present
    sorted_df = pd.concat([unknown_df, sorted_df])

    # Calculate the total number of races
    total_races = sorted_df['race_count'].sum()

    # Plotting the bar chart
    plt.figure(figsize=figsize)

    bars = plt.barh(
        sorted_df[group_by_col] + '\n(N=' + sorted_df['race_count'].astype(str) + ')', 
        sorted_df['finishtime_seconds'], 
        xerr=sorted_df['std_error'] if show_error_bars else None,  # Conditionally add standard error bars
        color=[
            highlight_color if race_mode in highlighted_lst else 
            main_bar_color if race_mode != unknown_label else 'grey' 
            for race_mode in sorted_df[group_by_col]
        ], 
        capsize=5 if show_error_bars else 0
    )

    # Overlay host median finish times as points with error bars if enabled
    if overlay_host_points:
        plt.errorbar(
            sorted_df['finishtime_seconds_host'], 
            range(len(sorted_df)), 
            xerr=sorted_df['std_error_host'] if show_error_bars else None,
            fmt='o', 
            color=host_point_color, 
            zorder=5, 
            label=f'{host_name}'
        )

        # Add labels to the host median finish time points and number of races completed by the host
        for i, (host_time, host_race_count) in enumerate(zip(sorted_df['finishtime_seconds_host'], sorted_df['host_race_count'])):
            plt.text(
                host_time,
                i + 0.1,  # Slightly above the point
                f'{host_time:.2f}' if not is_timedelta else format_timedelta(host_time, None),
                ha='center',
                va='bottom',
                color=host_point_color,
                fontsize=text_sizes[1]  # Reduced text size
            )
            plt.text(
                host_time,
                i - 0.15,  # Slightly below the point
                f'n={host_race_count}',
                ha='center',
                va='top',
                color=host_point_color,
                fontsize=text_sizes[1]  # Reduced text size
            )

    if is_timedelta:
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: format_timedelta(x, None)))
    plt.xlabel(x_axis_title, fontsize=text_sizes[3])
    plt.ylabel(y_axis_title, fontsize=text_sizes[3])
    plt.title(f'{chart_title} (N={total_races})', fontsize=text_sizes[3])

    # Annotate bars with the median finish times in HH:MM:SS or numeric value
    for bar in bars:
        plt.gca().text(
            100,  # x-coordinate (fixed at 100)
            bar.get_y() + bar.get_height() / 2,
            f'{bar.get_width():.2f}' if not is_timedelta else format_timedelta(bar.get_width(), None),
            ha='left',
            va='center',
            color='white',
            fontsize=text_sizes[0]
        )

    # Set tick label size
    plt.xticks(fontsize=text_sizes[3])
    plt.yticks(fontsize=text_sizes[3])

    # Add footnote
    plt.figtext(0.01, 0.01, footnote_str, ha='left', fontsize=text_sizes[2])

    # Add legend if overlay_host_points is True
    if overlay_host_points and show_legend:
        plt.legend()

    # Adjust layout to ensure labels are not cut off
    if tight_layout:
        plt.tight_layout()
        plt.subplots_adjust(left=0.3)  # Adjust left margin to fit long labels

    return plt

# Helper function to format timedelta values
def format_timedelta(x, pos):
    hours, remainder = divmod(x, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'



def plot_race_bars_and_points_for_ts(
        df: pd.DataFrame, host_name: str, min_races: int = 10, min_n_host: int = 5, highlighted_lst: Optional[List[str]] = None,
        main_bar_color: str = 'steelblue', highlight_color: str = 'teal', host_point_color: str = 'midnightblue',
        overlay_host_points: bool = True, show_error_bars: bool = True, filter_race_modes: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6), text_sizes: Tuple[int, int, int, int] = (9, 7, 8, 10), 
        chart_title: str = 'Median Racetime by Race-Modus with. Standard Error',
        x_axis_title: str = 'Median Racetime', y_axis_title: str = 'Race Mode',
        group_by_col: str = 'race_mode_simple', bar_col: str = 'entrant_finishtime|median', point_col: str = 'entrant_finishtime|median_host',
        footnote_str: str = '',
        show_legend: bool = True,
        agg_func: str = 'median',
        tight_layout: bool = False,
        unknown_label: str ='Unbekannt',
        ) -> plt:
    """
    Plot race statistics with median finish times, including optional host median finish times and error bars.

    Parameters:
    - df: DataFrame containing race data with median finish times and host median finish times.
    - host_name: Name of the host for which to plot the median finish times.
    - min_races: Minimum number of races for a race mode to be included in the plot (default is 10).
    - min_n_host: Minimum number of races for the host to be included in the plot (default is 5).
    - highlighted_lst: List of race modes to highlight (default is []).
    - main_bar_color: Color of the main bars (default is 'steelblue').
    - highlight_color: Color of the highlighted bars (default is 'teal').
    - host_point_color: Color of the host median finish time points (default is 'midnightblue').
    - overlay_host_points: Boolean to control overlay of host median time points (default is True).
    - show_error_bars: Boolean to control display of error bars (default is True).
    - filter_race_modes: List of race modes to filter (default is None).
    - figsize: Tuple specifying the figure size (default is (10, 6)).
    - text_sizes: Tuple specifying text sizes for bar annotations, host point labels, footnote, axis tick labels, and axis titles
                  (default is (9, 7, 8, 10)).
    - chart_title: Title of the chart (default is 'Medianzeiten nach Race-Modus inkl. Standardfehler').
    - x_axis_title: Title of the x-axis (default is 'Medianzeiten').
    - y_axis_title: Title of the y-axis (default is 'Race Mode').
    - group_by_col: Column to group by (default is 'race_mode_simple').
    - bar_col: Timedelta column to compute the bars (default is 'entrant_finishtime|median').
    - point_col: Timedelta column to compute the points (default is 'entrant_finishtime|median_host').
    - footnote_str: Str column to display as footnote at the bottom left of the plot
    - show_legend: Bool variable to show/hide the point legend (host_name)

    Returns:
    - plt: The matplotlib.pyplot object to allow further customization or saving.
    """

    # Format y-axis to display as HH:MM:SS without '0 days' and milliseconds
    def format_timedelta(x, pos):
        hours, remainder = divmod(x, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'    

    # Function to compute standard error of the median using bootstrapping
    def bootstrap_median_std_error(data, n_bootstraps=1000):
        medians = []
        for _ in range(n_bootstraps):
            sample = np.random.choice(data, size=len(data), replace=True)
            medians.append(np.median(sample))
        return np.std(medians)
    
    # Convert Timedelta to total seconds and round to nearest second
    df[bar_col] = df[bar_col].dt.round('s')
    df[point_col] = df[point_col].dt.round('s')
    df['finishtime_seconds'] = df[bar_col].dt.total_seconds()
    df['finishtime_seconds_host'] = df[point_col].dt.total_seconds()

    # Group by the specified column, calculate the median and count
    grouped_df = df.groupby(group_by_col).agg(
        finishtime_seconds=('finishtime_seconds', agg_func),
        finishtime_seconds_host=('finishtime_seconds_host', agg_func),
        race_count=('finishtime_seconds', 'count'),
        host_race_count=('finishtime_seconds_host', 'count')
    ).reset_index()

    # Compute standard error for each group
    grouped_df['std_error'] = df.groupby(group_by_col)['finishtime_seconds'].apply(bootstrap_median_std_error).reset_index(drop=True)
    grouped_df['std_error_host'] = df.groupby(group_by_col)['finishtime_seconds_host'].apply(bootstrap_median_std_error).reset_index(drop=True)

    # Filter out labels
    filtered_df = grouped_df[grouped_df['race_count'] >= min_races]
    filtered_df = filtered_df[filtered_df['host_race_count'] >= min_n_host]

    if filter_race_modes:
        filtered_df = filtered_df[filtered_df[group_by_col].isin(filter_race_modes)]
    
    # Separate 'Unknown' category
    unknown_df = filtered_df[filtered_df[group_by_col] == unknown_label]
    filtered_df = filtered_df[filtered_df[group_by_col] != unknown_label]

    # Sort by median finish time in descending order
    sorted_df = filtered_df.sort_values(by='finishtime_seconds', ascending=False)

    # Prepend 'Unknown' category to the beginning
    sorted_df = pd.concat([unknown_df, sorted_df])

    # Convert the median finish times back to Timedelta
    sorted_df['agg_finishtime'] = pd.to_timedelta(sorted_df['finishtime_seconds'], unit='s')
    sorted_df['agg_finishtime_host'] = pd.to_timedelta(sorted_df['finishtime_seconds_host'], unit='s')

    # Calculate the total number of races
    total_races = sorted_df['race_count'].sum()

    # Plotting the bar chart
    plt.figure(figsize=figsize)

    bars = plt.barh(
        sorted_df[group_by_col] + '\n(N=' + sorted_df['race_count'].astype(str) + ')', 
        sorted_df['finishtime_seconds'], 
        xerr=sorted_df['std_error'] if show_error_bars else None,  # Conditionally add standard error bars
        color=[
            highlight_color if race_mode in highlighted_lst else 
            main_bar_color if race_mode != unknown_label else 'grey' 
            for race_mode in sorted_df[group_by_col]
        ], 
        capsize=5 if show_error_bars else 0
    )

    # Overlay host median finish times as points with error bars if enabled
    if overlay_host_points:
        plt.errorbar(
            sorted_df['finishtime_seconds_host'], 
            range(len(sorted_df)), 
            xerr=sorted_df['std_error_host'] if show_error_bars else None,
            fmt='o', 
            color=host_point_color, 
            zorder=5, 
            label=f'{host_name}'
        )

        # Add labels to the host median finish time points and number of races completed by the host
        for i, (host_time, host_race_count) in enumerate(zip(sorted_df['finishtime_seconds_host'], sorted_df['host_race_count'])):
            plt.text(
                host_time,
                i + 0.1,  # Slightly above the point
                format_timedelta(host_time, None),
                ha='center',
                va='bottom',
                color=host_point_color,
                fontsize=text_sizes[1]  # Reduced text size
            )
            plt.text(
                host_time,
                i - 0.15,  # Slightly below the point
                f'n={host_race_count}',
                ha='center',
                va='top',
                color=host_point_color,
                fontsize=text_sizes[1]  # Reduced text size
            )

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_timedelta))
    plt.xlabel(x_axis_title, fontsize=text_sizes[3])
    plt.ylabel(y_axis_title, fontsize=text_sizes[3])
    plt.title(f'{chart_title} (N={total_races})', fontsize=text_sizes[3])

    # Annotate bars with the median finish times in HH:MM:SS at x-coordinate value 100
    for bar in bars:
        plt.gca().text(
            100,  # x-coordinate (fixed at 100)
            bar.get_y() + bar.get_height() / 2,
            format_timedelta(bar.get_width(), None),
            ha='left',
            va='center',
            color='white',
            fontsize=text_sizes[0]
        )

    # Set tick label size
    plt.xticks(fontsize=text_sizes[3])
    plt.yticks(fontsize=text_sizes[3])

    # Add footnote
    plt.figtext(0.01, 0.01, footnote_str, ha='left', fontsize=text_sizes[2])

    # Add legend if overlay_host_points is True
    if overlay_host_points and show_legend:
        plt.legend()

    # Adjust layout to ensure labels are not cut off
    if tight_layout:
        plt.tight_layout()
        plt.subplots_adjust(left=0.3)  # Adjust left margin to fit long labels

    return plt


def plot_race_box_plots_and_points(
        df: pd.DataFrame, host_name: str, min_races: int = 10, min_n_host: int = 5, highlighted_lst: Optional[List[str]] = None,
        box_color: str = 'steelblue', highlight_color: str = 'teal', host_point_color: str = 'midnightblue',
        overlay_host_points: bool = True, filter_race_modes: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (10, 6), text_sizes: Tuple[int, int, int, int] = (9, 7, 8, 10), 
        chart_title: str = 'Distribution of Racetime by Race Mode',
        y_axis_title: str = 'Racetime', x_axis_title: str = 'Race Mode',
        group_by_col: str = 'race_mode_simple', time_col: str = 'entrant_finishtime',
        point_col: str = 'entrant_finishtime|median_host',
        footnote_str: str = '',
        show_legend: bool = True,
        tight_layout: bool = False,
        unknown_label: str = 'Unbekannt',
        fix_x_origin: bool = False,
        ) -> plt:
    """
    Plot race statistics with box plots, including optional host finish times as overlaid points.

    Parameters:
    - df: DataFrame containing race data with finish times and host finish times.
    - host_name: Name of the host for which to plot the median finish times.
    - min_races: Minimum number of races for a race mode to be included in the plot (default is 10).
    - min_n_host: Minimum number of races for the host to be included in the plot (default is 5).
    - highlighted_lst: List of race modes to highlight (default is []).
    - box_color: Color of the box plots (default is 'steelblue').
    - highlight_color: Color of the highlighted boxes (default is 'teal').
    - host_point_color: Color of the host median finish time points (default is 'midnightblue').
    - overlay_host_points: Boolean to control overlay of host finish time points (default is True).
    - filter_race_modes: List of race modes to filter (default is None).
    - figsize: Tuple specifying the figure size (default is (10, 6)).
    - text_sizes: Tuple specifying text sizes for bar annotations, host point labels, footnote, axis tick labels, and axis titles
                  (default is (9, 7, 8, 10)).
    - chart_title: Title of the chart (default is 'Distribution of Racetime by Race Mode').
    - y_axis_title: Title of the y-axis (default is 'Racetime').
    - x_axis_title: Title of the x-axis (default is 'Race Mode').
    - group_by_col: Column to group by (default is 'race_mode_simple').
    - time_col: Timedelta column to compute the boxes (default is 'entrant_finishtime').
    - point_col: Timedelta column to compute the points to be overlaid (default is 'entrant_finishtime|median_host').
    - footnote_str: Str column to display as footnote at the bottom left of the plot.
    - show_legend: Bool variable to show/hide the point legend (host_name).
    - tight_layout: Bool variable to enable/disable tight layout.
    - unknown_label: String specifying the group_by_col label for the Unknown class (default is 'Unbekannt').
    - fix_x_origin: Bool variable to fix x-axis origin to 00:00:00 (default is False).

    Returns:
    - plt: The matplotlib.pyplot object to allow further customization or saving.
    """

    # Format y-axis to display as HH:MM:SS without '0 days' and milliseconds
    def format_timedelta(x, pos):
        hours, remainder = divmod(x, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f'{int(hours):02}:{int(minutes):02}:{int(seconds):02}'    

    # Filter by race modes if provided
    if filter_race_modes:
        df = df[df[group_by_col].isin(filter_race_modes)]

    # Convert Timedelta to total seconds for plotting
    df['finishtime_seconds'] = df[time_col].dt.total_seconds()
    df['finishtime_seconds_host'] = df[point_col].dt.total_seconds()

    # Filter out rows where the host does not meet the minimum number of races
    host_counts = df.dropna(subset=[point_col]).groupby(group_by_col).size()
    df = df[df[group_by_col].map(df[group_by_col].value_counts()) >= min_races]
    df = df[df[group_by_col].map(host_counts) >= min_n_host]

    # Get unique race modes
    race_modes = df[group_by_col].unique()

    # Separate unknown label
    if unknown_label in race_modes:
        known_race_modes = [mode for mode in race_modes if mode != unknown_label]
    else:
        known_race_modes = race_modes

    # Count the number of races for each race mode
    race_counts = df.groupby(group_by_col).size().reindex(known_race_modes, fill_value=0)
    
    # Calculate medians to sort the box plots
    medians = df.groupby(group_by_col)['finishtime_seconds'].median().reindex(known_race_modes, fill_value=0)
    sorted_indices = medians.sort_values(ascending=False).index
    sorted_labels = [f'{mode}\n(N={race_counts[mode]})' for mode in sorted_indices]

    # Add the unknown label at the top
    if unknown_label in race_modes:
        sorted_indices = [unknown_label] + sorted_indices.tolist()
        sorted_labels = [f'{unknown_label}\n(N={df[group_by_col].value_counts()[unknown_label]})'] + sorted_labels

    # Prepare box plot data and labels with race counts
    box_data = [df[df[group_by_col] == mode]['finishtime_seconds'] for mode in sorted_indices]

    # Plotting the box plot
    plt.figure(figsize=figsize)
    box = plt.boxplot(box_data, patch_artist=True, vert=False, labels=sorted_labels, showfliers=False)

    # Customize the box plot
    for patch, median, mode in zip(box['boxes'], box['medians'], sorted_indices):
        if mode == unknown_label:
            patch.set_facecolor('grey')
        else:
            patch.set_facecolor(highlight_color if mode in highlighted_lst else box_color)
        median.set_color('black')

    # Filter out rows where the host does not meet the minimum number of races for the host point overlay
    df_host_filtered = df.dropna(subset=[point_col])

    # Overlay host points if enabled
    if overlay_host_points:
        for i, mode in enumerate(sorted_indices):
            host_times = df_host_filtered[df_host_filtered[group_by_col] == mode]['finishtime_seconds_host']
            if not host_times.empty:
                plt.scatter(host_times, np.full_like(host_times, i + 1), color=host_point_color, zorder=5, label=host_name if i == 0 else "")

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(format_timedelta))
    plt.xlabel(x_axis_title, fontsize=text_sizes[3])
    plt.ylabel(y_axis_title, fontsize=text_sizes[3])
    plt.title(chart_title, fontsize=text_sizes[3])

    # Fix x-axis origin if required
    if fix_x_origin:
        plt.xlim(left=0)

    # Add footnote
    plt.figtext(0.01, 0.01, footnote_str, ha='left', fontsize=text_sizes[2])

    # Add legend if host points are overlaid and legend is enabled
    if overlay_host_points and show_legend:
        plt.legend()

    # Adjust layout to ensure labels are not cut off
    if tight_layout:
        plt.tight_layout()
        plt.subplots_adjust(left=0.3)  # Adjust left margin to fit long labels

    return plt


def plot_counts(
        df: pd.DataFrame,
        category_col: str,
        chart_type: str,
        group_by_col: Optional[str] = None,
        category_filter: Optional[List[str]] = None,
        unknown_col: str = 'Unbekannt',
        unknown_location: str = 'last',
        sort: str = 'ascending',
        highlighted_lst: Optional[List[str]] = None,
        color_set: List[str] = ['steelblue', 'teal', 'grey'],
        figsize: Tuple[int, int] = (10, 6),
        text_sizes: Tuple[int, int, int, int, int, int] = (9, 7, 8, 10, 12, 11),  # Added sixth text size for axis tick labels
        chart_title: str = '',
        x_axis_title: str = '',
        y_axis_title: str = '',
        footnote_str: str = '',
        show_legend: bool = True,
        tight_layout: bool = False,
        label_color: str = 'black',
        label_position: str = 'above',  # can be 'above' or 'centered'
        longtail_thres: int = 0,
        other_label: str = 'Andere'
        ) -> plt:

    # Filter out specified categories
    if category_filter:
        df = df[~df[category_col].isin(category_filter)]

    # Count the occurrences of each category
    category_counts = df[category_col].value_counts()

    # Handle longtail categories
    if longtail_thres > 0:
        longtail_counts = category_counts[category_counts <= longtail_thres].sum()
        category_counts = category_counts[category_counts > longtail_thres]
        if longtail_counts > 0:
            category_counts[other_label] = longtail_counts

    # Sorting
    if sort == 'ascending':
        category_counts = category_counts.sort_values(ascending=True)
    elif sort == 'descending':
        category_counts = category_counts.sort_values(ascending=False)
    elif sort == 'cat_ascending':
        category_counts = category_counts.sort_index(ascending=True)
    elif sort == 'cat_descending':
        category_counts = category_counts.sort_index(ascending=False)

    # Handle unknown column location
    if unknown_col in category_counts.index:
        unknown_count = category_counts[unknown_col]
        category_counts = category_counts.drop(unknown_col)
        if unknown_location == 'first':
            category_counts = pd.concat([pd.Series({unknown_col: unknown_count}), category_counts])
        else:  # 'last'
            category_counts[unknown_col] = unknown_count

    # Handle other_label location
    if other_label in category_counts.index:
        other_count = category_counts[other_label]
        category_counts = category_counts.drop(other_label)
        if unknown_location == 'first':
            category_counts = pd.concat([pd.Series({other_label: other_count}), category_counts])
        else:  # 'last'
            category_counts[other_label] = other_count

    # Pie chart
    if chart_type == 'pie':
        if group_by_col:
            unique_groups = df[group_by_col].unique()
            num_groups = len(unique_groups)
            fig, axes = plt.subplots(1, num_groups, figsize=figsize)
            if num_groups == 1:
                axes = [axes]
            for ax, group in zip(axes, unique_groups):
                group_counts = df[df[group_by_col] == group][category_col].value_counts()
                group_counts = group_counts[group_counts.index.isin(category_counts.index)]
                ax.pie(group_counts, labels=group_counts.index, autopct='%1.1f%%', colors=color_set, startangle=90,
                       textprops={'fontsize': text_sizes[0], 'color': label_color})
                ax.set_title(f'{group}', fontsize=text_sizes[3])
        else:
            plt.figure(figsize=figsize)
            plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=color_set, startangle=90,
                    textprops={'fontsize': text_sizes[0], 'color': label_color})

    # Horizontal bar chart
    elif chart_type == 'bars_horizontal':
        plt.figure(figsize=figsize)
        bar_colors = [color_set[0] if idx not in highlighted_lst else color_set[1] for idx in category_counts.index]
        if unknown_col in category_counts.index:
            bar_colors[category_counts.index.get_loc(unknown_col)] = color_set[2]
        if other_label in category_counts.index:
            bar_colors[category_counts.index.get_loc(other_label)] = color_set[2]
        bars = plt.barh(category_counts.index, category_counts.values, color=bar_colors)
        plt.xlabel(x_axis_title, fontsize=text_sizes[4])
        plt.ylabel(y_axis_title, fontsize=text_sizes[4])
        plt.xticks(fontsize=text_sizes[5])
        plt.yticks(fontsize=text_sizes[5])
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for bar in bars:
            width = bar.get_width()
            if label_position == 'centered':
                plt.gca().text(width / 2, bar.get_y() + bar.get_height() / 2, f'{int(width)}', va='center', ha='center', color=label_color, fontsize=text_sizes[0])
            else:
                plt.gca().text(width, bar.get_y() + bar.get_height() / 2, f'{int(width)}', va='center', ha='left', color=label_color, fontsize=text_sizes[0])

    # Vertical bar chart
    elif chart_type == 'bars_vertical':
        plt.figure(figsize=figsize)
        bar_colors = [color_set[0] if idx not in highlighted_lst else color_set[1] for idx in category_counts.index]
        if unknown_col in category_counts.index:
            bar_colors[category_counts.index.get_loc(unknown_col)] = color_set[2]
        if other_label in category_counts.index:
            bar_colors[category_counts.index.get_loc(other_label)] = color_set[2]
        bars = plt.bar(category_counts.index, category_counts.values, color=bar_colors)
        plt.xlabel(x_axis_title, fontsize=text_sizes[4])
        plt.ylabel(y_axis_title, fontsize=text_sizes[4])
        plt.xticks(fontsize=text_sizes[5])
        plt.yticks(fontsize=text_sizes[5])
        plt.gca().yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        for bar in bars:
            height = bar.get_height()
            if label_position == 'centered':
                plt.gca().text(bar.get_x() + bar.get_width() / 2, height / 2, f'{int(height)}', ha='center', va='center', color=label_color, fontsize=text_sizes[0])
            else:
                plt.gca().text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', color=label_color, fontsize=text_sizes[0])

    # Common settings
    plt.title(chart_title, fontsize=text_sizes[3])
    if footnote_str:
        plt.figtext(0.01, 0.01, footnote_str, ha='left', fontsize=text_sizes[2])
    if show_legend:
        plt.legend()
    if tight_layout:
        plt.tight_layout()

    return plt
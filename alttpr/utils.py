"""
Collection of general purpose helper functions.

"""

from os import rename
from os.path import exists, isfile, isdir  # join as pjoin
from os.path import splitext, split as psplit, normpath
from pandas import Series
from pandas.core.series import Series as pdSeriesType
from datetime import datetime as dt
from time import sleep
from os import remove
from shutil import rmtree, copytree, copy2 as copy
from typing import Optional, List, Union

import os
import json
import types
import pickle
import zipfile
import numpy as np
import pandas as pd
import re
import importlib.util
from pathlib import Path

def read_var_from_files(file_paths: Union[str, List[Path]], var_name: str) -> List[Optional[str]]:
    """
    Reads the 'var_name' variable from a list of Python files.
    
    :param file_paths: A list of Path objects pointing to Python files.
    :return: A list of 'var_name' values from each file, or None if not found.
    """
    values_list = []
    file_paths [file_paths] if isinstance(file_paths, str) else file_paths
    
    for file_path in file_paths:
        # Load the Python file as a module
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            
            # Access the 'NOTES' variable, if it exists
            values = getattr(module, var_name, None)
            if values is not None:
                values_list.append(values)
            else:
                values_list.append(None)
        except Exception as e:
            values_list.append(None)
    
    return values_list


def date_from_str(text: str, date_formats: List[str] = ["%d.%m.%Y", "%d.%m.%y"]) -> Optional[dt]:
    """
    Extracts a date from a given string based on the provided list of date formats.
    By default, it checks both 'DD.MM.YYYY' and 'DD.MM.YY' formats.
    
    :param text: The string containing the date.
    :param date_formats: A list of expected date formats. Default is ["%d.%m.%Y", "%d.%m.%y"].
    :return: A datetime object if a date is found and successfully parsed, else None.
    """
    # Define a regex pattern to match both "DD.MM.YYYY" and "DD.MM.YY"
    date_pattern = r"\b\d{2}\.\d{2}\.\d{2,4}\b"
    
    # Search for a date in the string
    match = re.search(date_pattern, text)
    if match:
        date_str = match.group(0)
        # Try each date format in the list
        for date_format in date_formats:
            try:
                # Try to parse the date using the current format
                parsed_date = dt.strptime(date_str, date_format)
                return parsed_date
            except ValueError:
                continue  # Try the next format if parsing fails
    # Return None if no date was found or successfully parsed
    return None


def notna(x):
    is_na = False if isna(x) else True
    return is_na


def isna(x):
    is_na = True if pd.isna(x) else False
    try:
        is_na = True if np.isnan(x) else False
    except:
        pass
    return is_na


def clear_console():
    # For Windows
    if os.name == 'nt':
        _ = os.system('cls')
    # For Linux and macOS
    else:
        _ = os.system('clear')

def dotenv2int(dotenv_var):
    dotenv_var = os.getenv(dotenv_var)
    dotenv_var = None if dotenv_var == "None" else int(dotenv_var)
    return dotenv_var

def dotenv2dict(dotenv_var):
    dotenv_var = json.loads(os.getenv(dotenv_var))
    dotenv_dict = {}
    for k,v in dotenv_var.items():
        v = None if v=="None" else v
        if v == "True":
            v = True
        elif v == "False":
            v = False
        try:
            if type(v).__name__ != 'bool':
                v = int(v)
        except:
            pass
        dotenv_dict[k] = v
    return dotenv_dict

def dotenv2lst(dotenv_var):
    return [x.strip() for x in os.getenv(dotenv_var).split(',')]

def get_workspace_vars(local_vars, filter_vars=None, exclude_vars=['__builtins__', '__name__']):
    filter_vars = filter_vars if filter_vars else local_vars.values()
    workspace_str = '\n'
    for var_name, value in local_vars.items():
        if value and not(callable(value)) and not(isinstance(value, types.ModuleType)) and var_name not in exclude_vars and value in filter_vars:  # and value[:2] != '__' 
            workspace_str += f">> {var_name}={value}\n"
    return workspace_str

# Define a function to export a dictionary to a text file
def export_dict_to_txt(output_path, dict_name, dictionary, delete=False, fn="config_trackerpoints.py"):
    txt_file = output_path / fn
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
def export_tuple_to_txt(output_path, tuple_name, tuple_value, delete=False, fn="config_trackerpoints.py"):
    txt_file = output_path /  fn
    # Ensure the directory exists
    txt_file.parent.mkdir(parents=True, exist_ok=True)
    # Delete the file if it exists
    if txt_file.exists() and delete:
        txt_file.unlink()
    # Write the tuple to the file, appending if the file exists
    with open(txt_file, 'a') as f:
        f.write(f"{tuple_name} = {tuple_value}\n")
    
    pprint(f"{tuple_name} exported to {txt_file}")


def pprintdesc(desc, tgt_len=26):
    if len(desc) <= tgt_len:
        while len(desc) < tgt_len - 1:
            desc += ' '
    return desc


def get_enum_str(lst, conjunction='und', separator=', '):
    conjunction = conjunction.strip()
    if len(lst) == 0:
        return ''
    elif len(lst) == 1:
        return f'{lst[0]}'
    else:
        lst_str = separator.join(lst[:-1]) + ' ' + conjunction + ' ' + lst[-1]
        return lst_str


def to_tstr(ts):
    tsn = ts
    if type(ts) == pd.Timedelta:
        tsn = pd.Timestamp('1900-01-01') + ts
    try:
        tsn = tsn.strftime('%H:%M:%S')
    except:
        pass
    return tsn


def to_dstr(ts):
    tsn = ts
    try:
        tsn = tsn.strftime('%d.%m.%Y')
    except:
        pass
    return tsn


def get_list(df_col, as_type='ts'):
    colname_as_lst = list(df_col)
    if as_type in ('ts', 'tstr'):   
        try:
            colname_as_lst = [pd.Timestamp(v) for v in colname_as_lst]
            colname_as_lst = [v.strftime('%H:%M:%S') for v in colname_as_lst] if as_type == 'tstr' else colname_as_lst
        except Exception as e:
            colname_as_lst = e
    if as_type in ('dstr', 'dayspast'):
        try:
            colname_as_lst = [pd.Timestamp(v) for v in colname_as_lst]
            colname_as_lst = [v.strftime('%d.%m.%Y') for v in colname_as_lst] if as_type == 'dstr' else colname_as_lst
            colname_as_lst = [(dt.now().date() - pd.Timestamp(v).date()).days for v in colname_as_lst] if as_type == 'dayspast' else colname_as_lst
        except:
            pass
    return colname_as_lst


def get_first(df_col, as_type='ts'):
    return get_list(df_col, as_type=as_type)[0]


def get_td(ts1, ts2, as_type='tstr'):
    if ts1 > ts2:
        td =  ts1-ts2
        is_better = False  # lower is better
    else:
        td =  ts2-ts1
        is_better = True
    if as_type == 'tstr':
        td = str(td)[-5:].replace(':', 'm') + 's'
    if as_type == 'ts':
        td = pd.Timestamp(str(td)[-6:])
    return td, is_better


def ts2str(ts):
    return ts.strftime('%H:%M:%S')


def min2tstr(x):
    if isna(x):
        return ''
    h = int(x / 60)
    m = int(x) - h * 60
    s = int((x - int(x)) * 60)
    return f'{str(h).zfill(2)}:{str(m).zfill(2)}:{str(s).zfill(2)}'


def sec2tstr(x):
    x = x / 60
    return min2tstr(x)


def td2int(td, unit='minutes', n_digits=2):
    if unit == 'minutes':
        td_int = round(td.seconds / 60, n_digits)
    elif unit == 'seconds':
        td_int = round(td.seconds, n_digits)
    else:
        raise ValueError(f'Unknown parameter: {unit=}')
    return td_int


def plistmatch(a, b):
    '''
    Returns three lists:
    intersection_lst: Elements in both lists
    a_not_in_b_lst: Elements of list a, that are not in list b
    b_not_in_a_lst: Elements of list b, that are not in list a
    '''
    intersection_lst = list(set(a) & set(b))
    a_not_in_b_lst = [x for x in a if x not in b]
    b_not_in_a_lst = [x for x in b if x not in a]
    return intersection_lst, a_not_in_b_lst, b_not_in_a_lst


def pwalk(top, maxdepth):
    dirs, nondirs = [], []
    for entry in os.scandir(top):
        (dirs if entry.is_dir() else nondirs).append(entry.path)
    yield top, dirs, nondirs
    if maxdepth > 1:
        for path in dirs:
            for x in pwalk(path, maxdepth - 1):
                yield x


def pdcols(df, replace_dict={
    'Reichweite': 'RW',
    'Szene': 'Sz',
    'BeginnEnde': 'BE',
    'Gesamt': 'Ges',
    'Prozent': '%',
}):
    '''
    Consistently shorten column names in a pandas df.
    '''
    cols = []
    for c in df.columns:
        for k, v in replace_dict.items():
            c = c.replace(k, v)
        cols.append(c)
    df.columns = cols
    return df


def pdidx(df, delimiter='|'):
    '''
    Flatten multiindex pandas dataframe and
    combine multiindex column names into a single level.
    '''
    idx = None
    try:
        idx = df.index
    except:
        pass
    col_lst = []
    for col in df.columns.values:
        if type(col) == tuple:
            if col[1] != '':
                col_lst += [delimiter.join(col).strip()]
            else:
                col_lst += [col[0]]
        else:
            col_lst += [col]
    df.columns=col_lst
    if idx is not None:
        df.index = idx
    return df


print('{:<25}: {}'.format(
    dt.now().strftime('%Y-%m-%d %H:%M:%S'),
    'Importing {}'.format(os.path.split(__file__)[1])), end='...')


def retry(fn, *args,
          n_retries=3, wait_secs=2, wait_style='constant', err_msg=None,
          start='', log=None, console=True, verbose=1):
    '''
    Pretty-retry function that can wrap another
    function and re-execute it on error.

    Parameters
    ----------
    fn : Python function
        Function to re-execute on error.
    *args : Arguments
        Any arguments that should be passed to fn in order of appearance.
    n_retries : int, optional
        Number of times fn should be executed. The default is 3.
    wait_secs : int, optional
        Number of seconds to wait if wait style is constant. The default is 2.
    wait_style : string, optional
        How to compute wait time.
        'constant': Retry will be executed after wait_secs seconds.
        'multiplicative': Retry after i * wait_secs in i-th retry
        'exponential': Retry after i ** wait_secs in i-th retry
        The default is 'constant'.
    err_msg : str, optional
        Message to pass to an exception.
        By default will pass exception name and fn name to error msg.
    start : str, optional
        start-parameter to pass to pprint. The default is ''.
    log : str, optional
        log-parameter to pass to pprint. The default is None.
    console : Flag, optional
        console-parameter to pass to pprint. The default is True.

    Raises
    ------
    ValueError if unrecognized wait_style is passed.

    Returns
    -------
    fn_out : variable
        Will return output from fn.

    '''
    try:
        fn_name = fn.__name__
    except:
        fn_name = 'unknown'
    if err_msg is None:
        err_msg = f'occured while executing function \'{fn_name}\''
    pprint(
        f'({1}/{n_retries}) Executing function \'{fn_name}\' with retry',
        start=start, log=log, console=console
    ) if verbose > 0 else None
    pprint(f'using {wait_style} wait time', log=log, console=console
           ) if verbose > 1 else None
    for i in range(1, n_retries + 1):
        try:
            fn_out = fn(*args)
            break
        except Exception as e:
            if wait_style == 'constant':
                wait_time = wait_secs
            elif wait_style == 'multiplicative':
                wait_time = i * wait_secs
            elif wait_style == 'exponential':
                wait_time = i ** wait_secs
            else:
                raise ValueError('Unrecognized wait_time parameter.')

            if i == n_retries:
                pprint(
                    f'({i}/{n_retries}) final retry failed. Exiting.',
                    start='\n',
                    log=log,
                    console=console
                ) if verbose > 1 else None
                pprint(e, log=log, console=False)
                raise e
            else:
                pprint(
                    f'{type(e).__name__} {e}',
                    start='\n',
                    log=log,
                    console=console
                ) if verbose > 1 else None
                wait_time_p = Sec2TC([wait_time])[0][0][:9]
                pprint(
                    f'({i}/{n_retries}) Re-executing in {wait_time_p}].',
                    end='...',
                    log=log,
                    console=console
                ) if verbose > 1 else None
                sleep(wait_time)

    return fn_out


def retry_until(
        val_fn, val_obj, val_res, fn, *args,
        n_retries=3, wait_secs=2, wait_style='constant', err_msg=None,
        start='', log=None, console=True, verbose=1):
    '''
    Pretty retry-until function that can wrap another function and re-execute
    it up to n_retries times or until val_fn(val_obj) == val_res.

    Parameters
    ----------
    val_fn : Python function
        Validation function to decide whether to retry or not.
    val_obj : variable
        Argument to pass to val_fn. Only a single argument may be passed.
    val_res : variable
        Expected result of validation. Evaluated against val_fn(val_obj).
    fn : Python function
        Function to re-execute on error.
    *args : Arguments
        Any arguments that should be passed to fn in order of appearance.
    n_retries : int, optional
        Number of times fn should be executed. The default is 3.
    wait_secs : int, optional
        Number of seconds to wait if wait style is constant. The default is 2.
    wait_style : string, optional
        How to compute wait time.
        'constant': Retry will be executed after wait_secs seconds.
        'multiplicative': Retry after i * wait_secs in i-th retry
        'exponential': Retry after i ** wait_secs in i-th retry
        The default is 'constant'.
    err_msg : str, optional
        Message to pass to an exception.
        By default will pass exception name and fn name to error msg.
    start : str, optional
        start-parameter to pass to pprint. The default is ''.
    log : str, optional
        log-parameter to pass to pprint. The default is None.
    console : Flag, optional
        console-parameter to pass to pprint. The default is True.

    Raises
    ------
    ValueError if unrecognized wait_style is passed.

    Returns
    -------
    fn_out : variable
        Will return output from fn.

    '''
    try:
        fn_name = fn.__name__
    except:
        fn_name = 'unknown'
    if err_msg is None:
        err_msg = f'occured while executing function \'{fn_name}\''
    pprint(
        f'({1}/{n_retries}) Executing function \'{fn_name}\' with retry',
        start=start, log=log, console=console
    ) if verbose > 0 else None
    pprint(
        f'using {wait_style} wait time',
        end='...', log=log, console=console) if verbose > 1 else None
    for i in range(1, n_retries + 1):
        try:
            fn_out = fn(*args)
            val_out = val_fn(val_obj)
            if val_out == val_res:
                break
            else:
                raise TypeError(
                    f'validation condition not met after {n_retries} retries. Expected {val_res}, got {val_out}.')
        except Exception as e:
            if wait_style == 'constant':
                wait_time = wait_secs
            elif wait_style == 'multiplicative':
                wait_time = i * wait_secs
            elif wait_style == 'exponential':
                wait_time = i ** wait_secs
            else:
                raise ValueError('Unrecognized wait_time parameter.')

            if i == n_retries:
                pprint(
                    f'({i}/{n_retries}) final retry failed. Exiting.',
                    start='\n',
                    log=log,
                    console=console
                ) if verbose > 1 else None
                raise e
            else:
                pprint(
                    f'{type(e).__name__} {e}',
                    start='\n',
                    log=log,
                    console=console
                ) if verbose > 1 else None
                wait_time_p = Sec2TC([wait_time])[0][0][:9]
                pprint(
                    f'({i}/{n_retries}) Re-executing in {wait_time_p}].',
                    end='...',
                    log=log,
                    console=console
                ) if verbose > 1 else None
                sleep(wait_time)
    print('done.') if verbose > 1 else None
    return fn_out


def validate_zip_mode_for(fp, is_zippable, target_zip_mode='copy',
                          copy_on_mode_error=False,
                          zip_log_fn='batches_zipnames.pkl'):
    '''
    Validates whether a desired target_zip_mode is applicable to a fp.

    Returns 'copy', 'zip' or 'unzip' depending on target_zip_mode,
    fp's type and type of files found in fp if fp is a zippable directory.

    Zippable directories are directories, for which (un)zipping should be
    applied and are defined in cfg['isZippable'].

    If target_zip_mode='zip', returns 'zip' if fp is a zippable directory and
    contains .png or .jpg files. For .zip files, it will return 'copy'.

    If target_zip_mode='unzip', returns 'unzip' if fp is a zippable directory
    and contains .zip files. For .png or .jpg files, it will return 'copy'.

    Otherwise returns 'copy'.

    Parameters
    ----------
    fp : str
        Path to directory or file.
    is_zippable : str
        Directory names that are zippable.
    target_zip_mode : str, optional
        Whether the desired mode is 'zip', 'unzip' or 'copy'.
        The default is 'copy'.
    copy_on_mode_error : boolean, optional
        Whether to raise an error or return 'copy' on error.
        The default is False.

    Raises
    ------
    ValueError
        Raises error if fp is a folder containing mixed file types.

    Returns
    -------
    str
        Zip mode 'copy', 'zip' or 'unzip'.

    '''
    if target_zip_mode not in ['copy', 'zip', 'unzip']:
        raise ValueError(f'unrecognized target_zip_mode {target_zip_mode}.')
    try:
        files = os.listdir(fp)
        files = [f for f in files if zip_log_fn not in f]
        files = [splitext(f)[1] for f in files]
        fileext = list(set(files))
    except NotADirectoryError:
        return 'copy'
    if len(fileext) > 1:
        if copy_on_mode_error:
            return 'copy'
        else:
            raise ValueError('mixed filetypes in folder')
    else:
        fileext = fileext[0]
    if psplit(fp)[1] in is_zippable:
        if fileext in ['.png', '.jpg']:
            if target_zip_mode == 'unzip':
                return 'copy'  # because is already unzipped
            else:
                return target_zip_mode
        elif fileext in ['.zip']:
            if target_zip_mode == 'zip':
                return 'copy'  # because is already zipped
            else:
                return target_zip_mode
    else:
        return 'copy'


def pmove(src, tgt, is_zippable=[], zip_mode='copy', batch_size=10,
          zip_log_fn='batches_zipnames.pkl',
          n_retries=3, wait_secs=2, wait_style='multiplicative',
          err_msg=None, start='', log=None, console=True, verbose=1):
    '''
    Wrapper function to move files using retry on error.

    If zip_mode='zip' or 'unzip' pmove will
    - zip if src is not zipped yet but is supposed to be zipped,
    - unzip if src is zipped and is supposed to be unzipped.

    If do_zip=False, will simply copy.

    Parameters
    ----------
    src : str
        Path to source object.
    tgt : str
        Path to target object.
    is_zippable : list, optional
        List of zippable objects. All other objects will be moved without zip.
    zip_mode : str, optional
        If 'zip', not zipped src will be zipped and moved if is in is_zippable;
        already zipped src and all objects not in is_zippable will be copied;
        If 'unzip',  zipped src will be unzipped and moved if is in is_zippable;
        already unzipped src and all objects not in is_zippable will be copied;
        The default is 'copy'.
    batch_size : int, optional
        Number of elements in zip batch. The default is 10.
    n_retries : int, optional
        Number of retries on error (see help(retry_until)). The default is 3.
    wait_secs : int, optional
        Number of wait seconds on error (see help(retry_until)).
        The default is 2.
    wait_style : str, optional
        Wait style on error (see help(retry_until)).
        The default is 'multiplicative'.
    err_msg : str, optional
        Error message to pass on to retry_until (see help(retry_until)).
        The default is None.
    start : str, optional
        Start string (see help(pprint)). The default is ''.
    log : str, optional
        Path to logfile (see help(pprint)). The default is None.
    console : boolean, optional
        Set console printout (see help(pprint)). The default is True.
    verbose : int, optional
        Verbosity parameter. The default is 1.

    Raises
    ------
    ValueError
        raised if zip mode of src can not be determined.

    Returns
    -------
    None.

    '''
    tgt_parent_fn, f = psplit(tgt)
    temp_tgt = pjoin(tgt_parent_fn, '_' + f)
    zip_mode = validate_zip_mode_for(
        src, is_zippable, zip_mode,
        copy_on_mode_error=True, zip_log_fn=zip_log_fn)
    if zip_mode == 'copy':
        # copy to tmp
        retry_until(
            checksums_are_equal, (src, temp_tgt), True,  # validate
            pcopy, src, temp_tgt,  # operation args
            n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
            err_msg=err_msg, start=start, log=log,
            console=console, verbose=verbose)  # cfg
    elif zip_mode == 'zip':
        batches, _, _ = get_batches(src, batch_size)
        # zip to tmp, will add a lookup file
        retry_until(
            checksum_equal_to, (temp_tgt, len(batches) + 1), True,  # validate
            pzip, src, temp_tgt, batch_size, zip_log_fn,  # operation args
            n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
            err_msg=err_msg, start=start, log=log,
            console=console, verbose=verbose)  # cfg
    elif zip_mode == 'unzip':
        # unzip to tmp
        zip_log_file = [f for f in os.listdir(src) if zip_log_fn in f][0]
        n_unzipped_files = int(zip_log_file.split('_')[1])
        retry_until(
            checksum_equal_to, (temp_tgt, n_unzipped_files), True,  # validate
            punzip, src, temp_tgt,  # operation args
            n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
            err_msg=err_msg, start=start, log=log,
            console=console, verbose=verbose)  # cfg
    else:
        raise ValueError(f'unrecognized mode parameter {zip_mode}')
    # rename tmp
    retry_until(
        exists, temp_tgt, False,  # validation args
        prename, temp_tgt, tgt,  # operation args
        n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
        err_msg=err_msg, start=start, log=log,
        console=console, verbose=verbose)  # cfg
    # delete src
    retry_until(
        exists, src, False,  # validation args
        pdelete, src,  # operation args
        n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
        err_msg=err_msg, start=start, log=log,
        console=console, verbose=verbose)  # cfg


def pzipcopy(src, tgt, is_zippable, zip_mode, batch_size=10,
             zip_log_fn='batches_zipnames.pkl',
             n_retries=3, wait_secs=2, wait_style='multiplicative',
             err_msg=None, start='', log=None, console=True, verbose=1):
    '''
    Wrapper function to zip or copy files using retry on error.
    Similar to pmove: If zip_mode='zip' or 'unzip' pmove will
    - zip if src is not zipped yet but is supposed to be zipped,
    - unzip if src is zipped and is supposed to be unzipped.

    If zip_mode='copy', will simply copy.

    Parameters
    ----------
    src : str
        Path to source object.
    tgt : str
        Path to target object.
    is_zippable : list, optional
        List of zippable objects. All other objects will be copied without zip.
    zip_mode : str, optional
        If 'zip', not zipped src will be zipped and copied if is in is_zippable;
        already zipped src and all objects not in is_zippable will be copied;
        If 'unzip',  zipped src will be unzipped and copied if is in is_zippable;
        already unzipped src and all objects not in is_zippable will be copied;
        The default is 'copy'.
    batch_size : int, optional
        Number of elements in zip batch. The default is 10.
    n_retries : int, optional
        Number of retries on error (see help(retry_until)). The default is 3.
    wait_secs : int, optional
        Number of wait seconds on error (see help(retry_until)).
        The default is 2.
    wait_style : str, optional
        Wait style on error (see help(retry_until)).
        The default is 'multiplicative'.
    err_msg : str, optional
        Error message to pass on to retry_until (see help(retry_until)).
        The default is None.
    start : str, optional
        Start string (see help(pprint)). The default is ''.
    log : str, optional
        Path to logfile (see help(pprint)). The default is None.
    console : boolean, optional
        Set console printout (see help(pprint)). The default is True.
    verbose : int, optional
        Verbosity parameter. The default is 1.

    Raises
    ------
    ValueError
        raised if zip mode of src can not be determined.

    Returns
    -------
    None.

    '''
    tgt_parent_fn, f = psplit(tgt)
    temp_tgt = pjoin(tgt_parent_fn, '_' + f)
    zip_mode = validate_zip_mode_for(
        src, is_zippable, zip_mode,
        copy_on_mode_error=True, zip_log_fn=zip_log_fn)
    if zip_mode == 'copy':
        # copy to tmp
        retry_until(
            checksums_are_equal, (src, temp_tgt), True,  # validate
            pcopy, src, temp_tgt,  # operation args
            n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
            err_msg=err_msg, start=start, log=log,
            console=console, verbose=verbose)  # cfg
    elif zip_mode == 'zip':
        batches, _, _ = get_batches(src, batch_size)
        # zip to tmp, will add a lookup file
        retry_until(
            checksum_equal_to, (temp_tgt, len(batches) + 1), True,  # validate
            pzip, src, temp_tgt, batch_size, zip_log_fn,  # operation args
            n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
            err_msg=err_msg, start=start, log=log,
            console=console, verbose=verbose)  # cfg
    elif zip_mode == 'unzip':
        # unzip to tmp
        zip_log_file = [f for f in os.listdir(src) if zip_log_fn in f][0]
        n_unzipped_files = int(zip_log_file.split('_')[1])
        retry_until(
            checksum_equal_to, (temp_tgt, n_unzipped_files), True,  # validate
            punzip, src, temp_tgt,  # operation args
            n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
            err_msg=err_msg, start=start, log=log,
            console=console, verbose=verbose)  # cfg
    else:
        raise ValueError(f'unrecognized mode parameter {zip_mode}')
    # rename tmp
    retry_until(
        exists, temp_tgt, False,  # validation args
        prename, temp_tgt, tgt,  # operation args
        n_retries=n_retries, wait_secs=wait_secs, wait_style=wait_style,
        err_msg=err_msg, start=start, log=log,
        console=console, verbose=verbose)  # cfg


def pzip(src, tgt, batch_size, zip_log_fn,
         exist_overwrite=True, exist_extend=False):
    tgt_parent_dir, tgt_f = psplit(normpath(tgt))
    src_parent_dir, src_f = psplit(normpath(src))
    tgt_f, tgt_ext = splitext(tgt_f)
    src_f, src_ext = splitext(src_f)
    tgt_is_dir = True if len(tgt_ext) == 0 else False
    if isfile(src):
        pmakedir(tgt)
    if isfile(src) and tgt_is_dir:
        tgt = pjoin(tgt, src_f + src_ext)
    elif isdir(src) and not(tgt_is_dir):
        tgt = pjoin(tgt_parent_dir, src_f + src_ext)
    if exists(tgt) and exist_overwrite:
        pdelete(tgt)
    elif exists(tgt) and exist_extend:
        i, tgt_n = 0, tgt
        while exists(tgt_n):
            i += 1
            tgt_n, ext = splitext(tgt)  # (dir, '') if tgt is directory
            tgt_n = f'{tgt_n}_{i}{ext}'
        tgt = tgt_n
    elif exists(tgt) and not(exist_overwrite) and not(exist_extend):
        raise ValueError(
            f'{tgt} exists but exist_overwrite={exist_overwrite}'
            + 'and exist_extend={exist_extend}')
    batches, zipnames, n_files = get_batches(src, batch_size, fullpaths=True)
    zipnames = [z.replace(src, tgt) for z in zipnames]
    for batch, zipname in zip(batches, zipnames):
        pmakedir(tgt)
        tozip(batch, zipname)
    with open(pjoin(tgt, f'_{n_files}_' + zip_log_fn), 'wb') as f:
        pickle.dump((batches, zipnames), f)


def punzip(src, tgt, exist_overwrite=True, exist_extend=False):
    tgt_parent_dir, tgt_f = psplit(normpath(tgt))
    src_parent_dir, src_f = psplit(normpath(src))
    tgt_f, tgt_ext = splitext(tgt_f)
    src_f, src_ext = splitext(src_f)
    tgt_is_dir = True if len(tgt_ext) == 0 else False
    if isfile(src):
        pmakedir(tgt)
    if isfile(src) and tgt_is_dir:
        tgt = pjoin(tgt, src_f + src_ext)
    elif isdir(src) and not(tgt_is_dir):
        tgt = pjoin(tgt_parent_dir, src_f + src_ext)
    if exists(tgt) and exist_overwrite:
        pdelete(tgt)
    elif exists(tgt) and exist_extend:
        i, tgt_n = 0, tgt
        while exists(tgt_n):
            i += 1
            tgt_n, ext = splitext(tgt)  # (dir, '') if tgt is directory
            tgt_n = f'{tgt_n}_{i}{ext}'
        tgt = tgt_n
    elif exists(tgt) and not(exist_overwrite) and not(exist_extend):
        raise ValueError(
            f'{tgt} exists but exist_overwrite={exist_overwrite}'
            + 'and exist_extend={exist_extend}')
    for zipname in os.listdir(src):
        if '.zip' in zipname:
            unzip(pjoin(src, zipname), tgt)


def pbackup(src, backup_extensions, backup_files, move_backup_to=None,
            backup_to_tmp=False, delete_src=True, log=None):
    '''
    Will backup all objects that qualify for backup inside src to
    a new folder src\_backup\<backupname>
    '''
    backup_dir = '_backups'
    tgt = pjoin(psplit(src)[0], '_' + psplit(src)[1]) if backup_to_tmp else src
    tgt = pjoin(
        tgt, backup_dir,
        psplit(src)[-1] + '_' + dt.now().strftime('%Y%m%d_%H%M%S'))
    pprint(f'Found {src}. Will create a backup.')
    pprint(f'Creating backup at {tgt}', log=log)
    pprint(f'Using backup_to_tmp={backup_to_tmp} and delete_src={delete_src}')
    pprint(f'Backed up extensions: {backup_extensions}')
    pprint(f'Backed up files: {backup_files}')
    files, files_to_backup = os.listdir(src), []
    for f in files:
        isBackupExtension = splitext(f)[1].replace(
            '.', '') in backup_extensions
        isBackupFile = splitext(f)[0] in backup_files
        if isBackupExtension or isBackupFile:
            files_to_backup.append(f)
    pprint(f'Found {len(files_to_backup)}/{len(files)} files to backup')
    if len(files_to_backup) == 0:
        pprint('Skipping backup')
    else:
        if backup_dir in files_to_backup:
            files_to_backup.remove(backup_dir)
            backup_subdirs = os.listdir(pjoin(src, backup_dir))
            backup_subdirs = [pjoin(backup_dir, f) for f in backup_subdirs]
            files_to_backup += backup_subdirs
            pprint(f'Added {len(backup_subdirs)} additional backup subdir(s)')
        for f in files_to_backup:
            from_fp = pjoin(src, f)
            to_fp = pjoin(
                psplit(tgt)[0], psplit(f)[-1]
            ) if backup_dir in f else pjoin(tgt, f)
            pprint(f'Creating backup for {f}', end='...', log=log)
            # NOTE: existing backup dirs will be backed up to themselves
            # which results in deletion
            if from_fp != to_fp:
                pcopy(from_fp, to_fp)
            print('done.')
        if delete_src and backup_to_tmp == True:
            pprint(f'Deleting {src}', end='...')
            pdelete(src)
        elif delete_src and backup_to_tmp == False:
            files_to_delete = [f for f in files if backup_dir not in f]
            pprint(f'Deleting {files_to_delete} in {src}', end='...')
            for f in files_to_delete:
                pdelete(pjoin(src, f))
            print('done.')
        if move_backup_to is not None:
            pmove(tgt, pjoin(move_backup_to, psplit(tgt)[1]), verbose=0)


def pjoin(*args):
    '''
    Join multiple strings or pandas series or mix of series and strings to a file path or uri.

    If first string starts with 'gs', a GCP compatible URI will be formed.

    Otherwise, will use 'os.path.join()'.

    Parameters
    ----------
    *args : tuple or listlike
        Collection of strings to be joined to a path or uri.

    Returns
    -------
    pj : str or list of strings
        File path.

    '''
    types_lst = [type(arg) for arg in args]
    n_args = len(args)
    arg_len = len(args[0])
    make_uri = False
    for a in args:
        arg_len = len(a) if len(a) > arg_len else arg_len
    types = types_lst[0]
    for t in types_lst:
        types = t if t == types else 'mixed'
    if types == str:
        if args[0][:2] == 'gs':
            make_uri = True
            pj = '/'.join(args)
        else:
            pj = os.path.join(*args)
    if types == 'mixed':
        args_srs = []
        for i in range(0, n_args):
            if types_lst[i] == str:
                args_srs += [Series([args[i] for _ in range(0, arg_len)])]
                types_lst[i] = pdSeriesType
        types = pdSeriesType
    else:
        args_srs = args
    if types == pdSeriesType:
        if args[0][0][:2] == 'gs':
            make_uri = True
            delim = '/'
        else:
            delim = os.sep
        pj = ''
        for i, arg in enumerate(args):
            pj += arg + delim if i < len(args)-1 else arg
    if make_uri:
        pj = pj.replace('\\\\', '/')
        pj = pj.replace('\\', '/')
    else:
        pj = pj.replace('/', '\\') if types == str else pj.str.replace('/', '\\')
    return pj


def pnow(datetime=None, for_filename=True):
    '''
    Returns a datetime.now() timestamp in default formats.

    Parameters
    ----------
    for_filename : boolean, optional
        If False will return now-timestamp optimized for readability
        using strftime format '%Y-%m-%d %H:%M:%S'.
        If True will return now-timestamp optimized for use as filename suffix
        format is  using strftime format '_%Y%m%d_%H%M%S'.
        The default is True.
    Returns
    -------
    str
        Datetime.now() timestamp as string.

    '''
    if datetime is None:
        datetime = dt.now()
    formatstring = '_%Y%m%d_%H%M%S' if for_filename else '%Y-%m-%d %H:%M:%S'
    return datetime.strftime(formatstring)


def pprint(msg,
           start='', end=None, log=None, console=True, returnstring=False):
    '''
    Build pretty console output from input string.
    Optionally specify logfile name to in addition print to logfile.
    '''
    timestamp = pnow(for_filename=False)
    prettystring = '{}{:<25}: {}'.format(start, timestamp, msg)
    if console:
        print(prettystring) if end is None else print(prettystring, end=end)
    if log is not None:
        f = open(log, 'a')
        f.write('\n' + prettystring)
        f.close()
    if returnstring:
        return prettystring


def prename(src, tgt, exist_overwrite=True, exist_extend=False):
    tgt_parent_dir, tgt_f = psplit(normpath(tgt))
    src_parent_dir, src_f = psplit(normpath(src))
    tgt_f, tgt_ext = splitext(tgt_f)
    src_f, src_ext = splitext(tgt_f)
    tgt_is_dir = True if len(tgt_ext) == 0 else False

    if isfile(src) and tgt_is_dir:
        tgt = pjoin(tgt, src_f)
    elif isdir(src) and not(tgt_is_dir):
        tgt = pjoin(tgt_parent_dir, src_f)

    if exists(tgt) and exist_overwrite:
        pdelete(tgt)
    elif exists(tgt) and exist_extend:
        i, tgt_n = 0, tgt
        while exists(tgt_n):
            i += 1
            tgt_n, ext = splitext(tgt)  # (dir, '') if tgt is directory
            tgt_n = f'{tgt_n}_{i}{ext}'
    elif exists(tgt) and not(exist_overwrite) and not(exist_extend):
        raise ValueError(
            f'{tgt} exists but exist_overwrite={exist_overwrite}'
            + 'and exist_extend={exist_extend}')

    rename(src, tgt)


def pcopy(src, tgt, exist_overwrite=True, exist_extend=False):
    tgt_parent_dir, tgt_f = psplit(normpath(tgt))
    src_parent_dir, src_f = psplit(normpath(src))
    tgt_f, tgt_ext = splitext(tgt_f)
    src_f, src_ext = splitext(src_f)
    tgt_is_dir = True if len(tgt_ext) == 0 else False
    if isfile(src):
        pmakedir(tgt)
    if isfile(src) and tgt_is_dir:
        tgt = pjoin(tgt, src_f + src_ext)
    elif isdir(src) and not(tgt_is_dir):
        tgt = pjoin(tgt_parent_dir, src_f + src_ext)
    if exists(tgt) and exist_overwrite:
        pdelete(tgt)
    elif exists(tgt) and exist_extend:
        i, tgt_n = 0, tgt
        while exists(tgt_n):
            i += 1
            tgt_n, ext = splitext(tgt)  # (dir, '') if tgt is directory
            tgt_n = f'{tgt_n}_{i}{ext}'
        tgt = tgt_n
    elif exists(tgt) and not(exist_overwrite) and not(exist_extend):
        raise ValueError(
            f'{tgt} exists but exist_overwrite={exist_overwrite}'
            + 'and exist_extend={exist_extend}')
    if isfile(src):
        copy(src, tgt)
    elif isdir(src):
        copytree(src, tgt)


def pdelete(fp):
    if isfile(fp):
        remove(fp)
    elif isdir(fp):
        rmtree(fp)


def pmakedir(fp, exist_overwrite=False):
    fp_parent_dir, f = psplit(normpath(fp))
    fn, ext = splitext(f)
    tgt = fp_parent_dir if len(ext) > 0 else fp
    if exist_overwrite and exists(tgt):
        pdelete(tgt)
    os.makedirs(tgt, exist_ok=True)


def plistsort(list1, list2, ascending=True):
    '''
    Sorts two lists together using the first as reference.

    Parameters
    ----------
    list1 : list
        List that will be used as sorting reference.
    list2 : list
        List that will be sorted in the same order as list1.
    ascending : boolean, optional
        True will sort ascending, False descending order. The default is True.

    Returns
    -------
    list1 : list
        Sorted list.
    list2 : list
        Sorted list.

    '''
    list1, list2 = \
        (list(t) for t in zip(*sorted(zip(list1, list2))))
    if ascending is False:
        list1.reverse()
        list2.reverse()
    return list1, list2


def plistdir(fp, extensions=None):
    '''
    Returns a list of all files in folder
    'fp' and its subfolders matching 'extension'.

    Extensions can be None for all, extension as string or several as list.
    '''
    files = []
    if extensions is not None:
        extensions = [extensions] if type(extensions) == str else extensions
        extensions = [e.replace('.', '') for e in extensions]
    # r=root, d=directories, f = files
    for r, d, f in os.walk(fp):
        for file in f:
            if extensions is None:
                files.append(pjoin(r, file))
            else:
                for extension in extensions:
                    if extension == os.path.splitext(file)[1].replace('.', ''):
                        files.append(pjoin(r, file))
    return files


def checksums_are_equal(obj_tuple):
    obj1, obj2 = obj_tuple
    return get_checksum(obj1) == get_checksum(obj2)


def checksum_equal_to(obj):
    fp, expected_n_files = obj
    return get_checksum(fp) == expected_n_files


def get_checksum(obj):
    '''
    Builds checksum depending on object type.
    - folder: returns number of files in folder
    - file: returns file size
    - object nonexistent: returns -1.0
    '''
    if os.path.exists(obj):
        if os.path.isdir(obj):
            return float(len(os.listdir(obj)))
        elif os.path.isfile(obj):
            return os.path.getsize(obj)
    else:
        return -1.0


def tozip(src_lst, tgt, if_exists='append', compress=True):
    if if_exists not in ['append', 'overwrite']:
        raise ValueError(f'unrecognised if_exist value {if_exists}')
    if compress:
        compression = zipfile.ZIP_DEFLATED
    else:
        compression = zipfile.ZIP_STORED
    i, t = 1, tgt
    while exists(tgt) and if_exists == 'append':
        tgt = os.path.splitext(t)[0] + '-' + str(i) + os.path.splitext(t)[1]
        i += 1
    with zipfile.ZipFile(tgt, 'w') as myzip:
        for f in src_lst:
            myzip.write(f, psplit(f)[1], compress_type=compression)


def unzip(src, tgt):
    '''
    Unzip a zip file at src filepath to target directory at tgt.
    Directory will be created if necessary.

    Parameters
    ----------
    src : str
        Source zip file.
    tgt : str
        Target directory.

    Returns
    -------
    None.

    '''
    pmakedir(tgt)
    with zipfile.ZipFile(src, 'r') as zip_ref:
        zip_ref.extractall(tgt)


def clean_race_info_str(text):
    # Remove links
    text = re.sub(r'http\S+', '', text)
    # Remove content within parentheses with exactly four '/'
    text = re.sub(r'\([^()]*\/[^()]*\/[^()]*\/[^()]*\/[^()]*\)', '', text)
    text = text.replace(' -', '').strip()
    text = text.lower().replace('_', '').replace(' ', '')
    return text

print('done.')
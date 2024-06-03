
import pandas as pd

def to_tstr(ts):
    tsn = ts
    if type(ts) == pd.Timedelta:
        tsn = pd.Timestamp('1900-01-01') + ts
    return tsn.strftime('%H:%M:%S')
"""Shared projection from the full modeling frame to the clustering feature matrix.

Both app.py (which runs the clustering) and unsupervised_ml_data_prep.py (which
writes the CSV on the daily refresh) need the exact same projection, so it lives
here as one source of truth. Previously it was duplicated in both files.

The projection keeps numeric columns, drops derived/metadata/redundant ones, and
fills remaining gaps with 0 -- which is only sound for columns where 0 is a real
value (e.g. minutes spent, counts). Columns where missing != 0 and 0 would be
actively misleading (rate/score metrics like run pace and golf differential) are
excluded here and left as honest NaN in modeling_data.csv instead.
"""

# Rate/score metrics where 0 is meaningless (0 min/mi reads as infinitely fast,
# 0 differential reads as a scratch golfer). Excluded from the clustering matrix.
RATE_LIKE_EXCLUDE = ['run_pace', 'golf_differential']

# Always-excluded derived / identifier / label columns.
BASE_EXCLUDE = ['date_column', 'Year', 'Month_Num', 'Day', 'DayOfYear', 'nap',
                'total', 'ovr_pirates', 'ovr_guardians', 'ovr_other', 'cycle_id']


def clustering_feature_columns(model_df):
    """Return the list of columns to exclude from the clustering matrix."""
    exclude = list(BASE_EXCLUDE) + list(RATE_LIKE_EXCLUDE)
    # Whoop metadata/timing (keep score_ columns: sleep, recovery, strain, HR, HRV).
    exclude += [c for c in model_df.columns if c.startswith('updated_at_')]
    exclude += [c for c in model_df.columns if c.endswith('_minutes_into_day')]
    # Duplicated zone-duration columns (keep zone_duration_, drop zone_durations_).
    exclude += [c for c in model_df.columns if 'zone_durations_' in c]
    # No-data / percent_recorded metadata.
    exclude += [c for c in model_df.columns if 'no_data' in c or 'percent_recorded' in c]
    return exclude


def clustering_numeric_frame(model_df):
    """Project model_df to the numeric clustering matrix (no date_column).

    Returns a frame on model_df's index: numeric columns only, derived/metadata
    columns dropped, all-zero and all-NaN columns dropped, remaining NaNs filled
    with 0. Callers that persist it add date_column back.
    """
    exclude = clustering_feature_columns(model_df)
    numeric_columns = model_df.select_dtypes(include=['float64', 'int64']).columns
    df = model_df[numeric_columns].drop(columns=exclude, errors='ignore')
    df = df.loc[:, (df != 0).any(axis=0)]      # drop all-zero columns
    df = df.loc[:, df.notna().any(axis=0)]     # drop all-NaN columns
    df = df.fillna(0)
    return df


def write_clustering_csv(model_df, path):
    """Build the clustering matrix and write it to path with date_column attached."""
    save_df = clustering_numeric_frame(model_df).copy()
    save_df['date_column'] = model_df['date_column']
    save_df.to_csv(path)
    return save_df

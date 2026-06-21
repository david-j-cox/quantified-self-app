#!/usr/bin/env python
"""Tier 2 clustering: labeled day-regimes over the recent multi-domain dense window.

The long-history view in app.py is an embedding/recurrence exploration on
always-present features (time + Whoop). This module is the complementary second
tier: over the recent window where food is logged densely, it assigns each day a
real regime label and shows how regimes move over time.

Design choices (see the daily-change-digest work):
  - Window is auto-detected: the earliest start from which food coverage stays
    dense through today, so it adapts as more data accrues.
  - Food (dense in the window) participates directly. Event-based pace/golf stay
    sparse even recently, so they enter only as honest binary indicators
    (ran / golfed) -- never a faked continuous 0.
  - Days without food logged are dropped from the window, so every clustered day
    genuinely has the multi-domain picture.
"""

import base64
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from clustering_features import clustering_feature_columns

RANDOM_STATE = 2225
FOOD_SIGNAL_COL = 'calories'   # presence of this == "food logged that day"


# --------------------------------------------------------------------------
# 1. Auto-detect the recent dense window
# --------------------------------------------------------------------------
def detect_dense_window(model_df, food_col=FOOD_SIGNAL_COL,
                        min_coverage=0.5, min_days=21):
    """Earliest start date from which food coverage stays dense through the end.

    Walks candidate start dates (days food was logged) from earliest to latest;
    coverage of [start, end] only rises as start advances, so the first start
    whose window clears min_coverage is the widest dense window. Falls back to
    the first logged day if nothing clears the bar.

    Returns (start_date, windowed_df) where windowed_df is model_df restricted to
    [start, end]. Returns (None, empty) if the food column is absent/empty.
    """
    if food_col not in model_df.columns:
        return None, model_df.iloc[0:0]

    df = model_df.copy()
    df['_date'] = pd.to_datetime(df['date_column']).dt.normalize()
    df = df.sort_values('_date')

    logged = df[df[food_col].notna()]
    if logged.empty:
        return None, model_df.iloc[0:0]

    end = df['_date'].max()
    logged_dates = logged['_date'].drop_duplicates().sort_values()

    chosen = logged_dates.iloc[0]  # fallback: first logged day
    for start in logged_dates:
        span_days = (end - start).days + 1
        if span_days < min_days:
            break  # remaining starts are even shorter
        window = df[(df['_date'] >= start) & (df['_date'] <= end)]
        covered = window[food_col].notna().sum()
        if covered / span_days >= min_coverage:
            chosen = start
            break

    windowed = df[df['_date'] >= chosen].drop(columns='_date')
    return chosen, windowed


# --------------------------------------------------------------------------
# 2. Build the food-inclusive regime feature matrix
# --------------------------------------------------------------------------
def build_regime_matrix(windowed_df, food_col=FOOD_SIGNAL_COL,
                        max_col_missing=0.3):
    """Complete-case multi-domain feature matrix for the window (no faked fills).

    Unlike the long-history projection, Tier 2 does NOT fillna(0): a missing
    Whoop score means "not worn", not zero, and faking it creates a bogus
    "low-Whoop" regime. So we keep values as-is, drop columns that are missing on
    too many window days, then keep only complete-case days. Time-allocation 0s
    are genuine ("didn't do it") and stay. Event-based pace/golf enter only as
    honest binary `ran` / `golfed` indicators.
    """
    if windowed_df.empty or food_col not in windowed_df.columns:
        return pd.DataFrame()

    df = windowed_df[windowed_df[food_col].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    dates = pd.to_datetime(df['date_column']).dt.normalize()
    exclude = clustering_feature_columns(df)
    numeric = df.select_dtypes(include=['float64', 'int64']).columns
    feats = df[numeric].drop(columns=exclude, errors='ignore').copy()
    feats.index = dates.values

    # Honest event indicators (always defined: 1 if it happened, else 0).
    if 'run_pace' in df.columns:
        feats['ran'] = df['run_pace'].notna().astype(int).values
    if 'golf_differential' in df.columns:
        feats['golfed'] = df['golf_differential'].notna().astype(int).values

    # Time/activity/food zeros are genuine ("didn't do it"), so a missing value
    # there means 0. A missing Whoop score (score_*) means "not worn", which we
    # will not fake -- those days drop out via the complete-case step below.
    score_cols = [c for c in feats.columns if c.startswith('score_')]
    non_score = [c for c in feats.columns if c not in score_cols]
    feats[non_score] = feats[non_score].fillna(0)

    # Drop near-empty / all-zero / constant columns, then keep complete-case days
    # (which removes only the days the Whoop was not worn).
    feats = feats.loc[:, feats.notna().mean() >= (1 - max_col_missing)]
    feats = feats.loc[:, (feats.fillna(0) != 0).any(axis=0)]
    feats = feats.dropna()
    feats = feats.loc[:, feats.std(ddof=0) > 1e-9]
    feats.index.name = 'date'
    return feats


# --------------------------------------------------------------------------
# 3. Assign regimes
# --------------------------------------------------------------------------
def assign_regimes(features, k_range=range(2, 6), random_state=RANDOM_STATE):
    """Scale, pick k by silhouette, and assign an agglomerative regime per day.

    Returns dict: labels (Series indexed by date), k, scaled (DataFrame),
    embedding2d (ndarray n x 2 via PCA for viz), silhouettes (dict k->score).
    Returns None if there are too few days to cluster.
    """
    if features.empty or len(features) < max(6, min(k_range) + 1):
        return None

    # PCA's SVD on a rank-deficient matrix (more features than days) emits benign
    # divide/overflow RuntimeWarnings in numpy's matmul; the results are sound.
    # Suppress just the floating-point warnings around the numeric work.
    with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
        scaler = StandardScaler()
        X = scaler.fit_transform(features.values)
        # A near-constant column gets a tiny std, so its rare outlier scales to a
        # huge z. Clip z-scores to a sane range and scrub any non-finite values.
        X = np.clip(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), -8, 8)

        # Reduce before clustering: with more features than days the covariance
        # is singular (noisy distances). Project to a handful of PCs capping at
        # n_samples-1, which also denoises the regime structure.
        n_samples, n_features = X.shape
        n_pcs = max(2, min(10, n_features, n_samples - 1))
        pca = PCA(n_components=n_pcs, random_state=random_state)
        X_red = pca.fit_transform(X)

        # Pick k by best silhouette, but reject solutions with a singleton regime
        # (a one-day "regime" is noise, not a pattern). Cap k for small windows.
        sils = {}
        max_k = min(max(k_range), max(2, n_samples // 6))
        best_k, best_score, best_labels = None, -1.0, None
        for k in k_range:
            if k > max_k or k >= n_samples:
                continue
            labels = AgglomerativeClustering(n_clusters=k).fit_predict(X_red)
            _, counts = np.unique(labels, return_counts=True)
            if counts.min() < 2:
                continue  # no singleton regimes
            score = silhouette_score(X_red, labels)
            sils[k] = float(score)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        if best_labels is None:  # fall back to k=2 if nothing else qualified
            best_labels = AgglomerativeClustering(n_clusters=2).fit_predict(X_red)
            best_k, best_score = 2, float(silhouette_score(X_red, best_labels))

    # Relabel regimes by first appearance so "Regime 1" is the earliest one.
    order = pd.unique(best_labels)
    remap = {old: new for new, old in enumerate(order)}
    relabeled = np.array([remap[v] for v in best_labels])

    embedding2d = X_red[:, :2] if X_red.shape[1] >= 2 \
        else np.column_stack([X_red[:, 0], np.zeros(len(X_red))])

    return {
        'labels': pd.Series(relabeled, index=features.index, name='regime'),
        'k': best_k,
        'silhouette': best_score,
        'silhouettes': sils,
        'scaled': pd.DataFrame(X, index=features.index, columns=features.columns),
        'embedding2d': embedding2d,
    }


# --------------------------------------------------------------------------
# 4. Characterize each regime
# --------------------------------------------------------------------------
def profile_regimes(features, labels, top_n=5):
    """Per regime: size, date range, and the features that most define it.

    Distinctiveness = z-scored regime mean vs the overall mean (so it reads as
    'this regime runs high on X, low on Y' relative to a typical window day).
    """
    z = (features - features.mean()) / features.std(ddof=0).replace(0, np.nan)
    rows = []
    for regime in sorted(labels.unique()):
        idx = labels[labels == regime].index
        member_z = z.loc[idx].mean().dropna().sort_values()
        # Show each regime's most distinctive features by direction, always, so
        # even the near-baseline cluster gets a readable profile.
        highs = member_z[member_z > 0].tail(top_n)[::-1]
        lows = member_z[member_z < 0].head(top_n)
        rows.append({
            'Regime': f'Regime {regime + 1}',
            'Days': int(len(idx)),
            'From': str(min(idx).date()),
            'To': str(max(idx).date()),
            'Runs high on': ', '.join(highs.index) or '(near typical)',
            'Runs low on': ', '.join(lows.index) or '(near typical)',
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Transitions over time
# --------------------------------------------------------------------------
def summarize_transitions(labels):
    """Counts, current regime, and the run lengths of each regime stint."""
    seq = labels.sort_index()
    changes = (seq.values[1:] != seq.values[:-1]).sum()
    # Length of the current (most recent) uninterrupted regime stint.
    cur = seq.iloc[-1]
    run = 0
    for v in seq.values[::-1]:
        if v == cur:
            run += 1
        else:
            break
    return {
        'n_days': int(len(seq)),
        'n_regimes': int(seq.nunique()),
        'n_transitions': int(changes),
        'current_regime': f'Regime {int(cur) + 1}',
        'current_run_days': int(run),
    }


# --------------------------------------------------------------------------
# 6. Plots
# --------------------------------------------------------------------------
def _b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return out


def plot_regime_timeline(labels):
    """Step plot of regime over time (which regime each day falls in)."""
    seq = labels.sort_index()
    k = int(seq.max()) + 1
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.scatter(seq.index, seq.values + 1, c=[cmap(v) for v in seq.values],
               s=60, zorder=3)
    ax.step(seq.index, seq.values + 1, where='post', color='#9ca3af',
            linewidth=1, alpha=0.6, zorder=1)
    ax.set_yticks(range(1, k + 1))
    ax.set_yticklabels([f'Regime {i}' for i in range(1, k + 1)])
    ax.set_xlabel('Date', fontsize=14, labelpad=8)
    ax.set_title('Daily regime over the recent dense window', fontsize=16,
                 fontweight='bold', pad=12)
    ax.tick_params(labelsize=11)
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha('right')
    fig.tight_layout()
    return _b64(fig)


def plot_regime_embedding(embedding2d, labels):
    """2D PCA scatter of the window's days, colored by regime."""
    seq = labels.sort_index()
    emb = pd.DataFrame(embedding2d, index=labels.index, columns=['x', 'y']).loc[seq.index]
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(8, 7))
    for regime in sorted(seq.unique()):
        m = seq == regime
        ax.scatter(emb['x'][m.values], emb['y'][m.values], s=70, alpha=0.8,
                   color=cmap(regime), label=f'Regime {regime + 1}',
                   edgecolors='white', linewidth=0.5)
    ax.set_xlabel('PC 1', fontsize=14, labelpad=8)
    ax.set_ylabel('PC 2', fontsize=14, labelpad=8)
    ax.set_title('Recent days in feature space, by regime', fontsize=16,
                 fontweight='bold', pad=12)
    ax.legend(frameon=False, fontsize=11, loc='upper left',
              bbox_to_anchor=(1.02, 1.0))
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    return _b64(fig)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def build_recent_regimes(model_df):
    """Full Tier 2 pipeline. Returns a dict of artifacts, or {'ok': False, ...}."""
    start, windowed = detect_dense_window(model_df)
    if windowed.empty:
        return {'ok': False, 'reason': 'no dense food-logged window yet'}

    features = build_regime_matrix(windowed)
    result = assign_regimes(features)
    if result is None:
        return {'ok': False, 'reason': f'too few days to cluster ({len(features)})'}

    labels = result['labels']
    return {
        'ok': True,
        'window_start': str(pd.Timestamp(start).date()),
        'window_end': str(labels.index.max().date()),
        'n_features': features.shape[1],
        'k': result['k'],
        'silhouette': round(result['silhouette'], 3),
        'profiles': profile_regimes(features, labels),
        'transitions': summarize_transitions(labels),
        'timeline_b64': plot_regime_timeline(labels),
        'embedding_b64': plot_regime_embedding(result['embedding2d'], labels),
        'labels': labels,
    }


def _load_model_df():
    """Standalone: read the modeling table (fresh) or fall back to the CSV."""
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    from sqlalchemy import create_engine
    load_dotenv()
    try:
        url = (f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
               f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")
        return pd.read_sql("SELECT * FROM modeling_ready_data", create_engine(url))
    except Exception as e:
        print(f"  DB read failed ({e}); using modeling_data.csv")
        csv = Path(__file__).resolve().parent.parent / 'Data' / 'modeling_data.csv'
        return pd.read_csv(csv)


def main():
    out = build_recent_regimes(_load_model_df())
    if not out['ok']:
        print(f"Tier 2 regimes unavailable: {out['reason']}")
        return
    print(f"Recent dense window: {out['window_start']} -> {out['window_end']}")
    print(f"{out['n_features']} features, k={out['k']} "
          f"(silhouette {out['silhouette']})")
    t = out['transitions']
    print(f"{t['n_days']} days, {t['n_regimes']} regimes, "
          f"{t['n_transitions']} transitions. Currently {t['current_regime']} "
          f"for {t['current_run_days']} day(s).\n")
    with pd.option_context('display.max_colwidth', 60, 'display.width', 140):
        print(out['profiles'].to_string(index=False))


if __name__ == '__main__':
    main()

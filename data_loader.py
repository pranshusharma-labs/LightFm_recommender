"""
data_loader.py
──────────────
Loads, cleans, and splits products + events data.
Used by both train.py and api.py.
"""

import pandas as pd
import numpy as np


EVENT_WEIGHTS = {
    'view_item':   0.3,
    'add_to_cart': 0.7,
    'purchase':    1.0,
}


def load_products(path: str) -> pd.DataFrame:
    """Load and clean products CSV."""
    df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
    df['product_id']  = df['product_id'].astype(str).str.strip()
    df['title']       = df['title'].fillna('').astype(str).str.strip()
    df['category']    = df['category'].fillna('unknown').astype(str).str.strip()
    df['brand']       = df['brand'].fillna('').astype(str).str.strip()
    df['variant']     = df['variant'].fillna('').astype(str).str.strip()
    df['price']       = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['description'] = df['description'].fillna('').astype(str)
    df['tags']        = df['tags'].fillna('').astype(str)

    # One canonical row per product_id
    if 'appearance_count' in df.columns:
        df = (df.sort_values('appearance_count', ascending=False)
                .drop_duplicates('product_id')
                .reset_index(drop=True))
    else:
        df = df.drop_duplicates('product_id').reset_index(drop=True)

    print(f"[Products] {len(df):,} unique products | {df['category'].nunique()} categories")
    return df


def build_item_text(row: pd.Series) -> str:
    """
    Build rich text string for each product.
    Used by content-based models (BGE etc.) if needed later.
    """
    parts = [row['title']]
    if row['brand'] and row['brand'] not in ['Google', '', 'nan']:
        parts.append(f"Brand: {row['brand']}")
    cat = row['category'].replace('Home/', '').replace('/', ' ').strip()
    if cat and cat != 'unknown':
        parts.append(f"Category: {cat}")
    if row['variant'] and row['variant'] not in ['Single Option Only', '', 'nan']:
        parts.append(f"Size: {row['variant'].strip()}")
    tags = row['tags'].replace('(not set)', '').strip().strip(',')
    if tags and tags != row['category']:
        parts.append(f"Tags: {tags}")
    if row['price'] > 0:
        parts.append(f"Price: ${row['price']:.0f}")
    return '. '.join(parts)


def load_events(path: str) -> pd.DataFrame:
    """Load and clean events CSV."""
    df = pd.read_csv(path, encoding='latin-1', on_bad_lines='skip')
    df.columns       = df.columns.str.strip()
    df['user_id']    = df['user_id'].astype(str).str.strip()
    df['item_id']    = df['item_id'].astype(str).str.strip()
    df['event_name'] = df['event_name'].str.strip()
    df['item_name']  = df['item_name'].fillna('').str.strip()
    df['quantity']   = pd.to_numeric(df.get('quantity', 1), errors='coerce').fillna(1).clip(lower=1)
    df['price']      = pd.to_numeric(df.get('price', 0),    errors='coerce').fillna(0)
    df['timestamp']  = pd.to_datetime(df['event_timestamp'], utc=True, errors='coerce')

    # Keep only known event types
    df = df[df['event_name'].isin(EVENT_WEIGHTS.keys())].copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    print(f"[Events] {len(df):,} rows | {df['user_id'].nunique():,} users")
    print(f"[Events] Event types:\n{df['event_name'].value_counts().to_string()}")
    return df


def resolve_item_ids(events: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """
    Bridge numeric purchase item_ids to product_ids via item_name.
    Purchase events use numeric IDs; browse events use alphanumeric product_ids.
    """
    prod_title_to_id = dict(zip(
        products['title'].str.strip().str.lower(),
        products['product_id']
    ))

    def resolve(row):
        if row['item_id'].isdigit():
            return prod_title_to_id.get(row['item_name'].lower(), row['item_id'])
        return row['item_id']

    events = events.copy()
    events['item_id'] = events.apply(resolve, axis=1)

    resolved = events['item_id'].isin(set(products['product_id'])).sum()
    print(f"[Events] Items resolved to catalog: {resolved:,}/{len(events):,}")
    return events


def apply_weights_and_decay(events: pd.DataFrame) -> pd.DataFrame:
    """
    Apply event weights and time decay.
    weight = event_weight × 1/(1 + days_ago)
    """
    events = events.copy()
    events['weight']   = events['event_name'].map(EVENT_WEIGHTS).fillna(0.1)
    max_time           = events['timestamp'].max()
    events['days_ago'] = (max_time - events['timestamp']).dt.days
    events['weight']   = events['weight'] * (1.0 / (1.0 + events['days_ago']))
    return events


def split_data(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Airtight temporal + session-boundary split. Zero data leakage.

    Strategy:
    ─────────
    1. Detect sessions per user using 30-min inactivity gaps
    2. Assign each session a globally unique ID
    3. Find global 80/20 time cutoff
    4. Train  = sessions that COMPLETELY ended before cutoff
    5. Test   = sessions that COMPLETELY started after cutoff
    6. Sessions that straddle the cutoff are DROPPED from both
       → this is the key to zero leakage: no session appears in both sets

    Why this is better than simple row-level split:
    ─────────────────────────────────────────────────
    A row-level split (e.g. 80% of rows = train) can put two rows
    from the same shopping session into train and test simultaneously.
    That creates micro-leakage: the model sees partial session context
    in train that also appears as a test target. Session-boundary split
    prevents this entirely by treating each session as an atomic unit.
    """
    events = events.sort_values(['user_id', 'timestamp']).copy()

    # ── Step 1: detect sessions (30-min gap = new session) ──
    events['time_diff']  = events.groupby('user_id')['timestamp'].diff()
    events['new_session'] = (
        events['time_diff'].isna() |
        (events['time_diff'] > pd.Timedelta(minutes=30))
    ).astype(int)
    events['session_local'] = events.groupby('user_id')['new_session'].cumsum()
    events['session_id']    = (
        events['user_id'].astype(str) + '_' +
        events['session_local'].astype(str)
    )

    # ── Step 2: find session start/end times ──
    session_bounds = events.groupby('session_id')['timestamp'].agg(['min', 'max'])

    # ── Step 3: global time cutoff at 80th percentile ──
    cutoff = events['timestamp'].quantile(0.80)
    print(f"[Split] Cutoff: {cutoff}")

    # ── Step 4: assign sessions to train or test (no overlap allowed) ──
    train_sessions = set(session_bounds[session_bounds['max'] <  cutoff].index)
    test_sessions  = set(session_bounds[session_bounds['min'] >= cutoff].index)
    straddling     = len(session_bounds) - len(train_sessions) - len(test_sessions)

    train_df = events[events['session_id'].isin(train_sessions)].copy()
    test_df  = events[events['session_id'].isin(test_sessions)].copy()

    # ── Step 5: tag cold-start users in test ──
    known_users = set(train_df['user_id'].unique())
    test_df['is_cold_start'] = ~test_df['user_id'].isin(known_users)

    cold = test_df['is_cold_start'].sum()
    print(f"[Split] Train: {len(train_df):,} | Test: {len(test_df):,}")
    print(f"[Split] Sessions — train: {len(train_sessions)} | "
          f"test: {len(test_sessions)} | dropped (straddling): {straddling}")
    print(f"[Split] Cold-start events in test: {cold:,}")

    # Drop helper columns
    drop_cols = ['time_diff', 'new_session', 'session_local', 'session_id']
    train_df  = train_df.drop(columns=drop_cols).reset_index(drop=True)
    test_df   = test_df.drop(columns=drop_cols).reset_index(drop=True)

    return train_df, test_df

"""Build dataset pipeline module for Hit Song Predictor.

Turns data/raw/ into modelling tables:
- data/processed/songs_full.parquet (pre-downsampling, natural imbalance)
- data/processed/songs.parquet (10:1 negative-to-positive ratio)

Target definition:
The target means "the track appeared in the Billboard Year-End Hot 100, 2000-2023".
"""

import os
import re
import string
import unicodedata
import numpy as np
import pandas as pd

VERSION_PATTERNS = [
    re.compile(r'[\(\[\{].*?\b(?:feat\.?|featuring|ft\.?)\b.*?[\)\]\}]', re.IGNORECASE),
    re.compile(r'-\s*\b(?:feat\.?|featuring|ft\.?)\b.*$', re.IGNORECASE),
    re.compile(r'\b(?:feat\.?|featuring|ft\.?)\b.*$', re.IGNORECASE),
    re.compile(r'[\(\[\{].*?\b(?:remaster(?:ed)?|radio edit|single version|album version|original mix|extended mix|deluxe(?: edition)?|bonus track)\b.*?[\)\]\}]', re.IGNORECASE),
    re.compile(r'-\s*\b(?:remaster(?:ed)?|radio edit|single version|album version|original mix|extended mix|deluxe(?: edition)?|bonus track)\b.*$', re.IGNORECASE),
    re.compile(r'\b(?:remaster(?:ed)?|radio edit|single version|album version|original mix|extended mix|deluxe(?: edition)?|bonus track)\b', re.IGNORECASE),
]

AMBIGUOUS_PATTERN = re.compile(
    r'\b(?:live|remix|acoustic|karaoke|cover|instrumental)\b',
    re.IGNORECASE,
)

CANONICAL_FEATURES = [
    'tempo', 'energy', 'danceability', 'valence', 'loudness',
    'acousticness', 'instrumentalness', 'speechiness', 'duration_sec', 'genre'
]


def normalize_text(text: str, is_track: bool = True) -> str:
    """Normalise string by lowercasing, stripping accents, punctuation, and release tags."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[’‘´`]", "'", text)
    text = re.sub(r'[“”"]', ' ', text)
    if is_track:
        for pat in VERSION_PATTERNS:
            text = pat.sub(' ', text)
    text = text.replace("'", "")
    for p in string.punctuation:
        text = text.replace(p, " ")
    return re.sub(r'\s+', ' ', text).strip()


def extract_primary_artist(artist: str, is_billboard: bool = False) -> str:
    """Extract and normalise primary artist without splitting band names."""
    if not isinstance(artist, str):
        return ""
    if is_billboard:
        primary = re.split(r'\b(?:feat\.?|featuring|ft\.?)\b', artist, flags=re.IGNORECASE)[0]
    else:
        primary = artist.split(';')[0]
        primary = re.split(r'\b(?:feat\.?|featuring|ft\.?)\b', primary, flags=re.IGNORECASE)[0]
    return normalize_text(primary, is_track=False)


def load_raw(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read raw CSV files and standardise column names to snake_case."""
    ds_path = os.path.join(data_dir, "dataset.csv")
    bb_path = os.path.join(data_dir, "billboard_24years_lyrics_spotify.csv")
    df_ds = pd.read_csv(ds_path)
    df_bb = pd.read_csv(bb_path)
    df_ds.columns = [c.strip().lower().replace(" ", "_") for c in df_ds.columns]
    df_bb.columns = [c.strip().lower().replace(" ", "_") for c in df_bb.columns]
    return df_ds, df_bb


def harmonise(df_ds: pd.DataFrame) -> pd.DataFrame:
    """Derive canonical duration_sec, genre, primary_artist, and normalisation keys."""
    df = df_ds.copy()
    df['duration_sec'] = df['duration_ms'] / 1000.0
    df['genre'] = df['track_genre']
    df['primary_artist'] = df['artists'].apply(lambda x: extract_primary_artist(x, is_billboard=False))
    df['norm_track'] = df['track_name'].apply(lambda x: normalize_text(x, is_track=True))
    df['norm_key'] = df['norm_track'] + "___" + df['primary_artist']
    return df


def assign_chart_year(df_clean: pd.DataFrame, df_bb: pd.DataFrame) -> pd.DataFrame:
    """Assign the earliest Billboard year to each positive track; null for negatives."""
    id_to_year = df_bb.groupby('bb_track_id')['year'].min().to_dict()
    key_to_year = df_bb.groupby('norm_key')['year'].min().to_dict()
    df_clean['chart_year'] = np.nan
    pos_mask = df_clean['charted'] == 1
    years = [
        min([y for y in [id_to_year.get(t), key_to_year.get(k)] if y is not None], default=np.nan)
        for t, k in zip(df_clean.loc[pos_mask, 'track_id'], df_clean.loc[pos_mask, 'norm_key'])
    ]
    df_clean.loc[pos_mask, 'chart_year'] = years
    return df_clean


def create_reports(df_ds: pd.DataFrame, df_bb: pd.DataFrame, is_name_only: pd.Series, is_ambig: pd.Series) -> None:
    """Generate match_sample.csv and ambiguous_excluded.csv reports."""
    os.makedirs("reports/tables", exist_ok=True)
    ambig_df = df_ds[is_ambig].copy()
    ambig_df.to_csv("reports/tables/ambiguous_excluded.csv", index=False)
    name_only_df = df_ds[is_name_only & ~is_ambig].copy()
    sample_40 = name_only_df.sample(n=min(40, len(name_only_df)), random_state=42)
    bb_map = {r['norm_key']: (r['song'], r['band_singer']) for _, r in df_bb.iterrows()}
    sample_rows = [{
        'Billboard song': bb_map.get(r['norm_key'], ("", ""))[0],
        'Billboard artist': bb_map.get(r['norm_key'], ("", ""))[1],
        'matched dataset.csv track_name': r['track_name'],
        'matched artists': r['artists'],
        'whether the match was also an ID match': False
    } for _, r in sample_40.iterrows()]
    pd.DataFrame(sample_rows).to_csv("reports/tables/match_sample.csv", index=False)


def recall_check(df_ds: pd.DataFrame, df_bb: pd.DataFrame) -> None:
    """Report Billboard recall and output unmatched_billboard_sample.csv."""
    os.makedirs("reports/tables", exist_ok=True)
    bb_unique = df_bb[df_bb['norm_key'] != "___"].drop_duplicates(subset=['norm_key']).copy()
    ds_keys = set(df_ds['norm_key'])
    matched_bb = bb_unique['norm_key'].isin(ds_keys)
    unmatched_bb = bb_unique[~matched_bb].copy()

    ds_title_map = {}
    for _, r in df_ds.iterrows():
        t = r['norm_track']
        if t and t not in ds_title_map:
            ds_title_map[t] = (r['track_name'], r['artists'])

    unmatched_bb['candidate_track'] = unmatched_bb['norm_track'].map(lambda t: ds_title_map.get(t, ("", ""))[0])
    unmatched_bb['candidate_artists'] = unmatched_bb['norm_track'].map(lambda t: ds_title_map.get(t, ("", ""))[1])
    has_cand = (unmatched_bb['candidate_track'] != "").sum()

    sample_40_unmatched = unmatched_bb.sample(n=min(40, len(unmatched_bb)), random_state=42)
    pd.DataFrame({
        'Billboard song': sample_40_unmatched['song'],
        'Billboard artist': sample_40_unmatched['band_singer'],
        'matched title-only candidate track_name': sample_40_unmatched['candidate_track'],
        'candidate artists': sample_40_unmatched['candidate_artists']
    }).to_csv("reports/tables/unmatched_billboard_sample.csv", index=False)

    print("\n--- Recall Check ---")
    print(f"unique normalised Billboard keys matched: {matched_bb.sum()}")
    print(f"unique normalised Billboard keys unmatched: {(~matched_bb).sum()}")
    print(f"unmatched Billboard keys with >= 1 title-only candidate: {has_cand}")


def match_labels(df_ds: pd.DataFrame, df_bb: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Match dataset rows to Billboard using Spotify track ID or normalised track + artist."""
    df_bb['bb_track_id'] = df_bb['uri'].str.replace('spotify:track:', '', regex=False).str.strip()
    df_bb['norm_track'] = df_bb['song'].apply(lambda x: normalize_text(x, is_track=True))
    df_bb['norm_artist'] = df_bb['band_singer'].apply(lambda x: extract_primary_artist(x, is_billboard=True))
    df_bb['norm_key'] = df_bb['norm_track'] + "___" + df_bb['norm_artist']

    is_id = df_ds['track_id'].isin(set(df_bb['bb_track_id'].dropna()))
    is_name = df_ds['norm_key'].isin(set(df_bb[df_bb['norm_key'] != "___"]['norm_key']))
    is_name_only = is_name & (~is_id)
    is_ambig = is_name_only & df_ds['track_name'].astype(str).str.contains(AMBIGUOUS_PATTERN, na=False)

    df_clean = df_ds[~is_ambig].copy()
    is_pos = is_id.loc[df_clean.index] | (is_name_only.loc[df_clean.index] & ~is_ambig.loc[df_clean.index])
    df_clean['charted'] = np.where(is_pos, 1, 0)

    key_counts = df_clean.groupby('norm_key')['charted'].nunique()
    conflict_keys = set(key_counts[key_counts > 1].index)
    if conflict_keys:
        df_clean.loc[df_clean['norm_key'].isin(conflict_keys), 'charted'] = 1

    create_reports(df_ds, df_bb, is_name_only, is_ambig)
    recall_check(df_ds, df_bb)
    df_clean = assign_chart_year(df_clean, df_bb)

    stats = {
        "matched_by_id": int(is_id.sum()),
        "matched_by_name_only": int(is_name_only.sum()),
        "ambiguous_excluded": int(is_ambig.sum()),
        "conflicts_resolved": len(conflict_keys),
    }
    return df_clean, stats


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop post-outcome variables, leakage, and out-of-scope metadata columns."""
    # Post-outcome / leakage / out of scope: drop ranking, songurl, url, titletext, lyrics,
    # popularity, unnamed:_0, album_name, explicit, track_id and non-canonical metadata
    leakage_cols = [
        'unnamed:_0', 'popularity', 'album_name', 'explicit', 'track_id',
        'track_name', 'artists', 'duration_ms', 'track_genre', 'key', 'mode'
    ]
    return df.drop(columns=[c for c in leakage_cols if c in df.columns])


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Deduplicate exact duplicates first, then collapse multi-genre rows with seeded random pick."""
    exact_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    before_genre = len(df)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    df = df.drop_duplicates(subset=['norm_key'], keep='first').reset_index(drop=True)
    genre_count = before_genre - len(df)
    return df, exact_count, genre_count


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing core identifiers or failed-analysis zero markers."""
    print("\n--- Missing Value Counts Before handle_missing ---")
    print(df[CANONICAL_FEATURES + ['primary_artist', 'charted']].isnull().sum().to_string())
    invalid = (
        df['primary_artist'].isnull() | (df['primary_artist'] == "") |
        df['norm_track'].isnull() | (df['norm_track'] == "") |
        (df['tempo'] == 0) | (df['duration_sec'] == 0) | (df['time_signature'] == 0)
    )
    df_clean = df[~invalid].reset_index(drop=True)
    print("\n--- Missing Value Counts After handle_missing ---")
    print(df_clean[CANONICAL_FEATURES + ['primary_artist', 'charted']].isnull().sum().to_string())
    for col in CANONICAL_FEATURES:
        assert df_clean[col].isnull().sum() == 0, f"Missing value remaining in canonical feature {col}"
    return df_clean


def handle_outliers(df: pd.DataFrame) -> tuple[pd.DataFrame, float, float, int]:
    """Winsorise duration_sec to the 1st and 99th percentiles."""
    p01 = float(df['duration_sec'].quantile(0.01))
    p99 = float(df['duration_sec'].quantile(0.99))
    affected = int(((df['duration_sec'] < p01) | (df['duration_sec'] > p99)).sum())
    df['duration_sec'] = df['duration_sec'].clip(lower=p01, upper=p99)
    return df, p01, p99, affected


def collapse_genre(df: pd.DataFrame) -> pd.DataFrame:
    """Skip genre collapsing per project specification; preserve all raw genres."""
    return df


def downsample_negatives(df: pd.DataFrame, ratio: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep all positives and sample 10 negatives per positive (random_state=42)."""
    os.makedirs("data/processed", exist_ok=True)
    df.to_parquet("data/processed/songs_full.parquet", index=False)
    positives = df[df['charted'] == 1].copy()
    negatives = df[df['charted'] == 0].copy()
    sampled_neg = negatives.sample(n=ratio * len(positives), random_state=42)
    downsampled = pd.concat([positives, sampled_neg]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df, downsampled


def verify_and_save(df_full: pd.DataFrame, df_final: pd.DataFrame, ambig_count: int) -> None:
    """Assert all integrity checks, save final parquet, and write dataset_summary.csv."""
    assert set(df_final['charted'].unique()).issubset({0, 1}), "charted must contain only 0 and 1"
    assert df_final[df_final['charted'] == 0]['chart_year'].isnull().all(), "chart_year must be null for negatives"
    assert not any(c in df_final.columns for c in ['popularity', 'track_id', 'ranking', 'lyrics', 'album_name', 'explicit']), "Leakage present"
    for col in CANONICAL_FEATURES:
        assert df_final[col].isnull().sum() == 0, f"NaN in feature {col}"

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports/tables", exist_ok=True)
    df_final.to_parquet("data/processed/songs.parquet", index=False)

    summary_df = pd.DataFrame([{
        'target_definition': 'the track appeared in the Billboard Year-End Hot 100, 2000-2023',
        'ambiguous_excluded_rows': ambig_count,
        'pre_downsample_total': len(df_full),
        'pre_downsample_positives': int((df_full['charted'] == 1).sum()),
        'pre_downsample_negatives': int((df_full['charted'] == 0).sum()),
        'post_downsample_total': len(df_final),
        'post_downsample_positives': int((df_final['charted'] == 1).sum()),
        'post_downsample_negatives': int((df_final['charted'] == 0).sum()),
        'class_ratio_positive': float((df_final['charted'] == 1).mean()),
        'class_ratio_negative': float((df_final['charted'] == 0).mean()),
    }])
    summary_df.to_csv("reports/tables/dataset_summary.csv", index=False)


def main() -> None:
    """Execute complete dataset build pipeline."""
    print("=== Hit Song Predictor: Complete Dataset Pipeline (Part B) ===")
    df_ds, df_bb = load_raw("data/raw")
    df_harmonised = harmonise(df_ds)
    df_clean, stats = match_labels(df_harmonised, df_bb)
    df_dropped = drop_leakage(df_clean)

    df_dedup, exact_rem, genre_rem = deduplicate(df_dropped)
    print(f"\nDeduplication: {exact_rem} exact duplicates removed, {genre_rem} multi-genre rows removed.")

    df_valid = handle_missing(df_dedup)
    df_outliers, p01, p99, affected = handle_outliers(df_valid)
    print(f"Duration winsorisation: 1st percentile={p01:.2f}s, 99th percentile={p99:.2f}s, affected={affected} rows.")

    df_genre = collapse_genre(df_outliers)
    canonical_schema = CANONICAL_FEATURES + ['charted', 'primary_artist', 'chart_year']
    df_canonical = df_genre[canonical_schema].copy()

    df_full, df_final = downsample_negatives(df_canonical, ratio=10)
    verify_and_save(df_full, df_final, stats['ambiguous_excluded'])

    print("\n--- Final Pipeline Summary ---")
    print(f"Ambiguous rows excluded: {stats['ambiguous_excluded']}")
    print(f"Pre-downsample shape: {df_full.shape} | Positives: {(df_full['charted'] == 1).sum()}, Negatives: {(df_full['charted'] == 0).sum()}")
    print(f"Final shape: {df_final.shape}")
    print(f"Class counts:\n{df_final['charted'].value_counts().to_string()}")
    print(f"Class ratio: {df_final['charted'].value_counts(normalize=True).to_dict()}")
    print(f"\nData types:\n{df_final.dtypes.to_string()}")


if __name__ == "__main__":
    main()

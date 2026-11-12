"""Dataset access layer.

Loads ``dataset.json`` once with pandas, normalises the messy raw values, and
exposes everything the UI needs: cascading dropdown options, region options,
plate lookups and comparable-listing statistics.

Nothing in here talks to Flask, so it is easy to unit-test.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Cascade order of the Year / Make / Model form. Each level is filtered by every
# level that comes before it. These names are also the column names in the
# cleaned DataFrame and the query-string keys of /api/v1/options.
LEVELS = [
    'year',
    'make',
    'model',
    'variant',
    'body_type',
    'engine_displacement',
    'transmission',
    'fuel_type',
]

# --- Normalisation tables (edit these if your notebook cleaned things differently) ---

VALID_BODY_TYPES = {
    'Commercial', 'Coupe', 'Crossover', 'Hatchback', 'Limousine',
    'MPV', 'Pickup Truck', 'SUV', 'Sedan', 'Van',
}

TRANSMISSION_MAP = {
    'automatic': 'AT', 'a/t': 'AT', 'at': 'AT', 'automatic (ags)': 'AT',
    'manual': 'MT', 'm/t': 'MT', 'mt': 'MT',
    'cvt': 'CVT', 'ivt': 'CVT',
}

REGION_LABELS = {
    'ncr': 'NCR (Metro Manila)',
    'central_luzon': 'Central Luzon',
    'visayas': 'Visayas',
    'mindanao': 'Mindanao',
}

# Raw "Location" value (lower-case) -> region key.
# NOTE: 'MEGA LAND' (EastWest) is intentionally unmapped because its city is
# ambiguous. Add it here once you know which region it belongs to.
LOCATION_TO_REGION = {
    'moa - pasay': 'ncr', 'howmart, qc': 'ncr', 'axis tower': 'ncr',
    'muntinlupa': 'ncr', 'entrata': 'ncr',
    'bustos': 'central_luzon',
    'cebu nicopolis': 'visayas', 'cebu oriental': 'visayas', 'cebu': 'visayas',
    'cebu city': 'visayas', 'buhang, iloilo': 'visayas', 'bacolod': 'visayas',
    'roxas city, capiz': 'visayas', 'talisay': 'visayas',
    'mandaue mileage': 'visayas',
    'davao diamond': 'mindanao', 'cagayan de oro': 'mindanao',
    'panabo city': 'mindanao', 'davao city': 'mindanao',
}


def _canonical_case(series: pd.Series) -> pd.Series:
    """Merge spelling variants that differ only by case ('CHANGAN' / 'Changan').

    The most common spelling in the dataset wins. Blank values become NaN.
    """
    s = series.fillna('').astype(str).str.strip()
    key = s.str.lower()
    canon = s.groupby(key).agg(lambda x: x.value_counts().index[0])
    out = key.map(canon)
    return out.where(out != '', np.nan)


def _parse_engine(value) -> float:
    """'1.5L' -> 1.5, '1.5L Turbo' -> 1.5, 'EV' / 'N/A' -> NaN."""
    match = re.search(r'\d+(?:\.\d+)?', str(value))
    return float(match.group()) if match else np.nan


def plate_key(value) -> str:
    """Normalise a plate for matching: 'NNY 1270' / 'nny-1270' -> 'NNY1270'."""
    return re.sub(r'[^A-Z0-9]', '', str(value).upper())


def _clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df['bank'] = _canonical_case(raw['Source'])
    df['year'] = pd.to_numeric(raw['Year'], errors='coerce')
    df['make'] = _canonical_case(raw['Make / Brand'])
    df['model'] = _canonical_case(raw['Model'])
    df['variant'] = _canonical_case(raw['Variant / Trim'])

    body = raw['Body Type'].fillna('').astype(str).str.strip()
    df['body_type'] = body.where(body.isin(VALID_BODY_TYPES), np.nan)

    df['engine_displacement'] = raw['Engine / Displacement'].map(_parse_engine)

    trans = raw['Transmission'].fillna('').astype(str).str.strip().str.lower()
    df['transmission'] = trans.map(TRANSMISSION_MAP)

    # 'Fuel Type' is clean; the duplicate 'FUEL TYPE' column contains junk
    # (e.g. spreadsheet "=AI(...)" formulas) and is deliberately ignored.
    fuel = raw['Fuel Type'].fillna('').astype(str).str.strip()
    df['fuel_type'] = fuel.where(fuel != '', np.nan)

    df['mileage'] = pd.to_numeric(raw['Mileage (km)'], errors='coerce')
    df['price'] = pd.to_numeric(raw['Listing / Bid Price ()'], errors='coerce')
    df['plate'] = raw['Plate / CS Number'].fillna('').astype(str).str.strip()
    df['color'] = raw['Color'].fillna('').astype(str).str.strip()

    location = raw['Location'].fillna('').astype(str).str.strip().str.lower()
    df['location'] = raw['Location'].fillna('').astype(str).str.strip()
    df['region'] = location.map(LOCATION_TO_REGION)
    return df


class VehicleData:
    def __init__(self, dataset_path):
        path = Path(dataset_path)
        with path.open(encoding='utf-8') as fh:
            raw = pd.DataFrame(json.load(fh))
        self.df = _clean_dataframe(raw)

        # plate / CS number -> row index. Some cells hold two numbers
        # ("HA557A/NKK9634"), so every part is indexed.
        self._plate_index = {}
        for idx, plate in self.df['plate'].items():
            for part in re.split(r'[/,]', plate):
                key = plate_key(part)
                if len(key) >= 4 and key not in ('NA',):
                    self._plate_index.setdefault(key, idx)

    # ------------------------------------------------------------------ options

    def regions(self):
        """Regions that actually have listings, in a fixed north-to-south order."""
        counts = self.df['region'].value_counts()
        return [
            {'value': key, 'label': label, 'count': int(counts[key])}
            for key, label in REGION_LABELS.items()
            if key in counts.index
        ]

    def options(self, level, filters):
        """Distinct values for ``level`` given the already-chosen earlier levels."""
        if level not in LEVELS:
            raise ValueError(f'Unknown level: {level}')
        mask = pd.Series(True, index=self.df.index)
        for earlier in LEVELS[:LEVELS.index(level)]:
            raw_value = filters.get(earlier)
            if raw_value not in (None, ''):
                mask &= self.df[earlier] == self._coerce(earlier, raw_value)
        return self._sorted_options(level, self.df.loc[mask, level])

    @staticmethod
    def _coerce(level, value):
        if level == 'year':
            return int(value)
        if level == 'engine_displacement':
            return float(value)
        return str(value).strip()

    @staticmethod
    def _sorted_options(level, series):
        values = series.dropna().unique().tolist()
        if level == 'year':
            return sorted((int(v) for v in values), reverse=True)
        if level == 'engine_displacement':
            return sorted(float(v) for v in values)
        return sorted((str(v) for v in values), key=str.lower)

    # ------------------------------------------------------------------ lookups

    def find_by_plate(self, plate):
        key = plate_key(plate)
        idx = self._plate_index.get(key)
        return None if idx is None else self._to_vehicle(idx)

    def featured(self, limit=12, per_bank=2):
        """A few complete, recent listings per bank for the 'Browse Repos' drawer."""
        usable = self.df[
            self.df['plate'].map(lambda p: plate_key(p) in self._plate_index)
            & self.df['mileage'].notna()
            & (self.df['price'] > 0)
            & self.df['make'].notna()
            & self.df['model'].notna()
        ]
        picked = (
            usable.sort_values(['year', 'price'], ascending=[False, False])
            .groupby('bank', sort=True)
            .head(per_bank)
            .sort_values(['bank', 'year'], ascending=[True, False])
            .head(limit)
        )
        return [self._to_vehicle(idx) for idx in picked.index]

    def _to_vehicle(self, idx):
        row = self.df.loc[idx]

        def clean(value):
            return None if pd.isna(value) else value

        region = clean(row['region'])
        return {
            '_idx': int(idx) if isinstance(idx, (int, np.integer)) else idx,
            'bank': clean(row['bank']),
            'year': None if pd.isna(row['year']) else int(row['year']),
            'make': clean(row['make']),
            'model': clean(row['model']),
            'variant': clean(row['variant']) or '',
            'body_type': clean(row['body_type']),
            'engine_displacement': (
                None if pd.isna(row['engine_displacement'])
                else float(row['engine_displacement'])
            ),
            'transmission': clean(row['transmission']),
            'fuel_type': clean(row['fuel_type']),
            'mileage': None if pd.isna(row['mileage']) else float(row['mileage']),
            'price': None if pd.isna(row['price']) else float(row['price']),
            'plate': row['plate'],
            'region': region,
            'region_label': REGION_LABELS.get(region),
        }

    # -------------------------------------------------------------- comparables

    def comparables(self, make, model, year, region=None, exclude_idx=None):
        """Listing-price stats for similar units (same make + model).

        Uses +/-2 model years when at least 3 such listings exist, otherwise
        falls back to any year of that make + model.
        """
        pool = self.df[
            (self.df['make'] == make)
            & (self.df['model'] == model)
            & (self.df['price'] > 0)
        ]
        if exclude_idx is not None:
            pool = pool.drop(index=exclude_idx, errors='ignore')

        near = pool[(pool['year'] - year).abs() <= 2]
        if len(near) >= 3:
            pool, scope = near, f'same model, {year - 2}–{year + 2}'
        else:
            scope = 'same model, any year'

        result = {
            'count': int(len(pool)),
            'scope': scope,
            'median': float(pool['price'].median()) if len(pool) else None,
            'min': float(pool['price'].min()) if len(pool) else None,
            'max': float(pool['price'].max()) if len(pool) else None,
            'region': None,
        }

        if region and region in REGION_LABELS:
            in_region = pool[pool['region'] == region]
            result['region'] = {
                'key': region,
                'label': REGION_LABELS[region],
                'count': int(len(in_region)),
                'median': (
                    float(in_region['price'].median()) if len(in_region) else None
                ),
            }
        return result
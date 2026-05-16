# ndbc_io.py
#
# Utilities for retrieving data from the National Data Buoy Center (NDBC):
# Nota bene: NDBC uses lat/lon for active stations and lat/lng (or lon) for historical data.
#
#  list_realtime_stations()
#  list_historical_stations()
#  read_realtime_stdmet(station_id: str) -> pd.DataFrame: 
#  read_historical_stdmet(station_id: str, year: int) 
#  read_historical_radiation(station_id: str, year: int)

from __future__ import annotations
import io, gzip
from io import StringIO
import requests
import pandas as pd
from lxml import etree

__all__ = [
    "list_realtime_stations",
    "list_historical_stations",
    "read_realtime_stdmet",
    "read_historical_stdmet",
    "read_historical_radiation",
    "list_active_stations",  # alias
]

# Fetch a DataFrame with active NDBC stations

def list_realtime_stations() -> pd.DataFrame:
    df = pd.read_xml(path_or_buffer="https://www.ndbc.noaa.gov/activestations.xml")
    for c in ("lat", "lon"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def list_active_stations() -> pd.DataFrame:
    """Alias of list_realtime_stations(), kept for discoverability."""
    return list_realtime_stations()

# Fetch a DataFrame with historic NDBC stations, https://www.ndbc.noaa.gov/metadata/stationmetadata.xml 
# is a daily generated XMLfile that contains the historical metadata back to 2000 for all stations.


def list_historical_stations(meta_url: str = "https://www.ndbc.noaa.gov/metadata/stationmetadata.xml"
                          ) -> pd.DataFrame:
    """
    Build a tidy DataFrame of historic station deployments with:
    id, name, lat, lon, stat, type, start, stop, met

    Notes:
    - lat/lng often live on <history> (deployment) and may differ from the top-level <station>.
    - Some attributes may be missing for certain stations/eras -> left as NaN.
    """
    r = requests.get(meta_url, timeout=60)
    r.raise_for_status()
    root = etree.fromstring(r.content)

    rows = []
    for st in root.findall(".//station"):
        sid   = st.get("id")
        name  = st.get("name")
        # Some feeds use lon vs lng at the station level; history usually has lat/lng
        st_lat = st.get("lat")
        st_lon = st.get("lng") or st.get("lon")
        st_type = st.get("type")     # sometimes present at station level
        st_stat = st.get("stat")     # may exist on some feeds as a status flag
        st_met  = st.get("met")      # rarely present at station level
        st_own  = st.get("owner")   

        # Iterate deployment histories
        for h in st.findall("./history"):
            # Prefer deployment-specific attributes when available
            lat  = h.get("lat") or st_lat
            lon  = h.get("lng") or h.get("lon") or st_lon
            stat = h.get("stat") or st_stat
            typ  = h.get("type") or st_type
            met  = h.get("met") or st_met
            own  = h.get("owner") or st_own

            rows.append({
                "id": sid,
                "name": name,
                "lat": pd.to_numeric(lat, errors="coerce"),
                "lon": pd.to_numeric(lon, errors="coerce"),
                "stat": stat,
                "type": typ,
                "start": h.get("start"),
                "stop":  h.get("stop"),
                "met":   met,
                "owner":   own
            })

    df = pd.DataFrame(rows)
    cols = ["id", "name", "lat", "lon", "stat", "type", "start", "stop", "met","owner"]
    cols = [c for c in cols if c in df.columns]
    
    for c in ("start", "stop"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    
    return df[cols]


# Fetch a DataFreme of measurments from an "active" NDBC buoy.

def read_realtime_stdmet(station_id: str) -> pd.DataFrame:
    """Realtime stdmet (txt) → tidy DataFrame indexed by UTC datetime."""
    base = "https://www.ndbc.noaa.gov/data/realtime2"
    url = f"{base}/{station_id.upper()}.txt"
    r = requests.get(url, timeout=30); r.raise_for_status()
    lines = r.text.splitlines()
    if len(lines) < 3:
        raise ValueError("Unexpected realtime2 format: too few lines")
    names = lines[0].split()
    data = "\n".join(lines[2:])
    df = pd.read_csv(StringIO(data), sep=r"\s+", names=names, engine="python",
                     na_values=[99, 999, 9999, 99.0, 999.0, 9999.0, "MM"])
    parts = dict(
        year=df.get("#YY", df.get("YY")),  # fallback to 'YY' if '#YY' missing
        month=df.get("MM"), 
        day=df.get("DD"), 
        hour=df.get("hh"))
    if "mm" in df.columns:
        parts["minute"] = df["mm"]
    dt = pd.to_datetime(parts, errors="coerce", utc=True)
    df.insert(0, "datetime", dt)
    # Coerce non-date columns to numeric for consistency
    date_cols = {"#YY", "YY", "MM", "DD", "hh", "mm"}
    for c in df.columns:
        if c not in date_cols and c != "datetime":
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.set_index("datetime").sort_index()


# Fetch a DataFrame from a historic NDBC buoy (going back to 2000)

def read_historical_stdmet(station_id: str, year: int) -> pd.DataFrame:
    """Historical stdmet (gz) → tidy DataFrame indexed by UTC datetime."""
    base = "https://www.ndbc.noaa.gov/data/historical/stdmet"
    url = f"{base}/{station_id.lower()}h{year}.txt.gz"
    r = requests.get(url, timeout=60); r.raise_for_status()
    with gzip.open(io.BytesIO(r.content), "rt", encoding="utf-8", errors="replace") as f:
        lines = [ln.rstrip("\n") for ln in f]
    if len(lines) < 3:
        raise ValueError("Unexpected stdmet format: too few lines")
    names = lines[0].split()      # keep '#YY'
    data_lines = [ln for ln in lines[2:] if ln and not ln.startswith("#")]
    df = pd.read_csv(StringIO("\n".join(data_lines)),
                     sep=r"\s+", names=names, engine="python",
                     na_values=["MM", 99, 999, 9999, 99.0, 999.0, 9999.0])

    # Fallback for year column name
    ycol = "#YY" if "#YY" in df.columns else ("YY" if "YY" in df.columns else None)
    if ycol is None:
        raise ValueError("No year column (#YY or YY) found in stdmet file.")

    date_cols = {ycol, "MM", "DD", "hh", "mm"}
    for c in df.columns:
        if c not in date_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    parts = dict(year=df[ycol], month=df["MM"], day=df["DD"], hour=df["hh"])
    if "mm" in df.columns:
        parts["minute"] = df["mm"]    
    df.insert(0, "datetime", pd.to_datetime(parts, errors="coerce", utc=True))
    return df.set_index("datetime").sort_index()

def read_historical_radiation(station_id: str, year: int) -> pd.DataFrame:
    """Historical radiation (srad gz) → tidy DataFrame indexed by UTC datetime."""
    base = "https://www.ndbc.noaa.gov/data/historical/srad"
    url = f"{base}/{station_id.lower()}r{year}.txt.gz"
    r = requests.get(url, timeout=60)
    r.raise_for_status()

    df = pd.read_csv(
        io.BytesIO(r.content),
        compression="gzip",
        sep=r"\s+",
        header=0,
        skiprows=[1],           # skip units line
        engine="python",
        na_values=[99, 999, 9999, 99.0, 999.0, 9999.0, "MM"],
    )
    
    for col in ("SRAD1", "SWRAD", "LWRAD"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    ycol = "#YY" if "#YY" in df.columns else "YY"
    parts = dict(year=df[ycol], month=df["MM"], day=df["DD"], hour=df["hh"])
    if "mm" in df.columns: parts["minute"] = df["mm"]
    df["datetime"] = pd.to_datetime(parts, errors="coerce", utc=True)
    return df.set_index("datetime").sort_index()

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Tuple

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="🚲 Vélib' Paris — Dashboard Temps Réel",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        html, body {background: radial-gradient(circle at top left, #EEF3FF 0%, #FBFCFF 40%, #FFFFFF 100%);} 
        .main {zoom: .90;}
        .kpi-card {background: linear-gradient(135deg, #6e8efb, #a777e3);color: #ffffff;padding: 1.2rem 1rem;border-radius: 12px;text-align: center;font-family: 'Segoe UI', Arial, sans-serif;transition: transform .25s ease, box-shadow .25s ease;}
        .kpi-card:hover {transform: scale(1.04); box-shadow: 0 0 18px rgba(0,0,255,.35);}        
        .kpi-value {font-size: 2rem; font-weight: 700; margin: .2rem 0;}
        .kpi-title {font-size: .85rem; letter-spacing: .05em; text-transform: uppercase;}
        .stPlotlyChart, .stAltairChart {border-radius: 10px; transition: box-shadow .3s;}
        .stPlotlyChart:hover, .stAltairChart:hover {box-shadow: 0 0 20px rgba(0,0,255,.35);}        
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=70_000, key="refresh")
except ModuleNotFoundError:
    pass

CSV_PATH = Path(__file__).with_name("velib_paris_snapshot.csv")


def find_lat_lon(cols: List[str]) -> Tuple[str | None, str | None]:
    if {"coordonnees_geo.lat", "coordonnees_geo.lon"}.issubset(cols):
        return "coordonnees_geo.lat", "coordonnees_geo.lon"
    geo_idx = [c for c in cols if re.fullmatch(r"coordonnees_geo[._][01]", c)]
    if len(geo_idx) == 2:
        return sorted(geo_idx)[0], sorted(geo_idx)[1]
    lat_c = [c for c in cols if re.search(r"(?:^|[_.])lat$", c, re.I)]
    lon_c = [c for c in cols if re.search(r"(?:^|[_.])lon|lng$", c, re.I)]
    if lat_c and lon_c:
        return lat_c[0], lon_c[0]
    if "coordonnees_geo" in cols:
        return "coordonnees_geo", None
    return None, None


@st.cache_data(ttl=60)
def load_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        st.error("CSV introuvable — lancez le collecteur.")
        st.stop()
    df = pd.read_csv(path)
    for d in ("capture_time", "duedate", "record_timestamp"):
        if d in df.columns:
            df[d] = pd.to_datetime(df[d], errors="coerce")
    lat_col, lon_col = find_lat_lon(df.columns.tolist())
    if lat_col:
        if lon_col is None:
            coords = df[lat_col].dropna().apply(ast.literal_eval)
            df.loc[coords.index, "lat"] = coords.apply(lambda p: p[0])
            df.loc[coords.index, "lon"] = coords.apply(lambda p: p[1])
        else:
            df["lat"], df["lon"] = df[lat_col], df[lon_col]
    df["fill_rate"] = df["numbikesavailable"] / df["capacity"].replace({0: pd.NA})
    return df


df = load_df(CSV_PATH)
if df.empty:
    st.warning("CSV vide — attendre la première capture…")
    st.stop()

df_sorted = df.sort_values(["stationcode", "capture_time"])
df_sorted["delta"] = df_sorted.groupby("stationcode")["numbikesavailable"].diff().abs().fillna(0)

df_sorted["hour"] = df_sorted["capture_time"].dt.hour
rotation_hourly = df_sorted.groupby(["stationcode", "hour"])["delta"].sum().reset_index()
rotation_hourly_mean = rotation_hourly.groupby("stationcode")["delta"].mean()
availability = (
    df_sorted.groupby("stationcode")["numbikesavailable"].apply(lambda s: (s > 0).mean())
)

station_capacity = df_sorted.groupby("stationcode")["capacity"].first()
fill_rate_mean = df_sorted.groupby("stationcode")["fill_rate"].mean()

station_features = (
    pd.DataFrame({
        "capacity": station_capacity,
        "fill_rate": fill_rate_mean,
        "rotation": rotation_hourly_mean,
        "availability": availability,
    })
    .dropna()
)
scaler = StandardScaler()
feat_scaled = scaler.fit_transform(station_features)
km = MiniBatchKMeans(n_clusters=5, random_state=42)
labels = km.fit_predict(feat_scaled)
pca = PCA(n_components=3, random_state=42)
pca_coords = pca.fit_transform(feat_scaled)
station_features["cluster"] = labels
station_features[["pca1", "pca2", "pca3"]] = pca_coords

hour_bins = list(range(10, 18))
rot_table = rotation_hourly[rotation_hourly["hour"].isin(hour_bins)]
rot_pivot = rot_table.pivot(index="stationcode", columns="hour", values="delta").fillna(0)
rot_pivot = rot_pivot.reindex(columns=hour_bins).fillna(0)

table_df = (
    pd.DataFrame({
        "station": rot_pivot.index,
        "profile": rot_pivot.values.tolist(),
        "availability": availability.loc[rot_pivot.index] * 100,
    })
    .reset_index(drop=True)
)
station_names = df.set_index("stationcode")["name"].to_dict()

table_df["station"] = table_df["station"].apply(lambda c: f"[{station_names.get(c, c)}](/Station?station_code={c})")

latest_ts = df["capture_time"].max()
latest_df = df[df["capture_time"] == latest_ts]

kpi_cols = st.columns([1, 1, 1, 1])

global_hourly = df_sorted.groupby(df_sorted["capture_time"].dt.floor("H"))["delta"].sum()
peak_hour = global_hourly.idxmax().strftime("%Hh")

global_fill = df_sorted["fill_rate"].mean()

global_rotation = rotation_hourly["delta"].mean()

global_unavail = (
    df_sorted[(df_sorted["numbikesavailable"] == 0) | (df_sorted["numdocksavailable"] == 0)]
    .shape[0]
    / df_sorted.shape[0]
)

kpis = [
    ("Heure de pointe", peak_hour),
    ("Occupation moyenne", f"{global_fill:.0%}"),
    ("Rotation horaire", f"{global_rotation:.1f}"),
    ("Indisponibilité", f"{global_unavail:.0%}"),
]
for col, (title, value) in zip(kpi_cols, kpis):
    col.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-title'>{title}</div>
            <div class='kpi-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.divider()

if {"lat", "lon"}.issubset(latest_df.columns):
    map_col, table_col = st.columns([2, 3])
    with map_col:
        mp = latest_df.copy()
        mp["radius"] = mp["capacity"].fillna(15).astype(float) * 2
        mp["fill_rate"] = mp["fill_rate"].fillna(0).clip(0, 1)
        mp["color_r"] = (mp["fill_rate"] * 255).round().astype(int)
        mp["color_g"] = ((1 - mp["fill_rate"]) * 255).round().astype(int)
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=mp,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="[color_r, color_g, 80, 170]",
            pickable=True,
        )
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=pdk.ViewState(latitude=mp["lat"].mean(), longitude=mp["lon"].mean(), zoom=12),
                tooltip={"text": "{name}\nRemplissage: {fill_rate:.0%}"},
                map_style="mapbox://styles/mapbox/light-v9",
            )
        )
    with table_col:
        st.dataframe(
            table_df,
            column_config={
                "profile": st.column_config.LineChartColumn(label="Profil horaire", width="large"),
                "availability": st.column_config.ProgressColumn(label="Disponibilité moyenne", min_value=0, max_value=100, width="medium", format="%.0f%%"),
            },
            hide_index=True,
            use_container_width=True,
        )

st.divider()

cluster_col, cluster_table_col = st.columns([2, 1])

with cluster_col:
    tabs = st.tabs(["3D", "2D"])
    cluster_options = ["Tous"] + [f"Cluster {i}" for i in sorted(station_features["cluster"].unique())]
    sel = st.selectbox("Sélection cluster", cluster_options, key="cluster_select")
    if sel != "Tous":
        sel_idx = int(sel.split()[-1])
        plot_df = station_features[station_features["cluster"] == sel_idx]
    else:
        plot_df = station_features
    with tabs[0]:
        fig3d = go.Figure(data=[
            go.Scatter3d(
                x=plot_df["pca1"],
                y=plot_df["pca2"],
                z=plot_df["pca3"],
                mode="markers",
                marker=dict(size=4, color=plot_df["cluster"], colorscale="Viridis"),
                text=plot_df.index,
            )
        ])
        fig3d.update_layout(height=500, margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig3d, use_container_width=True)
    with tabs[1]:
        alt_df = plot_df.reset_index()
        chart2d = (
            alt.Chart(alt_df)
            .mark_circle(size=60)
            .encode(
                x="pca1:Q",
                y="pca2:Q",
                color="cluster:N",
                tooltip=["stationcode", "cluster"],
            )
            .properties(height=500)
        )
        st.altair_chart(chart2d, use_container_width=True)

with cluster_table_col:
    clust_stats = station_features.groupby("cluster").agg(
        stations=("capacity", "count"),
        capacity=("capacity", "mean"),
        fill_rate=("fill_rate", "mean"),
        rotation=("rotation", "mean"),
    ).reset_index()
    st.dataframe(clust_stats, hide_index=True, use_container_width=True)

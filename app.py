import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from streamlit_extras.add_vertical_space import add_vertical_space
import time
from data_utils import load_monthly_crime_data
from algorithms import run_dbscan, run_hdbscan, run_kmeans, run_spatiotemporal_dbscan, run_prophet, run_anomaly_detection
import plotly.express as px
import folium
import geopandas as gpd
import pandas as pd

data_dir = "UKPOLICEDATA/2025-01"

gdf, filter_columns = load_monthly_crime_data(data_dir)

# Custom CSS and logo placeholder
st.markdown("""
    <style>
    body {background-color: #f5f7fa;}
    .main-title {color: #0072B5; font-size: 40px; font-weight: bold; text-align: center; margin-bottom: 10px;}
    .logo-placeholder {width: 120px; height: 120px; background: #e0e0e0; border-radius: 50%; display: block; margin: 0 auto 20px auto; text-align: center; line-height: 120px; font-size: 32px; color: #0072B5; font-weight: bold;}
    .stButton>button {color: white; background: #0072B5; border-radius: 6px;}
    .stSidebar {background-color: #eaf1fb;}
    .footer {text-align: center; color: #888; font-size: 14px; margin-top: 40px;}
    /* Sidebar Enhancements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eaf1fb 0%, #c7e0fa 100%);
        padding: 24px 16px 24px 16px;
        font-size: 18px;
        color: #1a2a3a;
    }
    .sidebar-header {
        font-size: 24px;
        font-weight: bold;
        color: #0072B5;
        margin-bottom: 18px;
        margin-top: 8px;
        letter-spacing: 1px;
    }
    .sidebar-section {
        background: #f5f7fa;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        padding: 16px;
        margin-bottom: 18px;
    }
    .stMultiSelect, .stDateInput, .stSlider, .stSelectbox {
        margin-bottom: 14px !important;
    }
    .stMultiSelect>div, .stDateInput>div, .stSlider>div, .stSelectbox>div {
        font-size: 16px !important;
    }
    .stButton>button {
        background: #0072B5;
        color: #fff;
        font-size: 16px;
        border-radius: 6px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
    <div class='logo-placeholder'>LOGO</div>
    <div class='main-title'>UK Crime Hotspot Analysis</div>
""", unsafe_allow_html=True)

# Sidebar Sectioning
with st.sidebar:
    st.markdown('<div class="sidebar-header">Dynamic Filters</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    filter_widgets = {}
    for col in filter_columns:
        if col == 'crime_type':
            options = [opt for opt in gdf[col].dropna().unique().tolist() if opt != 'Unknown' and opt != '']
            filter_widgets[col] = st.multiselect("Crime Type", options, default=options)
        elif col == 'date':
            min_date, max_date = gdf[col].min(), gdf[col].max()
            if pd.isnull(min_date) or pd.isnull(max_date):
                min_date, max_date = pd.to_datetime("2025-01-01"), pd.to_datetime("2025-01-31")
            min_date, max_date = min_date if hasattr(min_date, 'date') else pd.to_datetime(min_date), max_date if hasattr(max_date, 'date') else pd.to_datetime(max_date)
            min_date, max_date = min_date.date(), max_date.date()
            filter_widgets[col] = st.date_input("Date Range", [min_date, max_date])
        elif col == 'street_name':
            options = [opt for opt in gdf[col].dropna().unique().tolist() if opt != 'Unknown' and opt != '']
            filter_widgets[col] = st.multiselect("Street Name", options)
        elif col == 'outcome_category':
            options = [opt for opt in gdf[col].dropna().unique().tolist() if opt != 'Unknown' and opt != '']
            filter_widgets[col] = st.multiselect("Outcome Category", options)
        elif col == 'region':
            options = [opt for opt in gdf[col].dropna().unique().tolist() if opt != 'Unknown' and opt != '']
            filter_widgets[col] = st.multiselect("Region", options, default=options)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">Clustering Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    eps = st.slider("DBSCAN eps", 0.001, 0.05, 0.01)
    min_samples = st.slider("min_samples / min_cluster_size", 1, 50, 10)
    n_clusters = st.slider("KMeans n_clusters", 2, 20, 5)
    hdbscan_min_samples = st.slider("HDBSCAN min_samples (optional)", 1, 50, 10)
    use_hdbscan_st = st.checkbox("Use HDBSCAN for spatiotemporal clustering", value=False)
    st.markdown('</div>', unsafe_allow_html=True)
    add_vertical_space(2)
    st.markdown('<div class="sidebar-header">Other Controls</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    crime_type_for_pred = st.selectbox("Crime type for forecasting (optional)", [None] + gdf['crime_type'].dropna().unique().tolist())
    pred_periods = st.slider("Forecast periods (days)", 7, 90, 30)
    features_for_anomaly = st.multiselect("Features for anomaly detection", ["geometry.x", "geometry.y", "crime_type", "date"], default=["geometry.x", "geometry.y"])
    anomaly_contamination = st.slider("Anomaly contamination", 0.001, 0.1, 0.01)
    st.markdown('</div>', unsafe_allow_html=True)

# Apply dynamic filters
filtered_gdf = gdf.copy()
for col, val in filter_widgets.items():
    if col == 'crime_type' or col == 'street_name' or col == 'outcome_category' or col == 'region':
        if val:
            filtered_gdf = filtered_gdf[filtered_gdf[col].isin(val)]
    elif col == 'date':
        if val and len(val) == 2:
            # Ensure both sides are datetime.date
            filtered_gdf = filtered_gdf[
                (pd.to_datetime(filtered_gdf[col]).dt.date >= val[0]) &
                (pd.to_datetime(filtered_gdf[col]).dt.date <= val[1])
            ]

# Error/Empty State Messages
if filtered_gdf.empty:
    st.warning(
        f"No data matches the selected filters:\n"
        f"Crime Types: {filter_widgets.get('crime_type', [])}\n"
        f"Date Range: {filter_widgets.get('date', [])}\n"
        f"Street Names: {filter_widgets.get('street_name', [])}\n"
        f"Outcome Categories: {filter_widgets.get('outcome_category', [])}\n"
        f"Regions: {filter_widgets.get('region', [])}\n"
        "Please adjust your selections or reset filters."
    )
    st.button("Reset Filters")
    st.stop()

# Responsive Layout
col1, col2 = st.columns([2, 1])
with col1:
    st.header("Summary Statistics")
    st.write(f"Total crimes: {len(filtered_gdf)}")
    st.write(filtered_gdf['crime_type'].value_counts())
with col2:
    st.header("Download Data")
    st.download_button("Download filtered data as CSV", filtered_gdf.to_csv(index=False), "filtered_crimes.csv")

# Loading Spinner for clustering
with st.spinner("Clustering in progress..."):
    st.header("Clustering Algorithms")
    cluster_method = st.selectbox("Choose clustering algorithm", ["DBSCAN", "HDBSCAN", "KMeans", "Spatiotemporal DBSCAN"])
    if cluster_method == "DBSCAN":
        clustered_gdf, cluster_summary = run_dbscan(filtered_gdf, eps, min_samples)
        st.subheader("DBSCAN Cluster Summary")
        st.dataframe(cluster_summary)
    elif cluster_method == "HDBSCAN":
        clustered_gdf, cluster_summary = run_hdbscan(filtered_gdf, min_cluster_size=min_samples, min_samples=hdbscan_min_samples)
        st.subheader("HDBSCAN Cluster Summary")
        st.dataframe(cluster_summary)
        st.write("HDBSCAN Outlier Scores and Probabilities:")
        st.dataframe(clustered_gdf[["probability", "outlier_score"]])
    elif cluster_method == "KMeans":
        clustered_gdf, inertia, cluster_summary = run_kmeans(filtered_gdf, n_clusters=n_clusters)
        st.subheader("KMeans Cluster Summary")
        st.dataframe(cluster_summary)
        st.write(f"KMeans Inertia (lower is better): {inertia}")
    else:
        clustered_gdf, cluster_summary = run_spatiotemporal_dbscan(filtered_gdf, eps=eps, min_samples=min_samples, use_hdbscan=use_hdbscan_st)
        st.subheader("Spatiotemporal Cluster Summary")
        st.dataframe(cluster_summary)

# Improved Download Buttons
st.download_button("Download clustered data as CSV", clustered_gdf.to_csv(index=False), "clustered_crimes.csv")

# Interactive Map with Popups/Tooltips
st.header("Crime Locations Map")
fig = px.scatter_mapbox(filtered_gdf, lat=filtered_gdf.geometry.y, lon=filtered_gdf.geometry.x,
                        color="crime_type", hover_name="crime_type", hover_data=["street_name", "outcome_category"], zoom=7, height=500)
fig.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig, use_container_width=True)

st.header("Cluster Map")
fig2 = px.scatter_mapbox(clustered_gdf, lat=clustered_gdf.geometry.y, lon=clustered_gdf.geometry.x,
                        color="cluster", hover_name="crime_type", hover_data=["street_name", "outcome_category"], zoom=7, height=500)
fig2.update_layout(mapbox_style="open-street-map")
st.plotly_chart(fig2, use_container_width=True)

# Chart Interactivity
st.header("Charts & Trends")
st.plotly_chart(px.bar(filtered_gdf, x="crime_type", title="Crime Type Distribution", color="crime_type"), use_container_width=True)

# Predictive Modeling with Spinner
with st.spinner("Forecasting in progress..."):
    st.header("Predictive Modeling")
    predictions = run_prophet(filtered_gdf, crime_type=crime_type_for_pred, periods=pred_periods)
    st.write(predictions)

# Anomaly Detection with Spinner
with st.spinner("Detecting anomalies..."):
    st.header("Anomaly Detection")
    anomalies = run_anomaly_detection(filtered_gdf, features=features_for_anomaly, contamination=anomaly_contamination)
    st.write(anomalies)

# Custom Footer
st.markdown("""
    <div class='footer'>
        &copy; 2025 UK Crime Hotspot Analysis | Powered by Streamlit & Plotly
    </div>
""", unsafe_allow_html=True)

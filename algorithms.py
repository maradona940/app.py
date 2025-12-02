import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import hdbscan
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import pandas as pd

# DBSCAN clustering
def run_dbscan(gdf, eps=0.01, min_samples=10):
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(coords)
    gdf['cluster'] = db.labels_
    summary = gdf[gdf['cluster'] != -1].groupby('cluster').agg(
        crime_count=('crime_type', 'count'),
        centroid_lon=('longitude', 'mean'),
        centroid_lat=('latitude', 'mean'),
        top_crime_type=('crime_type', lambda x: x.value_counts().idxmax())
    ).reset_index()
    return gdf, summary

# HDBSCAN clustering
def run_hdbscan(gdf, min_cluster_size=10, min_samples=1):
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples)
    gdf['cluster'] = clusterer.fit_predict(coords)
    gdf['probability'] = clusterer.probabilities_
    gdf['outlier_score'] = clusterer.outlier_scores_
    summary = gdf[gdf['cluster'] != -1].groupby('cluster').agg(
        crime_count=('crime_type', 'count'),
        centroid_lon=('longitude', 'mean'),
        centroid_lat=('latitude', 'mean'),
        top_crime_type=('crime_type', lambda x: x.value_counts().idxmax())
    ).reset_index()
    return gdf, summary

# KMeans clustering
def run_kmeans(gdf, n_clusters=5):
    coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(coords)
    gdf['cluster'] = kmeans.labels_
    inertia = kmeans.inertia_
    summary = gdf.groupby('cluster').agg(
        crime_count=('crime_type', 'count'),
        centroid_lon=('longitude', 'mean'),
        centroid_lat=('latitude', 'mean'),
        top_crime_type=('crime_type', lambda x: x.value_counts().idxmax())
    ).reset_index()
    return gdf, inertia, summary

# Spatiotemporal clustering
def run_spatiotemporal_dbscan(gdf, eps=0.01, min_samples=10, use_hdbscan=False, min_cluster_size=10):
    if 'date' in gdf.columns:
        time_numeric = (gdf['date'] - gdf['date'].min()).dt.days
        coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y, time_numeric)))
    elif 'month' in gdf.columns:
        time_numeric = pd.to_datetime(gdf['month']).astype(int)
        coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y, time_numeric)))
    else:
        coords = np.array(list(zip(gdf.geometry.x, gdf.geometry.y)))
    if use_hdbscan:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        gdf['cluster'] = clusterer.fit_predict(coords)
    else:
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(coords)
        gdf['cluster'] = db.labels_
    summary = gdf[gdf['cluster'] != -1].groupby('cluster').agg(
        crime_count=('crime_type', 'count'),
        centroid_lon=('longitude', 'mean'),
        centroid_lat=('latitude', 'mean'),
        top_crime_type=('crime_type', lambda x: x.value_counts().idxmax())
    ).reset_index()
    return gdf, summary

# Prophet forecasting
def run_prophet(gdf, crime_type=None, periods=30):
    if 'date' in gdf.columns:
        df = gdf.copy()
        if crime_type:
            df = df[df['crime_type'] == crime_type]
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = 1
        daily = df.groupby('ds').y.sum().reset_index()
        m = Prophet()
        m.fit(daily)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    else:
        return "No date column for forecasting."

# Anomaly detection
def run_anomaly_detection(gdf, features=None, contamination=0.05):
    if features is None:
        features = ['geometry.x', 'geometry.y']
    X = np.column_stack([getattr(gdf.geometry, f.split('.')[-1]) if f.startswith('geometry.') else gdf[f] for f in features if f in gdf.columns or f.startswith('geometry.')])
    clf = IsolationForest(contamination=contamination)
    preds = clf.fit_predict(X)
    scores = clf.decision_function(X)
    gdf['anomaly'] = preds
    gdf['anomaly_score'] = scores
    return gdf[gdf['anomaly'] == -1]

# UK Crime Hotspot Analysis

This project visualizes and analyzes spatial and temporal patterns of crime in the UK, focusing on hotspot detection and predictive analytics using police data.

## Features
- Load and combine monthly UK Police CSV data
- Interactive Streamlit dashboard
- Dynamic filtering by crime type, date, street, outcome, region
- Clustering algorithms: DBSCAN, HDBSCAN, KMeans, Spatiotemporal DBSCAN
- Cluster summaries and map visualizations
- Predictive modeling with Prophet
- Anomaly detection with Isolation Forest
- Download filtered and clustered data
- Custom UI with branding and responsive layout

## File Structure
- `app.py` — Main Streamlit app
- `data_utils.py` — Data loading and preprocessing
- `algorithms.py` — Clustering, forecasting, anomaly detection
- `requirements.txt` — Python dependencies
- `UKPOLICEDATA/` — Monthly police CSV data
- `README.md` — Project documentation

## Getting Started
1. Install dependencies:
   ```shell
   pip install -r requirements.txt
   ```
2. Run the app:
   ```shell
   streamlit run app.py
   ```
3. Explore the dashboard and filter, cluster, and analyze crime data interactively.

## Data
- Place monthly CSV files in the `UKPOLICEDATA/2025-01/` folder.
- Supported columns: `crime_type`, `date`, `street_name`, `outcome_category`, `region`, `longitude`, `latitude`

## Algorithms
- **DBSCAN/HDBSCAN/KMeans:** Detect spatial clusters (hotspots)
- **Spatiotemporal DBSCAN:** Cluster by location and time
- **Prophet:** Forecast crime trends
- **Isolation Forest:** Detect anomalies

## Customization
- Update `app.py` for UI changes, branding, or new features
- Add new algorithms in `algorithms.py`

## License
MIT

## Contact
For questions or contributions, open an issue or contact the project maintainer.

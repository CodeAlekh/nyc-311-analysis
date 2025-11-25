import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def plot_elbow(data, title = "Elbow Method"):
    inertia = []
    k_range = range(1, 10)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)

    # Save the Elbow Plot
    plt.figure(figsize=(8, 4))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title(title)
    plt.grid(True)

    
def plot_cluster_map(df, title='Cluster Map'):
    """
    Generates a choropleth map of clusters using spatial join between 
    incident centroids and map polygons.

    Args:
        df (pd.DataFrame): DataFrame with 'Incident Zip' and 'Cluster' columns.
        title (str): Title of the map.
    """
    
    print("--- Starting Map Generation ---")

    original_data_path = "../../cleaned_data.parquet"
    map_geojson_path = "../../geo_data.geojson"

    # 1. Prepare Cluster Data
    cluster_df = df.copy()
    cluster_df['Incident Zip'] = cluster_df['Incident Zip'].astype(str).str.strip()
    
    # 2. Calculate Centroids from Original Data
    print("Calculating zip code centroids...")
    try:
        # Read only necessary columns to save memory
        original_df = pd.read_parquet(original_data_path, columns=['Incident Zip', 'Latitude', 'Longitude'])
        zip_centroids = original_df.groupby('Incident Zip')[['Latitude', 'Longitude']].mean().reset_index()
        zip_centroids['Incident Zip'] = zip_centroids['Incident Zip'].astype(str).str.strip()
    except Exception as e:
        print(f"Error reading original data: {e}")
        return

    # 3. Merge Clusters with Centroids
    cluster_geo_points = cluster_df.merge(zip_centroids, on='Incident Zip', how='left')
    cluster_geo_points = cluster_geo_points.dropna(subset=['Latitude', 'Longitude'])

    # Convert to GeoDataFrame (Points)
    gdf_points = gpd.GeoDataFrame(
        cluster_geo_points,
        geometry=gpd.points_from_xy(cluster_geo_points.Longitude, cluster_geo_points.Latitude),
        crs="EPSG:4326"
    )

    # 4. Load Map Polygons
    print(f"Loading map from {map_geojson_path}...")
    try:
        nyc_zips_poly = gpd.read_file(map_geojson_path)
        if nyc_zips_poly.crs is None:
            nyc_zips_poly.set_crs("EPSG:4326", inplace=True)
        else:
            nyc_zips_poly = nyc_zips_poly.to_crs("EPSG:4326")
    except Exception as e:
        print(f"Error loading map file: {e}")
        return

    # 5. Spatial Join
    print("Performing spatial join...")
    joined = gpd.sjoin(nyc_zips_poly, gdf_points, how="left", predicate="contains")
    
    # Aggregating to handle multiple points per polygon (taking the first/mode)
    # Using dissolve to merge geometries back to unique map zones
    group_col = 'modzcta' if 'modzcta' in joined.columns else (
        'postalCode' if 'postalCode' in joined.columns else joined.index.name
    )
    
    if group_col is None: 
        # Fallback if index has no name, use the first column
        group_col = joined.columns[0] 

    final_map_data = joined.dissolve(by=group_col, aggfunc='first')
    final_map_data['Cluster'] = final_map_data['Cluster'].fillna(-1)

    # 6. Plotting
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Background (No Data)
    final_map_data.plot(ax=ax, color='#f0f0f0', edgecolor='white')
    
    # Clusters
    valid_data = final_map_data[final_map_data['Cluster'] != -1]
    
    if not valid_data.empty:
        valid_data.plot(
            column='Cluster',
            ax=ax,
            legend=True,
            categorical=True,
            cmap='tab10',
            edgecolor='white',
            linewidth=0.2,
            legend_kwds={'title': 'Cluster', 'loc': 'upper left'}
        )
    
    plt.title(title, fontsize=16)
    plt.axis('off')
    
    plt.show()
    print("--- Map Generation Complete ---")
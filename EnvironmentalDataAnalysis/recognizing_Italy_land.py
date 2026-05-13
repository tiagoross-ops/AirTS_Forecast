"""
Location Identifier and High-Precision Grid Filter
Determines if coordinates are in Italy and creates filtered coordinate
grids for dataset extraction using OSMnx and Shapely.
"""

from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import Point, box


def get_italy_grid_coordinates(
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
        resolution: float = 0.1
) -> list[tuple[float, float]]:
    """
    Generates a grid of coordinates and filters out points outside Italy.
    Uses OSMnx for high-resolution, precise geographic boundaries.

    Args:
        min_lat (float): The southern boundary of the grid.
        max_lat (float): The northern boundary of the grid.
        min_lon (float): The western boundary of the grid.
        max_lon (float): The eastern boundary of the grid.
        resolution (float): The step size between coordinates.

    Returns:
        list[tuple[float, float]]: A list of valid (lat, lon) tuples.
    """
    print("Fetching high-resolution geographic boundary for Italy...")
    italy_gdf = ox.geocode_to_gdf("Italy")
    italy_polygon = italy_gdf.geometry.iloc[0]

    print(f"Generating grid with {resolution}° resolution...")
    lats = np.arange(min_lat, max_lat + resolution, resolution)
    lons = np.arange(min_lon, max_lon + resolution, resolution)

    italy_coordinates = []

    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            if italy_polygon.contains(point):
                italy_coordinates.append((round(lat, 4), round(lon, 4)))

    print(f"Grid filtering complete. Extracted {len(italy_coordinates)} points strictly inside Italy.")
    return italy_coordinates


def map_italy_grid_classification(
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
        resolution: float = 0.1,
        output_filename: str = "italy_grid_classification.html"
) -> Path:
    """
    Generates a map displaying a bounding box divided into Italian (green)
    and non-Italian (red) regions, overlaid with the exact grid points
    determined by the resolution parameter.

    Args:
        min_lat (float): The southern boundary of the square.
        max_lat (float): The northern boundary of the square.
        min_lon (float): The western boundary of the square.
        max_lon (float): The eastern boundary of the square.
        resolution (float): The spatial granularity to visually plot.
        output_filename (str): The name of the resulting HTML file.

    Returns:
        Path: The absolute path to the generated map file.
    """
    print("Fetching high-resolution geographic boundary for Italy...")
    italy_gdf = ox.geocode_to_gdf("Italy")
    italy_polygon = italy_gdf.geometry.iloc[0]
    bounding_box = box(min_lon, min_lat, max_lon, max_lat)

    print("Calculating spatial intersections...")
    italy_in_box = bounding_box.intersection(italy_polygon)
    non_italy_in_box = bounding_box.difference(italy_polygon)

    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=7, tiles="CartoDB positron")

    print("Rendering geographic region layers...")

    # 1. Base Polygon Layers
    if not italy_in_box.is_empty:
        folium.GeoJson(
            gpd.GeoSeries([italy_in_box]).to_json(),
            style_function=lambda x: {'fillColor': 'green', 'color': 'darkgreen', 'weight': 2, 'fillOpacity': 0.3},
            name="Italian Territory (Polygon)"
        ).add_to(m)

    if not non_italy_in_box.is_empty:
        folium.GeoJson(
            gpd.GeoSeries([non_italy_in_box]).to_json(),
            style_function=lambda x: {'fillColor': 'red', 'color': 'darkred', 'weight': 2, 'fillOpacity': 0.1},
            name="Outside Territory (Polygon)"
        ).add_to(m)

    print(f"Rendering grid coordinates at {resolution}° granularity...")

    # 2. Granularity Grid Point Layers
    valid_points_layer = folium.FeatureGroup(name="Valid Grid Points (Inside)")
    invalid_points_layer = folium.FeatureGroup(name="Invalid Grid Points (Outside)", show=False) # Hidden by default

    lats = np.arange(min_lat, max_lat + resolution, resolution)
    lons = np.arange(min_lon, max_lon + resolution, resolution)

    for lat in lats:
        for lon in lons:
            point = Point(lon, lat)
            is_inside = italy_polygon.contains(point)

            # Plot Green dots for valid bins, Red dots for rejected bins
            folium.CircleMarker(
                location=(lat, lon),
                radius=3,
                color='darkgreen' if is_inside else 'darkred',
                fill=True,
                fill_color='lime' if is_inside else 'red',
                fill_opacity=0.8,
                tooltip=f"Lat: {lat:.4f}, Lon: {lon:.4f} <br>Status: {'Valid' if is_inside else 'Rejected'}"
            ).add_to(valid_points_layer if is_inside else invalid_points_layer)

    # Add the point layers to the map
    valid_points_layer.add_to(m)
    invalid_points_layer.add_to(m)

    folium.LayerControl().add_to(m)

    output_path = Path(output_filename)
    m.save(str(output_path))

    return output_path


if __name__ == '__main__':
    # Define bounding box (covering Italy roughly)
    grid_min_lat = 43.5
    grid_max_lat = 43.7
    grid_min_lon = 1.4
    grid_max_lon = 1.5
    target_resolution = 1 # Increased for visual clarity in testing

    # Execute the filter
    filtered_italy_grid = get_italy_grid_coordinates(
        min_lat=grid_min_lat,
        max_lat=grid_max_lat,
        min_lon=grid_min_lon,
        max_lon=grid_max_lon,
        resolution=target_resolution
    )

    print("\nSample of valid Italy coordinates:")
    for coord in filtered_italy_grid[:5]:
        print(coord)

    # Execute the visual classification map
    map_file = map_italy_grid_classification(
        min_lat=grid_min_lat,
        max_lat=grid_max_lat,
        min_lon=grid_min_lon,
        max_lon=grid_max_lon,
        resolution=target_resolution
    )

    print(f"\nClassification map successfully generated: \n{map_file.absolute()}")
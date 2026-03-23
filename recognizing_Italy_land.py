"""
Location Identifier and High-Precision Grid Filter
Determines if coordinates are in Italy and creates filtered coordinate
grids for dataset extraction using OSMnx and Shapely.
"""

import numpy as np
import osmnx as ox
from shapely.geometry import Point
import folium
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path

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

    # ox.geocode_to_gdf queries OpenStreetMap and returns a GeoDataFrame
    # This provides a highly accurate MultiPolygon of Italy's borders
    italy_gdf = ox.geocode_to_gdf("Italy")

    # Extract the Shapely geometry object from the GeoDataFrame
    italy_polygon = italy_gdf.geometry.iloc[0]

    print(f"Generating grid with {resolution}° resolution...")

    # Generate arrays of latitudes and longitudes
    # Adding 'resolution' ensures the upper bounds are included in the mathematical grid
    lats = np.arange(min_lat, max_lat + resolution, resolution)
    lons = np.arange(min_lon, max_lon + resolution, resolution)

    italy_coordinates = []

    # Evaluate every grid intersection
    for lat in lats:
        for lon in lons:
            # Shapely Point uses (Longitude, Latitude) ordering
            point = Point(lon, lat)

            # Spatial intersection check
            if italy_polygon.contains(point):
                # Round coordinates to prevent floating-point drift (e.g., 41.900000000000006)
                italy_coordinates.append((round(lat, 4), round(lon, 4)))

    print(f"Grid filtering complete. Extracted {len(italy_coordinates)} points strictly inside Italy.")
    return italy_coordinates


def map_italy_grid_classification(
        min_lat: float, max_lat: float,
        min_lon: float, max_lon: float,
        output_filename: str = "italy_grid_classification.html"
) -> Path:
    """
    Generates a map displaying a bounding box divided into Italian (green)
    and non-Italian (red) regions using spatial intersection and difference.

    Args:
        min_lat (float): The southern boundary of the square.
        max_lat (float): The northern boundary of the square.
        min_lon (float): The western boundary of the square.
        max_lon (float): The eastern boundary of the square.
        output_filename (str): The name of the resulting HTML file.

    Returns:
        Path: The absolute path to the generated map file.
    """
    print("Fetching high-resolution geographic boundary for Italy...")

    # Retrieve the official boundary of Italy
    italy_gdf = ox.geocode_to_gdf("Italy")
    italy_polygon = italy_gdf.geometry.iloc[0]

    # Create the bounding box geometry
    # Shapely coordinates are strictly (X, Y), which translates to (Longitude, Latitude)
    bounding_box = box(min_lon, min_lat, max_lon, max_lat)

    print("Calculating spatial intersections...")

    # The Green region: The overlapping area between the bounding box and Italy
    italy_in_box = bounding_box.intersection(italy_polygon)

    # The Red region: The area of the bounding box minus the Italy polygon
    non_italy_in_box = bounding_box.difference(italy_polygon)

    # Initialize the map centered exactly on the middle of the bounding box
    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

    print("Rendering map layers...")

    # Render the Italian region (Green) if it exists within the box
    if not italy_in_box.is_empty:
        folium.GeoJson(
            gpd.GeoSeries([italy_in_box]).to_json(),
            style_function=lambda x: {
                'fillColor': 'green',
                'color': 'darkgreen',
                'weight': 2,
                'fillOpacity': 0.4
            },
            name="Italian Region (Mainland & Islands)"
        ).add_to(m)

    # Render the Non-Italian region (Red) if it exists within the box
    if not non_italy_in_box.is_empty:
        folium.GeoJson(
            gpd.GeoSeries([non_italy_in_box]).to_json(),
            style_function=lambda x: {
                'fillColor': 'red',
                'color': 'darkred',
                'weight': 2,
                'fillOpacity': 0.4
            },
            name="Non-Italian Region (Sea & Foreign Land)"
        ).add_to(m)

    # Add a layer control panel to toggle the red and green regions on/off
    folium.LayerControl().add_to(m)

    output_path = Path(output_filename)
    m.save(str(output_path))

    return output_path

if __name__ == '__main__':
    # Define bounding box (covering Italy roughly)
    grid_min_lat = 42.0
    grid_max_lat = 44.5
    grid_min_lon = 10.5
    grid_max_lon = 14.5

    # Execute the filter
    filtered_italy_grid = get_italy_grid_coordinates(
        min_lat=grid_min_lat,
        max_lat=grid_max_lat,
        min_lon=grid_min_lon,
        max_lon=grid_max_lon,
        resolution=0.1
    )

    print("\nSample of valid Italy coordinates:")
    for coord in filtered_italy_grid[:5]:
        print(coord)

    # Execute the visual classification map
    map_file = map_italy_grid_classification(
        min_lat=grid_min_lat,
        max_lat=grid_max_lat,
        min_lon=grid_min_lon,
        max_lon=grid_max_lon
    )

    print(f"\nClassification map successfully generated: \n{map_file.absolute()}")
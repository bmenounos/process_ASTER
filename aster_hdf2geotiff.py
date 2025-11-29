#!/usr/bin/env python3
"""
ASTER L1A HDF4 Extractor - Explicit Block Separation & Geometric Diagnostics

This version addresses the reported issue of an extra row of zeros/NO_DATA_VALUEs 
by removing the final, extraneous blank line that was being appended to the 
Latitude and Longitude grid files.

REVISION 6 (Current): Removed the final check in `format_array_to_text` that 
was appending a blank line. This was the likely cause of the "last row of zeros" 
bug in the computed Lat/Lon files. Diagnostic counters remain for the geometric 
ray-ellipsoid failure analysis.
"""

import os
import re
import sys
from pathlib import Path
from osgeo import gdal
import numpy as np
import logging

# Set the standard No-Data Value for output text files
NO_DATA_VALUE = -9999.0 

# --- DATASETS TO BE EXTRACTED ---
IMAGE_DATA_DATASETS = [
    'VNIR_Band3N:ImageData',
    'VNIR_Band3B:ImageData'
]

# The bands we are focusing on for all geometry and image extraction
TARGET_BANDS = ['VNIR_Band3N', 'VNIR_Band3B']

# Define the geometry datasets required by aster2asp
ASP_GEOMETRY_DATASETS = [
    'LatticePoint',
    'SatellitePosition',
    'SatelliteVelocity',
    'SightVector',
    'AttitudeAngle',
    'ObservationTime'
]

def extract_granule_name(hdf_filename):
    """Extract the granule name from the HDF filename."""
    base_name = os.path.splitext(os.path.basename(hdf_filename))[0]
    return base_name

def parse_subdataset_name(subdataset_name):
    """
    Parse a GDAL subdataset name to extract band and dataset information.
    Returns: (band, dataset, full_name)
    """
    match = re.search(r':([\w-]+):([\w-]+)$', subdataset_name)
    if match:
        band = match.group(1)
        dataset = match.group(2)
        full_name = f"{band}:{dataset}"
        return band, dataset, full_name
    
    match_top = re.search(r':([\w-]+)$', subdataset_name)
    if match_top:
        dataset = match_top.group(1)
        return "Global", dataset, dataset
        
    return None, None, None

def format_array_to_text(arr, dataset_name):
    """
    Format a 2D or 3D numpy array into a list of strings, with each row/vector
    becoming a space-separated line of text. 
    
    For geometry datasets destined for ASP (SightVector, etc.), we add an explicit 
    blank line (\n) between time steps (T) blocks.
    
    For Latitude/Longitude grids ("LatLon_Grid"), block separation is disabled.
    """
    lines = []
    
    # CRITICAL FIX 1: Ensure the array is processed as high-precision float64 
    arr = np.array(arr, dtype=np.float64)
    
    # Check if the array needs time-block separation 
    needs_block_sep = (
        dataset_name in ['LatticePoint', 'SightVector', 'ObservationTime', 'AttitudeAngle'] and 
        dataset_name not in ['LatLon_Grid'] and # Explicitly exclude computed grids
        (arr.ndim == 3 or arr.ndim == 2 and dataset_name == 'LatticePoint')
    )

    if needs_block_sep:
        # If the array is 3D (e.g., SightVector: (T, C, 3))
        if arr.ndim == 3:
            num_time_steps = arr.shape[0]
            for t in range(num_time_steps):
                time_step_block = arr[t].reshape(-1, arr.shape[-1]) # Reshape to (C, V)
                for row in time_step_block:
                    lines.append(" ".join(f"{x:.10f}" for x in row) + " ")
                lines.append("") # CRITICAL: Add an extra blank line between time steps
        
        elif arr.ndim == 2 and dataset_name == 'LatticePoint':
            # LatticePoint can be 2D where T is the time step
            num_time_steps = arr.shape[0]
            for t in range(num_time_steps):
                row = np.atleast_1d(arr[t])
                lines.append(" ".join(f"{x:.10f}" for x in row) + " ")
                lines.append("") # CRITICAL: Add an extra blank line between time steps
                
    else:
        # For SatellitePosition/Velocity (T x 3) and computed Lat/Lon grids
        # We need to reshape only if the last dimension is 2 or 3 (i.e., vectors)
        if arr.ndim > 1 and arr.shape[-1] in [2, 3]: 
             arr = arr.reshape(-1, arr.shape[-1])
        elif arr.ndim == 1:
            arr = np.atleast_2d(arr)
        
        # If the input is a standard grid (N x M), ensure it is treated as N rows
        if arr.ndim == 2 and arr.shape[-1] > 3:
             # This handles the (T, C) shape of Lat/Lon data correctly
             pass
        
        for row in arr.reshape(-1, arr.shape[-1]): # Iterate over rows regardless of original shape complexity
            row_flat = np.atleast_1d(row)
            lines.append(" ".join(f"{x:.10f}" for x in row_flat) + " ") # Add trailing space

    # CRITICAL FIX (v6): Removed the final conditional check that was appending 
    # a spurious blank line for non-block-separated files (Lat/Lon grids).
    # The \n characters will be handled by the caller's "\n".join(lines).
        
    return lines


def find_geometry_paths(hdf_file, band, required_suffixes=ASP_GEOMETRY_DATASETS):
    """Find the HDF subdataset paths for the required geometry data."""
    ds = gdal.Open(hdf_file, gdal.GA_ReadOnly)
    if ds is None:
        return None, {}
    
    sensor_group = 'VNIR'
    search_terms = [f":{band}:"]
    search_terms.append(f":{sensor_group}:")
    
    subdatasets_meta = ds.GetMetadata('SUBDATASETS')
    dataset_paths = {}
    ds = None # Close the main HDF dataset handle
    
    for term in search_terms:
        if len(dataset_paths) == len(required_suffixes):
            break
            
        temp_paths = dataset_paths.copy()
        
        for i in range(1, len(subdatasets_meta) // 2 + 1):
            subdataset_path = subdatasets_meta.get(f'SUBDATASET_{i}_NAME')
            if not subdataset_path:
                continue
                
            if term in subdataset_path:
                for suffix in required_suffixes:
                    if subdataset_path.endswith(f":{suffix}"):
                        if suffix not in temp_paths:
                            temp_paths[suffix] = subdataset_path
                            
        if len(temp_paths) > len(dataset_paths):
            dataset_paths = temp_paths
            if len(dataset_paths) == len(required_suffixes):
                break
                
    missing_datasets = [s for s in required_suffixes if s not in dataset_paths]
    if missing_datasets:
        return missing_datasets, {}
        
    return [], dataset_paths

def extract_geometry_data_for_band(hdf_file, target_band, output_dir, granule_name):
    """
    Extracts the raw geometric data (in Meters, unmodified) for ASP.
    """
    missing, dataset_paths = find_geometry_paths(hdf_file, target_band, ASP_GEOMETRY_DATASETS)
    
    if missing:
        print(f"Warning: Missing required geometric datasets for {target_band}: {', '.join(missing)}. Skipping geometry extraction.")
        return

    print(f"  Extracting raw geometry data for {target_band}...")

    try:
        for suffix, hdf_path in dataset_paths.items():
            data_ds = gdal.Open(hdf_path)
            if data_ds:
                output_filename = os.path.join(output_dir, f"{granule_name}.{target_band}.{suffix}.txt")
                
                # Read raw data from HDF - DO NOT SCALE or MODIFY
                arr = data_ds.ReadAsArray()
                
                if arr is not None:
                    # Format and save array, using block separation if necessary
                    lines = format_array_to_text(arr, suffix)
                    with open(output_filename, 'w') as f:
                        # lines already contain the data, newlines, and block separators
                        f.write("\n".join(lines)) 
                        
                    print(f"    -> Exported raw geometry: {output_filename}")
        print(f"  Successfully extracted all raw geometry files for {target_band}.")

    except Exception as e:
        print(f"Error extracting geometry for {target_band}: {str(e)}")


def compute_lat_lon_from_lattice(hdf_file, band):
    """
    Compute latitude and longitude arrays from LatticePoint, SatellitePosition, and SightVector data
    using a Ray-Ellipsoid Intersection method, normalized to Kilometers for stability.
    The core latitude calculation logic remains untouched as requested.
    """
    print(f"  Attempting to compute Lat/Lon for {band}...")
    
    # WGS84 ellipsoid parameters (Normalized to KILOMETERS)
    KM_TO_M = 1000.0
    a = 6378.137       # Semi-major axis (km)
    b = 6356.752314245 # Semi-minor axis (km)
    e_sq = (a**2 - b**2) / a**2 # First eccentricity squared
    
    # --------------------------------------------------------------------------
    # 1. Search and Read Geometry Data (paths found using find_geometry_paths)
    # --------------------------------------------------------------------------
    required = ['LatticePoint', 'SatellitePosition', 'SightVector']
    missing, dataset_paths = find_geometry_paths(hdf_file, band, required)

    if missing:
        print(f"Warning: Missing required geometric datasets for {band}. Skipping Lat/Lon calculation.")
        return None, None
    
    try:
        # Read data from HDF using the discovered paths
        lattice_points = gdal.Open(dataset_paths['LatticePoint']).ReadAsArray()
        
        # ASTER L1A geometry data is typically in METERS. Scale to KM for stable internal calculation.
        sat_pos_raw = gdal.Open(dataset_paths['SatellitePosition']).ReadAsArray() / KM_TO_M
        sight_vec_raw = gdal.Open(dataset_paths['SightVector']).ReadAsArray() / KM_TO_M
        
        # Extract lattice dimensions
        num_time_steps = lattice_points.shape[0]
        num_lattice_cols = lattice_points.shape[1]
        
        # Initialize arrays with the No-Data value
        lat_lattice = np.full((num_time_steps, num_lattice_cols), NO_DATA_VALUE, dtype=np.float64)
        lon_lattice = np.full((num_time_steps, num_lattice_cols), NO_DATA_VALUE, dtype=np.float64)
        
        # Initialize failure counters
        miss_count = 0
        neg_t_count = 0
        valid_count = 0
        
        # ----------------------------------------------------------------------
        # 2. Perform geometric calculation for each lattice point (Ray-Ellipsoid Intersection)
        # ----------------------------------------------------------------------
        for t in range(num_time_steps):
            sat_pos = sat_pos_raw[t] # [X, Y, Z] in KM
            
            for c in range(num_lattice_cols):
                sight_vec = sight_vec_raw[t, c] # [X, Y, Z] direction vector in KM
                
                # Check for zero vector, which indicates invalid data
                magnitude = np.linalg.norm(sight_vec)
                if magnitude < 1e-10:
                    continue 
                sight_vec_unit = sight_vec / magnitude # Unit vector (normalized)
                
                # --- VECTOR FLIP ---
                # ASTER SightVector often points away from the Earth, requiring a flip.
                dot_product = np.dot(sat_pos, sight_vec_unit)
                if dot_product > 0:
                    sight_vec_unit = -sight_vec_unit # Force vector to point inward

                # A*t^2 + B*t + C = 0 (Quadratic equation for ray-ellipsoid intersection)
                
                # A: quadratic coefficient
                A = (sight_vec_unit[0]/a)**2 + (sight_vec_unit[1]/a)**2 + (sight_vec_unit[2]/b)**2
                
                # B: linear coefficient
                B = 2.0 * ( (sat_pos[0]*sight_vec_unit[0]/a**2) + 
                            (sat_pos[1]*sight_vec_unit[1]/a**2) + 
                            (sat_pos[2]*sight_vec_unit[2]/b**2) )
                
                # C: constant term
                C = (sat_pos[0]/a)**2 + (sat_pos[0]/a)**2 + (sat_pos[2]/b)**2 - 1.0
                
                # Discriminant: B^2 - 4AC
                discriminant = B**2 - 4.0*A*C
                
                if discriminant < 0:
                    miss_count += 1
                    continue # Ray misses the ellipsoid (Earth)
                
                # Solve for t (parameter along the ray). 
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-B - sqrt_disc) / (2.0*A)
                t2 = (-B + sqrt_disc) / (2.0*A)
                
                # --- ROBUST ROOT SELECTION ---
                # The closest intersection to the satellite is the smallest positive distance 't'.
                positive_ts = [t_val for t_val in [t1, t2] if t_val > 0]
                
                if not positive_ts:
                    neg_t_count += 1
                    continue # Neither intersection is in the 'forward' direction (positive t)
                    
                t_param = min(positive_ts)
                # --- END ROBUST ROOT SELECTION ---

                
                # Ground position in ECEF coordinates (KM)
                ground_pos = sat_pos + t_param * sight_vec_unit
                
                # ------------------------------------------------------------------
                # Convert ECEF (X, Y, Z in KM) to Geodetic Lat/Lon (Degrees)
                # ------------------------------------------------------------------
                # *** THIS SECTION IS PRESERVED AS PER USER REQUEST ***
                x, y, z = ground_pos
                p = np.sqrt(x**2 + y**2)
                
                # Longitude
                if p < 1e-10:
                    lon = 0.0 # At pole
                else:
                    lon = np.arctan2(y, x) * 180.0 / np.pi
                
                # Latitude (iterative approximation using Bowring's method)
                lat_rad = np.arctan2(z, p / (1 - e_sq))
                for _ in range(3):
                    sin_lat = np.sin(lat_rad)
                    N = a / np.sqrt(1.0 - e_sq * sin_lat**2) 
                    lat_rad = np.arctan2(z + N * e_sq * sin_lat, p)
                    
                lat = lat_rad * 180.0 / np.pi

                # --- HEMISPHERE CORRECTION ---
                # Ensures latitude sign matches the satellite's position (N/S hemisphere)
                if np.sign(lat) != np.sign(sat_pos[2]):
                     lat = -lat 
                # -----------------------------

                # Store lat/lon in lattice array
                lat_lattice[t, c] = lat
                lon_lattice[t, c] = lon
                valid_count += 1
        
        # --- DIAGNOSTIC REPORTING ---
        total_points = num_time_steps * num_lattice_cols
        
        if valid_count > 0:
            valid_lats = lat_lattice[lat_lattice != NO_DATA_VALUE] 
            valid_lons = lon_lattice[lon_lattice != NO_DATA_VALUE] 
            print(f"  Computed {valid_count} valid lattice points (Total: {total_points}).")
            print(f"  Latitude range: {np.min(valid_lats):.4f} to {np.max(valid_lats):.4f}")
        else:
            print(f"  CRITICAL: No valid ground points were calculated for {band} (0/{total_points} valid points).")
            
        if miss_count > 0 or neg_t_count > 0:
            print(f"  Failure Analysis for {band}:")
            if miss_count > 0:
                 print(f"    - {miss_count} points missed the ellipsoid (Geometric Miss, Discriminant < 0).")
            if neg_t_count > 0:
                 print(f"    - {neg_t_count} points had no positive intersection distance 't' (Incorrect Ray Direction, even after flip).")
        # --- END DIAGNOSTIC REPORTING ---
        
        return lat_lattice, lon_lattice
        
    except Exception as e:
        print(f"Error computing lat/lon for {band}: {str(e)}")
        return None, None

def extract_geotiff(subdataset_name, output_dir, granule_name):
    """
    Extracts a single subdataset as a GeoTIFF file.
    """
    _, _, full_name = parse_subdataset_name(subdataset_name)
    if full_name not in IMAGE_DATA_DATASETS:
        return # Skip non-image data
    
    band, dataset, _ = parse_subdataset_name(subdataset_name)
    output_filename = os.path.join(output_dir, f"{granule_name}.{band}.{dataset}.tif")
    
    # Use gdal_translate to export the subdataset directly
    options = gdal.TranslateOptions(
        format='GTiff',
        creationOptions=['COMPRESS=DEFLATE', 'PREDICTOR=2']
    )
    try:
        gdal.Translate(output_filename, subdataset_name, options=options)
    except Exception as e:
        print(f"Error exporting GeoTIFF {band}:{dataset}: {e}")

    
def extract_metadata_and_text(subdataset_name, output_dir, granule_name):
    """
    Extracts generic ancillary data (non-GeoTIFF, non-Geometry), limited to 3N/3B or global data.
    """
    band_name, dataset_name, full_name = parse_subdataset_name(subdataset_name)
    
    if full_name in IMAGE_DATA_DATASETS or dataset_name in ASP_GEOMETRY_DATASETS:
        return
    
    is_general_data = band_name in ["Global", "Ancillary_Data", "Cloud_Coverage_Table"]
    is_target_band_data = band_name in TARGET_BANDS
    
    if not (is_general_data or is_target_band_data):
        return

    try:
        data_ds = gdal.Open(subdataset_name)
        if data_ds:
            output_filename = os.path.join(output_dir, f"{granule_name}.{band_name}.{dataset_name}.txt")
            
            arr = data_ds.ReadAsArray()
            if arr is not None:
                lines = format_array_to_text(arr, dataset_name)
                with open(output_filename, 'w') as f:
                    f.write("\n".join(lines)) 
                
    except Exception as e:
        pass # Silently skip non-readable datasets for simplicity

def extract_aster_hdf(hdf_file, output_dir=None):
    """
    Main function to extract all data from an ASTER HDF file.
    """
    hdf_file = Path(hdf_file).resolve()
    if not hdf_file.exists():
        print(f"Error: Input HDF file not found: {hdf_file}", file=sys.stderr)
        sys.exit(1)
        
    granule_name = extract_granule_name(hdf_file)
    output_dir = Path(output_dir) if output_dir else Path(granule_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {hdf_file} to {output_dir}/ (Only VNIR_Band3N and VNIR_Band3B)")

    ds = gdal.Open(str(hdf_file), gdal.GA_ReadOnly)
    if ds is None:
        print(f"Error: Failed to open HDF file: {hdf_file}", file=sys.stderr)
        sys.exit(1)
    
    subdatasets = ds.GetMetadata('SUBDATASETS')

    # 1. Extract GeoTIFFs and Generic Text/Ancillary Data
    print("\n[STEP 1] Extracting Imagery and Ancillary Text (3N/3B only)...")
    for i in range(1, len(subdatasets) // 2 + 1):
        subdataset_name = subdatasets[f'SUBDATASET_{i}_NAME']
        extract_geotiff(subdataset_name, output_dir, granule_name)
        extract_metadata_and_text(subdataset_name, output_dir, granule_name)

    ds = None # Close the main HDF dataset handle
    
    # 2. Extract and rename geometry files specifically for stereo bands (ASP input)
    print(f"\n[STEP 2] Extracting RAW Geometry Data for ASP ({', '.join(TARGET_BANDS)}) with Block Separation...")
    for band in TARGET_BANDS:
        extract_geometry_data_for_band(str(hdf_file), band, output_dir, granule_name)

    # 3. Compute and Save Latitude/Longitude files 
    print(f"\n[STEP 3] Computing and Exporting Latitude/Longitude Grids ({', '.join(TARGET_BANDS)}) without blank lines...")
    for band in TARGET_BANDS:
        lat_data, lon_data = compute_lat_lon_from_lattice(str(hdf_file), band)

        if lat_data is not None and lon_data is not None:
            # Format and save Latitude
            lat_file = os.path.join(output_dir, f"{granule_name}.{band}.Latitude.txt")
            # Using the dummy name "LatLon_Grid" to explicitly disable block separation
            lines = format_array_to_text(lat_data, "LatLon_Grid") 
            with open(lat_file, 'w') as f:
                f.write("\n".join(lines))
            print(f"Exported Latitude: {Path(lat_file).name}")
            
            # Format and save Longitude
            lon_file = os.path.join(output_dir, f"{granule_name}.{band}.Longitude.txt")
            # Using the dummy name "LatLon_Grid" to explicitly disable block separation
            lines = format_array_to_text(lon_data, "LatLon_Grid") 
            with open(lon_file, 'w') as f:
                f.write("\n".join(lines))
            print(f"Exported Longitude: {Path(lon_file).name}")
        
    print(f"\nExtraction complete! Files saved to: {output_dir.resolve()}/")

def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python extract_aster.py <input_hdf_file> [output_directory]")
        print("\nExample:")
        print("  python extract_aster.py AST_L1A_00408172025181237_20251010234802.hdf")
        sys.exit(1)
    
    hdf_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_aster_hdf(hdf_file, output_dir)

if __name__ == '__main__':
    main()

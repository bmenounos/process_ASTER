#!/usr/bin/env python3
"""
Extract ASTER L1A HDF datasets to separate files with CORRECTED lat/lon.
Creates directory structure for stereopipe/micmac processing.
"""
from osgeo import gdal
import numpy as np
import sys
import os

def compute_corrected_latlon_lattice(hdf_file, band='VNIR_Band3N'):
    """
    Compute corrected lat/lon for LATTICE POINTS ONLY (not full image).
    Uses orbital→ECI transformation + geocentric latitude.
    
    Returns:
        lat_lattice: (num_time_steps, num_lattice_cols) array
        lon_lattice: (num_time_steps, num_lattice_cols) array
    """
    # WGS84
    a = 6378137.0
    b = 6356752.314245
    
    # Find datasets
    ds = gdal.Open(hdf_file)
    sds = ds.GetMetadata('SUBDATASETS')
    
    paths = {}
    for key, val in sds.items():
        if 'NAME' in key and band in val:
            if 'SatellitePosition' in val:
                paths['pos'] = val
            elif 'SatelliteVelocity' in val:
                paths['vel'] = val
            elif 'SightVector' in val:
                paths['sight'] = val
    
    # Read geometry data
    sat_pos_raw = gdal.Open(paths['pos']).ReadAsArray()
    sat_vel_raw = gdal.Open(paths['vel']).ReadAsArray()
    sight_vec_raw = gdal.Open(paths['sight']).ReadAsArray()
    
    num_time_steps = sight_vec_raw.shape[0]
    num_lattice_cols = sight_vec_raw.shape[1]
    
    # Compute lattice lat/lon
    lat_lattice = np.zeros((num_time_steps, num_lattice_cols), dtype=np.float64)
    lon_lattice = np.zeros((num_time_steps, num_lattice_cols), dtype=np.float64)
    
    for t in range(num_time_steps):
        sat_pos = sat_pos_raw[t].astype(np.float64)
        sat_vel = sat_vel_raw[t].astype(np.float64)
        
        # Build orbital frame in ECI
        z_orb = -sat_pos / np.linalg.norm(sat_pos)
        orbit_normal = np.cross(sat_pos, sat_vel)
        y_orb = -orbit_normal / np.linalg.norm(orbit_normal)
        x_orb = np.cross(y_orb, z_orb)
        R_orb_to_eci = np.column_stack([x_orb, y_orb, z_orb])
        
        for c in range(num_lattice_cols):
            sv_orb = sight_vec_raw[t, c].astype(np.float64)
            if np.linalg.norm(sv_orb) < 1e-10:
                continue
            
            # Transform: orbital → ECI
            sv_eci = R_orb_to_eci @ sv_orb
            sv_unit = sv_eci / np.linalg.norm(sv_eci)
            
            # Ray-ellipsoid intersection
            A = (sv_unit[0]/a)**2 + (sv_unit[1]/a)**2 + (sv_unit[2]/b)**2
            B = 2*((sat_pos[0]*sv_unit[0]/a**2) + (sat_pos[1]*sv_unit[1]/a**2) + (sat_pos[2]*sv_unit[2]/b**2))
            C = (sat_pos[0]/a)**2 + (sat_pos[1]/a)**2 + (sat_pos[2]/b)**2 - 1
            
            disc = B**2 - 4*A*C
            if disc < 0:
                continue
            
            t_param = (-B - np.sqrt(disc)) / (2*A)
            if t_param <= 0:
                continue
            
            # Ground point
            gx = sat_pos[0] + t_param * sv_unit[0]
            gy = sat_pos[1] + t_param * sv_unit[1]
            gz = sat_pos[2] + t_param * sv_unit[2]
            
            # Geocentric lat/lon
            p = np.sqrt(gx**2 + gy**2)
            lon_lattice[t, c] = np.degrees(np.arctan2(gy, gx))
            lat_lattice[t, c] = np.degrees(np.arctan2(gz, p))
    
    return lat_lattice, lon_lattice

def extract_all_datasets(hdf_file, output_dir):
    """Extract all datasets from HDF to separate files."""
    base_name = os.path.splitext(os.path.basename(hdf_file))[0]
    
    ds = gdal.Open(hdf_file)
    sds = ds.GetMetadata('SUBDATASETS')
    
    # Group by band
    bands = {}
    for key, val in sds.items():
        if 'NAME' in key:
            # Extract band name from path
            for band_name in ['VNIR_Band1', 'VNIR_Band2', 'VNIR_Band3N', 'VNIR_Band3B',
                             'SWIR_Band4', 'SWIR_Band5', 'SWIR_Band6', 'SWIR_Band7', 'SWIR_Band8', 'SWIR_Band9',
                             'TIR_Band10', 'TIR_Band11', 'TIR_Band12', 'TIR_Band13', 'TIR_Band14']:
                if band_name in val:
                    if band_name not in bands:
                        bands[band_name] = {}
                    
                    # Extract dataset type
                    for ds_type in ['ImageData', 'Latitude', 'Longitude', 'LatticePoint', 
                                   'SatellitePosition', 'SatelliteVelocity', 'SightVector',
                                   'AttitudeAngle', 'AttitudeRate', 'ObservationTime', 
                                   'RadiometricCorrTable']:
                        if ds_type in val:
                            bands[band_name][ds_type] = val
                            break
    
    # Process only VNIR bands
    vnir_bands = {k: v for k, v in bands.items() if k.startswith('VNIR')}
    
    for band_name, datasets in vnir_bands.items():
        print(f"\nProcessing {band_name}...")
        
        # Compute corrected lat/lon lattice for VNIR bands
        print(f"  Computing corrected lat/lon lattice...")
        lat_lattice, lon_lattice = compute_corrected_latlon_lattice(hdf_file, band_name)
        
        # Write lattice lat/lon files (not full image grid)
        lat_file = os.path.join(output_dir, f"{base_name}.{band_name}.Latitude.txt")
        lon_file = os.path.join(output_dir, f"{base_name}.{band_name}.Longitude.txt")
        
        print(f"  Writing CORRECTED Latitude ({lat_lattice.shape[0]}×{lat_lattice.shape[1]}) → {os.path.basename(lat_file)}")
        np.savetxt(lat_file, lat_lattice, fmt='%.6f')
        
        print(f"  Writing CORRECTED Longitude ({lon_lattice.shape[0]}×{lon_lattice.shape[1]}) → {os.path.basename(lon_file)}")
        np.savetxt(lon_file, lon_lattice, fmt='%.6f')
        
        for ds_type, ds_path in datasets.items():
            # Skip Latitude/Longitude since we already wrote corrected versions
            if ds_type in ['Latitude', 'Longitude']:
                continue
            
            output_file = os.path.join(output_dir, f"{base_name}.{band_name}.{ds_type}")
            
            # Handle different dataset types
            if ds_type == 'ImageData':
                # Save as GeoTIFF
                output_file += '.tif'
                print(f"  Extracting {ds_type} → {os.path.basename(output_file)}")
                
                src_ds = gdal.Open(ds_path)
                data = src_ds.ReadAsArray()
                
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(output_file, data.shape[1], data.shape[0], 1, 
                                      gdal.GDT_UInt16, options=['COMPRESS=LZW'])
                out_ds.GetRasterBand(1).WriteArray(data)
                out_ds.FlushCache()
                out_ds = None
                
            else:
                # Save as text file
                output_file += '.txt'
                print(f"  Extracting {ds_type} → {os.path.basename(output_file)}")
                
                src_ds = gdal.Open(ds_path)
                data = src_ds.ReadAsArray()
                
                if data is not None:
                    # Handle different array dimensions
                    if data.ndim == 1:
                        np.savetxt(output_file, data, fmt='%.6f')
                    elif data.ndim == 2:
                        np.savetxt(output_file, data, fmt='%.6f')
                    elif data.ndim == 3:
                        # For 3D arrays (e.g., SightVector with shape [time, cols, 3])
                        # Reshape to 2D: each row is flattened vector
                        reshaped = data.reshape(-1, data.shape[-1])
                        np.savetxt(output_file, reshaped, fmt='%.6f')
                    else:
                        print(f"    Warning: Skipping {ds_type} - unsupported dimensions: {data.shape}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_hdf_datasets.py <input.hdf> [output_dir]")
        sys.exit(1)
    
    hdf_file = sys.argv[1]
    
    # Create output directory named after HDF file
    base_name = os.path.splitext(os.path.basename(hdf_file))[0]
    
    if len(sys.argv) > 2:
        output_dir = os.path.join(sys.argv[2], base_name)
    else:
        output_dir = base_name
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracting to: {output_dir}")
    
    extract_all_datasets(hdf_file, output_dir)
    
    print("\n" + "="*70)
    print("Done! Directory structure:")
    print("="*70)
    print(f"\n{output_dir}/")
    for f in sorted(os.listdir(output_dir))[:5]:
        print(f"  {f}")
    print(f"  ... (and more)")

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
ASTER Complete Workflow for Highest Quality DEMs - WITH AUTO UTM DETECTION
Combines: HDF extraction + Radiometric calibration + Snow enhancement + s2p processing

NEW: Automatically detects UTM zone from lat/lon files (no --utm-zone needed!)

Usage:
    python3 aster_complete_workflow.py AST_L1A_xxx.hdf [options]

The script now AUTO-DETECTS the UTM zone from the extracted Latitude/Longitude files!
No need to specify --utm-zone unless you want to override.
"""

import os
import sys
import subprocess
import argparse
import json
import shutil
from pathlib import Path
import numpy as np

try:
    from osgeo import gdal
    import rasterio
    from skimage import exposure
    from scipy import ndimage
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    sys.exit(1)


def determine_utm_zone_from_latlon(lat, lon):
    """
    Determine UTM zone from latitude and longitude.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
    
    Returns:
        UTM zone string (e.g., "11 +north")
    """
    zone_num = int((lon + 180) / 6) + 1
    hemisphere = "+north" if lat >= 0 else "+south"
    return f"{zone_num} {hemisphere}"


def read_latlon_from_file(latlon_file):
    """Read latitude or longitude from extracted text file. Returns center point value."""
    try:
        with open(latlon_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        values = []
        for line in lines:
            parts = line.split()
            values.extend([float(p) for p in parts if p])
        
        values = np.array(values)
        valid_values = values[values != -9999.0]
        
        if len(valid_values) == 0:
            return None
        
        return np.median(valid_values)
        
    except Exception as e:
        print(f"Warning: Could not read {latlon_file}: {e}")
        return None


class ASTERWorkflow:
    """Complete ASTER processing workflow with auto UTM detection"""
    
    def __init__(self, hdf_file, utm_zone=None, **kwargs):
        self.hdf_file = Path(hdf_file).resolve()
        self.utm_zone_provided = utm_zone
        self.utm_zone = utm_zone
        self.scene_name = self.hdf_file.stem
        
        self.enhance = kwargs.get('enhance', 'none')
        self.config_file = kwargs.get('config')
        self.output_dir = Path(kwargs.get('output_dir', self.scene_name)).resolve()
        self.resolution = kwargs.get('resolution', 30)
        self.skip_extraction = kwargs.get('skip_extraction', False)
        self.skip_enhancement = kwargs.get('skip_enhancement', False)
        self.quality = kwargs.get('quality', 'balanced')
        
        self.quality_presets = {
            'fast': {
                'matching_algorithm': 'stereosgm_gpu',
                'tile_size': 500,
                'max_processes': 1,
                'enhance': False
            },
            'balanced': {
                'matching_algorithm': 'stereosgm_gpu',
                'tile_size': 400,
                'max_processes': 1,
                'horizontal_margin': 100,
                'vertical_margin': 50,
                'enhance': 'auto'
            },
            'best': {
                'matching_algorithm': 'mgm',
                'tile_size': 600,
                'max_processes': 4,
                'horizontal_margin': 150,
                'vertical_margin': 75,
                'mgm_p1': 4,
                'mgm_p2': 50,
                'mgm_nb_directions': 16,
                'mgm_leftright_control': False,
                'enhance': True
            }
        }
        
        self.extraction_dir = None
        self.calibrated_dir = self.output_dir / 'calibrated'  # Initialize early
        self.tif_3n = None
        self.tif_3b = None
        self.rpc_3n = None
        self.rpc_3b = None
        
    def log(self, message, level='INFO'):
        prefix = {'INFO': '  →', 'SUCCESS': '  ✓', 'WARNING': '  ⚠', 'ERROR': '  ✗', 'SECTION': '\n==='}
        print(f"{prefix.get(level, '  ')} {message}")
    
    def run_command(self, cmd, description=None, capture=False):
        if description:
            self.log(description, 'INFO')
        
        cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
        self.log(f"Running: {cmd_str}", 'INFO')
        
        try:
            if capture:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                return result.stdout
            else:
                subprocess.run(cmd, check=True)
            return True
        except subprocess.CalledProcessError as e:
            self.log(f"Command failed: {e}", 'ERROR')
            if hasattr(e, 'stderr') and e.stderr:
                self.log(f"Error: {e.stderr}", 'ERROR')
            return False
    
    def detect_utm_zone(self):
        """Auto-detect UTM zone from extracted Latitude/Longitude files"""
        self.log("Auto-detecting UTM zone from Latitude/Longitude files...", 'INFO')
        
        # Try multiple naming patterns to find lat/lon files
        patterns = [
            # Pattern from aster_hdf2geotiff.py: AST_xxx.VNIR_Band3N.Latitude.txt
            (f"{self.scene_name}.VNIR_Band3N.Latitude.txt", f"{self.scene_name}.VNIR_Band3N.Longitude.txt"),
            (f"{self.scene_name}.VNIR_Band3B.Latitude.txt", f"{self.scene_name}.VNIR_Band3B.Longitude.txt"),
            # Alternative: with wildcards
            (f"{self.scene_name}*.VNIR_Band3N*.Latitude.txt", f"{self.scene_name}*.VNIR_Band3N*.Longitude.txt"),
            (f"{self.scene_name}*.VNIR_Band3B*.Latitude.txt", f"{self.scene_name}*.VNIR_Band3B*.Longitude.txt"),
            # Flexible patterns
            (f"{self.scene_name}*Latitude.txt", f"{self.scene_name}*Longitude.txt"),
        ]
        
        lat_file, lon_file = None, None
        
        for lat_pattern, lon_pattern in patterns:
            # Use rglob for recursive search to handle nested directories
            lat_files = list(self.extraction_dir.rglob(lat_pattern))
            lon_files = list(self.extraction_dir.rglob(lon_pattern))
            
            if lat_files and lon_files:
                lat_file = lat_files[0]
                lon_file = lon_files[0]
                self.log(f"Found in: {lat_file.parent.relative_to(self.extraction_dir)}", 'INFO')
                break
        
        if not lat_file or not lon_file:
            self.log("Could not find Latitude/Longitude files", 'WARNING')
            # Show what files ARE there for debugging
            txt_files = list(self.extraction_dir.rglob("*.txt"))
            if txt_files:
                self.log(f"Found {len(txt_files)} .txt files in subdirectories:", 'INFO')
                for f in txt_files[:5]:
                    self.log(f"  - {f.relative_to(self.extraction_dir)}", 'INFO')
            return False
        
        self.log(f"Reading: {lat_file.name}", 'INFO')
        self.log(f"Reading: {lon_file.name}", 'INFO')
        
        lat_center = read_latlon_from_file(lat_file)
        lon_center = read_latlon_from_file(lon_file)
        
        if lat_center is None or lon_center is None:
            self.log("Could not parse Latitude/Longitude values", 'WARNING')
            return False
        
        self.utm_zone = determine_utm_zone_from_latlon(lat_center, lon_center)
        
        self.log(f"Scene center: {lat_center:.4f}°N, {lon_center:.4f}°E", 'SUCCESS')
        self.log(f"Auto-detected UTM zone: {self.utm_zone}", 'SUCCESS')
        
        return True
    
    def extract_hdf(self):
        self.log("STEP 1: Extracting HDF Data", 'SECTION')
        
        if self.skip_extraction:
            self.log("Skipping extraction (--skip-extraction)", 'WARNING')
            self.extraction_dir = self.output_dir / self.scene_name
            
            if not self.utm_zone_provided:
                if not self.detect_utm_zone():
                    self.log("Auto-detection failed, UTM zone required", 'ERROR')
                    return False
            return True
        
        script_dir = Path(__file__).parent
        extractor = script_dir / 'aster_hdf2geotiff.py'
        
        if not extractor.exists():
            extractor = Path.cwd() / 'aster_hdf2geotiff.py'
        
        if not extractor.exists():
            if shutil.which('aster_hdf2geotiff.py'):
                extractor = 'aster_hdf2geotiff.py'
            else:
                self.log("aster_hdf2geotiff.py not found", 'ERROR')
                return False
        
        self.extraction_dir = self.output_dir / self.scene_name
        
        cmd = ['python3', str(extractor), str(self.hdf_file), str(self.extraction_dir)]
        
        if not self.run_command(cmd, "Extracting HDF file"):
            return False
        
        self.log(f"Extraction complete: {self.extraction_dir}", 'SUCCESS')
        
        if not self.utm_zone_provided:
            if not self.detect_utm_zone():
                self.log("Auto-detection failed, please provide --utm-zone", 'ERROR')
                return False
        
        return True
    
    def extract_ucc_from_hdf(self, band='3N'):
        """Extract Unit Conversion Coefficients from HDF metadata. Always returns a valid value."""
        default_ucc = {'3N': 0.676, '3B': 0.676}
        
        try:
            hdf_ds = gdal.Open(str(self.hdf_file), gdal.GA_ReadOnly)
            if not hdf_ds:
                self.log(f"Could not open HDF, using default UCC: {default_ucc[band]}", 'WARNING')
                return default_ucc[band]
            
            subdatasets = hdf_ds.GetSubDatasets()
            target_name = f'ImageData{band}'
            band_ds = None
            
            for subdataset in subdatasets:
                if target_name in subdataset[0]:
                    band_ds = gdal.Open(subdataset[0], gdal.GA_ReadOnly)
                    break
            
            if not band_ds:
                self.log(f"Band {band} not found in HDF, using default UCC: {default_ucc[band]}", 'WARNING')
                return default_ucc[band]
            
            metadata = band_ds.GetMetadata()
            ucc_keys = ['UnitConversionCoeff', 'INCL', 'RadianceConversionCoeff']
            
            for key in ucc_keys:
                if key in metadata:
                    try:
                        ucc = float(metadata[key])
                        if ucc > 0:  # Sanity check
                            self.log(f"Found UCC for Band {band}: {ucc}", 'SUCCESS')
                            return ucc
                    except:
                        pass
            
            self.log(f"UCC not found in metadata, using default: {default_ucc[band]}", 'WARNING')
            return default_ucc[band]
            
        except Exception as e:
            self.log(f"Error extracting UCC: {e}", 'WARNING')
            self.log(f"Using default UCC: {default_ucc[band]}", 'WARNING')
            return default_ucc[band]
    
    def apply_radiometric_calibration(self):
        self.log("STEP 2: Applying Radiometric Calibration", 'SECTION')
        
        # Try multiple patterns to find 3N and 3B ImageData TIFs
        patterns_3n = [
            f"{self.scene_name}.VNIR_Band3N.ImageData.tif",
            f"{self.scene_name}*VNIR_Band3N*ImageData*.tif",
            f"{self.scene_name}*3N*ImageData*.tif"
        ]
        patterns_3b = [
            f"{self.scene_name}.VNIR_Band3B.ImageData.tif",
            f"{self.scene_name}*VNIR_Band3B*ImageData*.tif",
            f"{self.scene_name}*3B*ImageData*.tif"
        ]
        
        tifs_3n = []
        for pattern in patterns_3n:
            tifs_3n = list(self.extraction_dir.rglob(pattern))
            if tifs_3n:
                break
        
        tifs_3b = []
        for pattern in patterns_3b:
            tifs_3b = list(self.extraction_dir.rglob(pattern))
            if tifs_3b:
                break
        
        if not tifs_3n or not tifs_3b:
            self.log("Could not find extracted TIF files", 'ERROR')
            # Show what TIFs are available
            all_tifs = list(self.extraction_dir.rglob("*.tif"))
            if all_tifs:
                self.log(f"Found {len(all_tifs)} TIF files:", 'INFO')
                for f in all_tifs[:5]:
                    self.log(f"  - {f.relative_to(self.extraction_dir)}", 'INFO')
            return False
        
        input_3n, input_3b = tifs_3n[0], tifs_3b[0]
        
        self.log(f"Found 3N: {input_3n.name}", 'INFO')
        self.log(f"Found 3B: {input_3b.name}", 'INFO')
        
        # Extract UCC from HDF (will use defaults if not found)
        ucc_3n = self.extract_ucc_from_hdf('3N')
        ucc_3b = self.extract_ucc_from_hdf('3B')
        
        # Verify we have valid UCC values
        if ucc_3n is None or ucc_3b is None:
            self.log("Failed to get UCC values (this should not happen!)", 'ERROR')
            return False
        
        self.calibrated_dir = self.output_dir / 'calibrated'
        self.calibrated_dir.mkdir(parents=True, exist_ok=True)
        
        self.tif_3n = self.calibrated_dir / f"{self.scene_name}_3N_radiance.tif"
        if not self.calibrate_image(input_3n, self.tif_3n, ucc_3n, '3N'):
            return False
        
        self.tif_3b = self.calibrated_dir / f"{self.scene_name}_3B_radiance.tif"
        if not self.calibrate_image(input_3b, self.tif_3b, ucc_3b, '3B'):
            return False
        
        self.log("Radiometric calibration complete", 'SUCCESS')
        return True
    
    def calibrate_image(self, input_file, output_file, ucc, band_name):
        """Convert DN to radiance: Radiance = DN * UCC"""
        self.log(f"Calibrating Band {band_name}...", 'INFO')
        
        # Validate UCC
        if ucc is None:
            self.log(f"ERROR: UCC is None for band {band_name}", 'ERROR')
            return False
        
        if ucc <= 0:
            self.log(f"WARNING: Invalid UCC value {ucc}, using default 0.676", 'WARNING')
            ucc = 0.676
        
        self.log(f"Using UCC = {ucc} for Band {band_name}", 'INFO')
        
        try:
            with rasterio.open(input_file) as src:
                dn = src.read(1)
                
                # Check data type and range
                self.log(f"Input data type: {dn.dtype}, range: {dn.min()}-{dn.max()}", 'INFO')
                
                # Convert DN to radiance
                radiance = dn.astype(np.float32) * ucc
                
                self.log(f"Output range: {radiance.min():.2f}-{radiance.max():.2f} W/(m²·sr·μm)", 'INFO')
                
                # Create a clean output profile (don't copy all settings from input)
                output_profile = {
                    'driver': 'GTiff',
                    'dtype': rasterio.float32,
                    'width': src.width,
                    'height': src.height,
                    'count': 1,
                    'crs': src.crs,
                    'transform': src.transform,
                    'compress': 'lzw',
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256
                }
                
                # Delete existing file if present (may be corrupted from previous attempt)
                if output_file.exists():
                    output_file.unlink()
                
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    dst.write(radiance, 1)
                    dst.update_tags(UNITS='W/(m^2*sr*um)', UCC=str(ucc), CALIBRATION='Radiance')
            
            self.log(f"Calibrated: {output_file.name}", 'SUCCESS')
            return True
        except Exception as e:
            self.log(f"Calibration failed: {e}", 'ERROR')
            import traceback
            self.log(f"Traceback: {traceback.format_exc()}", 'ERROR')
            return False
    
    def detect_snow_coverage(self, image_path, threshold=0.2):
        try:
            with rasterio.open(image_path) as src:
                data = src.read(1)
                valid = data > 0
                bright = data > 50
                snow_fraction = bright.sum() / valid.sum() if valid.sum() > 0 else 0
            return snow_fraction > threshold
        except:
            return False
    
    def apply_enhancement(self):
        self.log("STEP 3: Applying Enhancement", 'SECTION')
        
        if self.skip_enhancement:
            self.log("Skipping enhancement (--skip-enhancement)", 'WARNING')
            return True
        
        preset = self.quality_presets[self.quality]
        enhance_mode = preset.get('enhance', self.enhance)
        
        if enhance_mode == 'auto':
            has_snow = self.detect_snow_coverage(self.tif_3n)
            enhance_mode = has_snow
            self.log(f"Snow detected: {has_snow}", 'INFO')
        
        if not enhance_mode:
            self.log("Enhancement not needed", 'INFO')
            return True
        
        enhanced_3n = self.calibrated_dir / f"{self.scene_name}_3N_enhanced.tif"
        enhanced_3b = self.calibrated_dir / f"{self.scene_name}_3B_enhanced.tif"
        
        if self.enhance_image(self.tif_3n, enhanced_3n, '3N'):
            self.tif_3n = enhanced_3n
        
        if self.enhance_image(self.tif_3b, enhanced_3b, '3B'):
            self.tif_3b = enhanced_3b
        
        self.log("Enhancement complete", 'SUCCESS')
        return True
    
    def enhance_image(self, input_file, output_file, band_name):
        self.log(f"Enhancing Band {band_name}...", 'INFO')
        
        try:
            with rasterio.open(input_file) as src:
                radiance = src.read(1)
                
                nodata_mask = radiance == 0
                valid_data = radiance[~nodata_mask]
                
                if len(valid_data) == 0:
                    return False
                
                vmin, vmax = valid_data.min(), valid_data.max()
                rad_norm = np.zeros_like(radiance, dtype=np.float32)
                rad_norm[~nodata_mask] = (radiance[~nodata_mask] - vmin) / (vmax - vmin)
                
                rad_enhanced = exposure.equalize_adapthist(rad_norm, kernel_size=100, clip_limit=0.01, nbins=256)
                
                lowpass = ndimage.gaussian_filter(rad_enhanced, sigma=2)
                highpass = rad_enhanced - lowpass
                rad_enhanced = rad_enhanced + 0.3 * highpass
                rad_enhanced = np.clip(rad_enhanced, 0, 1)
                
                rad_out = np.zeros_like(radiance)
                rad_out[~nodata_mask] = rad_enhanced[~nodata_mask] * (vmax - vmin) + vmin
                
                # Create clean output profile
                output_profile = {
                    'driver': 'GTiff',
                    'dtype': rasterio.float32,
                    'width': src.width,
                    'height': src.height,
                    'count': 1,
                    'crs': src.crs,
                    'transform': src.transform,
                    'compress': 'lzw',
                    'tiled': True,
                    'blockxsize': 256,
                    'blockysize': 256
                }
                
                # Delete existing file if present
                if output_file.exists():
                    output_file.unlink()
                
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    dst.write(rad_out, 1)
                    dst.update_tags(ENHANCEMENT='CLAHE+HighPass')
            
            self.log(f"Enhanced: {output_file.name}", 'SUCCESS')
            return True
        except Exception as e:
            self.log(f"Enhancement failed: {e}", 'ERROR')
            return False
    
    def generate_rpc_files(self):
        self.log("STEP 2: Generating RPC Files with MicMac", 'SECTION')
        
        if not shutil.which('mm3d'):
            self.log("MicMac not found", 'ERROR')
            return False
        
        # Ensure calibrated_dir is absolute before changing directories
        self.calibrated_dir = self.calibrated_dir.absolute()
        self.calibrated_dir.mkdir(parents=True, exist_ok=True)
        
        original_cwd = Path.cwd()
        
        try:
            # The extraction creates nested directories
            # Need to find the directory with raw .ImageData.tif files (not processed _3N.tif files)
            self.log(f"Extraction dir: {self.extraction_dir}", 'INFO')
            
            # Search for raw ImageData.tif files (from aster_hdf2geotiff.py)
            raw_tif_files = list(self.extraction_dir.rglob("*.ImageData.tif"))
            
            if not raw_tif_files:
                self.log("No .ImageData.tif files found - looking for raw ASTER data", 'ERROR')
                self.log("These files should be created by aster_hdf2geotiff.py", 'ERROR')
                # Show what we DO have
                all_tifs = list(self.extraction_dir.rglob("*.tif"))
                if all_tifs:
                    self.log("TIF files found:", 'INFO')
                    for t in all_tifs[:5]:
                        self.log(f"  - {t.name}", 'INFO')
                return False
            
            # The actual data directory is where the ImageData TIF files are
            actual_data_dir = raw_tif_files[0].parent
            self.log(f"Found raw data files in: {actual_data_dir}", 'INFO')
            
            # Change to the directory with the raw data files
            os.chdir(actual_data_dir)
            self.log(f"Working directory: {Path.cwd()}", 'INFO')
            
            # Verify we have the expected files
            required_patterns = [
                "*.VNIR_Band3N.ImageData.tif",
                "*.VNIR_Band3B.ImageData.tif",
                "*.VNIR_Band3N.Latitude.txt",
                "*.VNIR_Band3N.Longitude.txt"
            ]
            
            found_all = True
            for pattern in required_patterns:
                files = list(Path.cwd().glob(pattern))
                if not files:
                    self.log(f"Missing: {pattern}", 'WARNING')
                    found_all = False
            
            if not found_all:
                self.log("Missing required raw data files for ASTERGT2MM", 'ERROR')
                return False
            
            # MicMac ASTERGT2MM command
            self.log("Running MicMac ASTERGT2MM for RPC generation...", 'INFO')
            cmd = ['mm3d', 'SateLib', 'ASTERGT2MM', self.scene_name]
            self.log(f"Command: {' '.join(cmd)}", 'INFO')
            self.log("", 'INFO')
            
            # Run command without capturing output
            result = subprocess.run(cmd)
            
            if result.returncode != 0:
                self.log("", 'INFO')
                self.log("ASTERGT2MM command failed", 'ERROR')
                return False
            
            self.log("", 'INFO')
            self.log("RPC generation successful!", 'SUCCESS')
            
            # ASTERGT2MM creates AST_L1A_xxx.xml files (MicMac format)
            # Now we need to run Aster2Grid to create RPC_ files (s2p format)
            
            # Find the AST_L1A XML files first - check current and parent directory
            ast_xml_files = []
            ast_xml_files.extend(Path.cwd().glob(f"{self.scene_name}_3N.xml"))
            ast_xml_files.extend(Path.cwd().glob(f"{self.scene_name}_3B.xml"))
            
            # Also check parent directory
            if not ast_xml_files:
                ast_xml_files.extend(Path.cwd().parent.glob(f"{self.scene_name}_3N.xml"))
                ast_xml_files.extend(Path.cwd().parent.glob(f"{self.scene_name}_3B.xml"))
            
            # Also try wildcard patterns
            if not ast_xml_files:
                ast_xml_files.extend(Path.cwd().glob("*_3N.xml"))
                ast_xml_files.extend(Path.cwd().glob("*_3B.xml"))
                # Filter out RPC_ and FalseColor files
                ast_xml_files = [f for f in ast_xml_files if not f.name.startswith('RPC_') and 'FalseColor' not in f.name]
            
            if not ast_xml_files or len(ast_xml_files) < 2:
                self.log("Could not find AST_L1A XML files to convert", 'ERROR')
                self.log(f"Searched for: {self.scene_name}_3[NB].xml", 'INFO')
                self.log(f"In directory: {Path.cwd()}", 'INFO')
                # List what XML files exist
                all_xml = list(Path.cwd().glob("*.xml")) + list(Path.cwd().parent.glob("*.xml"))
                if all_xml:
                    self.log("XML files found:", 'INFO')
                    for x in all_xml[:10]:
                        self.log(f"  - {x.name}", 'INFO')
                return False
            
            self.log(f"Found {len(ast_xml_files)} MicMac XML files, converting to RPC format...", 'INFO')
            
            # Run Aster2Grid to generate RPC_ files
            # This creates the RPC_xxx.xml files that s2p can read
            utm_proj = f"+proj=utm +zone={self.utm_zone} +datum=WGS84 +units=m +no_defs"
            
            # Change to the directory containing the XML files
            xml_dir = ast_xml_files[0].parent
            current_dir = Path.cwd()
            if xml_dir != current_dir:
                os.chdir(xml_dir)
                self.log(f"Changed to directory: {xml_dir}", 'INFO')
            
            for xml_file in ast_xml_files:
                self.log(f"Converting: {xml_file.name}", 'INFO')
                cmd = [
                    'mm3d', 'SateLib', 'Aster2Grid', xml_file.name, '20',
                    utm_proj, 'HMin=-500', 'HMax=9000', 
                    'expDIMAP=1', 'expGrid=1'
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    self.log(f"Warning: Aster2Grid failed for {xml_file.name}", 'WARNING')
                    if result.stderr:
                        self.log(f"stderr: {result.stderr[:200]}", 'INFO')
                else:
                    self.log(f"✓ Generated RPC for {xml_file.name}", 'SUCCESS')
            
            # Stay in the directory where RPC_ files were created
            # (they should be in the same directory as the XML files)
            
            # Now find the RPC_ XML files that were just created
            xml_files = []
            
            # 1. PRIORITY: Look for RPC_ files in current directory
            xml_files.extend(Path.cwd().glob("RPC_*_3N.xml"))
            xml_files.extend(Path.cwd().glob("RPC_*_3B.xml"))
            
            # 2. Check parent directory
            if not xml_files:
                xml_files.extend(Path.cwd().parent.glob("RPC_*_3N.xml"))
                xml_files.extend(Path.cwd().parent.glob("RPC_*_3B.xml"))
            
            # Remove duplicates
            xml_files = list(set(xml_files))
            
            if xml_files:
                self.log(f"Found {len(xml_files)} RPC XML files:", 'SUCCESS')
                for xml in xml_files:
                    self.log(f"  → {xml.name}", 'INFO')
            else:
                self.log("No RPC_ XML files found after Aster2Grid!", 'ERROR')
                return False
            
            # Find MicMac's calibrated TIF files (created by ASTERGT2MM/Aster2Grid)
            # These have calibration already applied by MicMac
            self.log("Finding MicMac's calibrated TIF files...", 'INFO')
            micmac_tif_3n = list(Path.cwd().glob(f"{self.scene_name}_3N.tif"))
            micmac_tif_3b = list(Path.cwd().glob(f"{self.scene_name}_3B.tif"))
            
            if not micmac_tif_3n or not micmac_tif_3b:
                self.log("Could not find MicMac TIF files", 'ERROR')
                all_tifs = list(Path.cwd().glob("*.tif"))
                self.log(f"TIF files in directory: {[t.name for t in all_tifs[:10]]}", 'INFO')
                return False
            
            micmac_tif_3n = micmac_tif_3n[0]
            micmac_tif_3b = micmac_tif_3b[0]
            
            self.log(f"Found MicMac TIFs:", 'SUCCESS')
            self.log(f"  3N: {micmac_tif_3n.name}", 'INFO')
            self.log(f"  3B: {micmac_tif_3b.name}", 'INFO')
            
            # Copy MicMac's TIF files to calibrated directory
            self.tif_3n = self.calibrated_dir / f"{self.scene_name}_3N_micmac.tif"
            self.tif_3b = self.calibrated_dir / f"{self.scene_name}_3B_micmac.tif"
            
            shutil.copy2(micmac_tif_3n, self.tif_3n)
            shutil.copy2(micmac_tif_3b, self.tif_3b)
            
            self.log(f"Copied MicMac TIFs to: {self.calibrated_dir.name}", 'SUCCESS')
            
            # Apply enhancement if requested
            if self.enhance != 'none':
                self.log("Applying enhancement to MicMac TIFs...", 'INFO')
                
                # Enhance 3N
                enhanced_3n = self.calibrated_dir / f"{self.scene_name}_3N_enhanced.tif"
                if self.enhance_image(self.tif_3n, enhanced_3n, '3N'):
                    self.tif_3n_enhanced = enhanced_3n
                    self.log(f"✓ Enhanced 3N: {enhanced_3n.name}", 'SUCCESS')
                
                # Enhance 3B
                enhanced_3b = self.calibrated_dir / f"{self.scene_name}_3B_enhanced.tif"
                if self.enhance_image(self.tif_3b, enhanced_3b, '3B'):
                    self.tif_3b_enhanced = enhanced_3b
                    self.log(f"✓ Enhanced 3B: {enhanced_3b.name}", 'SUCCESS')
            else:
                self.log("Enhancement skipped (enhance=none)", 'INFO')
            
            # Create calibrated directory if needed
            # This was already created as absolute path at start of method
            self.log(f"Copying XML files to: {self.calibrated_dir}", 'INFO')
            
            # Copy XML files to calibrated directory
            # Need to identify which is 3N and which is 3B
            # IMPORTANT: s2p requires RPC_ prefix on filenames
            found_3n = False
            found_3b = False
            
            for xml in xml_files:
                xml_name_lower = xml.name.lower()
                
                # Determine output filename - RPC_ files already have correct prefix
                if xml.name.startswith('RPC_'):
                    # Already has RPC_ prefix - use as-is
                    output_name = xml.name
                else:
                    # Add RPC_ prefix for s2p compatibility
                    output_name = f"RPC_{xml.name}"
                
                # Check if this is a 3N file
                if '_3n' in xml_name_lower or '3n.' in xml_name_lower:
                    dest_path = self.calibrated_dir / output_name
                    shutil.copy2(xml, dest_path)
                    self.rpc_3n = dest_path
                    self.log(f"✓ Copied RPC 3N: {output_name}", 'SUCCESS')
                    found_3n = True
                
                # Check if this is a 3B file
                elif '_3b' in xml_name_lower or '3b.' in xml_name_lower:
                    dest_path = self.calibrated_dir / output_name
                    shutil.copy2(xml, dest_path)
                    self.rpc_3b = dest_path
                    self.log(f"✓ Copied RPC 3B: {output_name}", 'SUCCESS')
                    found_3b = True
                
                else:
                    self.log(f"⚠ Unclear XML file type: {xml.name}", 'WARNING')
            
            if not found_3n or not found_3b:
                self.log("", 'ERROR')
                self.log("Could not identify both 3N and 3B RPC files", 'ERROR')
                self.log(f"Found 3N: {found_3n}, Found 3B: {found_3b}", 'ERROR')
                self.log("XML files found:", 'INFO')
                for xml in xml_files:
                    self.log(f"  - {xml.name}", 'INFO')
                return False
            
            return True
        finally:
            os.chdir(original_cwd)
    
    def create_s2p_config(self):
        self.log("STEP 3: Creating s2p Configuration", 'SECTION')
        
        # Find actual files in calibrated directory
        self.log(f"Searching for files in: {self.calibrated_dir}", 'INFO')
        
        # Find enhanced TIF files (prefer enhanced over micmac)
        tif_3n_candidates = (list(self.calibrated_dir.glob("*3N*enhanced.tif")) or 
                            list(self.calibrated_dir.glob("*3N*micmac.tif")))
        tif_3b_candidates = (list(self.calibrated_dir.glob("*3B*enhanced.tif")) or 
                            list(self.calibrated_dir.glob("*3B*micmac.tif")))
        
        # Find RPC XML files - s2p requires RPC_ prefix
        xml_3n_candidates = (list(self.calibrated_dir.glob("RPC_*3N*.xml")) or
                            list(self.calibrated_dir.glob("*3N*.xml")))
        xml_3b_candidates = (list(self.calibrated_dir.glob("RPC_*3B*.xml")) or
                            list(self.calibrated_dir.glob("*3B*.xml")))
        
        # Validate we have all required files
        if not tif_3n_candidates:
            self.log("Could not find 3N TIF file in calibrated directory", 'ERROR')
            return None
        if not tif_3b_candidates:
            self.log("Could not find 3B TIF file in calibrated directory", 'ERROR')
            return None
        if not xml_3n_candidates:
            self.log("Could not find 3N XML RPC file in calibrated directory", 'ERROR')
            return None
        if not xml_3b_candidates:
            self.log("Could not find 3B XML RPC file in calibrated directory", 'ERROR')
            return None
        
        # Use first match for each and get absolute paths
        tif_3n = tif_3n_candidates[0].absolute()
        tif_3b = tif_3b_candidates[0].absolute()
        xml_3n = xml_3n_candidates[0].absolute()
        xml_3b = xml_3b_candidates[0].absolute()
        
        self.log(f"Using files:", 'SUCCESS')
        self.log(f"  3N Image: {tif_3n.name}", 'INFO')
        self.log(f"  3N RPC: {xml_3n.name}", 'INFO')
        self.log(f"  3B Image: {tif_3b.name}", 'INFO')
        self.log(f"  3B RPC: {xml_3b.name}", 'INFO')
        
        # Start with either custom config or quality preset
        if self.config_file:
            # Load custom config and replace only the file paths
            self.log(f"Loading custom config: {self.config_file}", 'INFO')
            with open(self.config_file) as f:
                config = json.load(f)
            self.log("Using custom s2p parameters from config file", 'SUCCESS')
        else:
            # Use quality preset
            preset = self.quality_presets[self.quality]
            config = {
                "full_img": True,
                "dsm_resolution": self.resolution,
                "clean_tmp": True,
                "clean_intermediate": False
            }
            # Add quality preset settings
            config.update({k: v for k, v in preset.items() if k != 'enhance'})
            self.log(f"Using quality preset: {self.quality}", 'INFO')
        
        # ALWAYS replace these fields with correct paths
        config["out_dir"] = str((self.output_dir / "s2p_output").absolute())
        config["images"] = [
            {"img": str(tif_3n), "rpc": str(xml_3n)},
            {"img": str(tif_3b), "rpc": str(xml_3b)}
        ]
        
        config_path = self.calibrated_dir / 's2p_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, indent=2, fp=f)
        
        self.log(f"Config created: {config_path.name}", 'SUCCESS')
        self.log(f"  Output directory: {config['out_dir']}", 'INFO')
        if 'matching_algorithm' in config:
            self.log(f"  Matching algorithm: {config['matching_algorithm']}", 'INFO')
        if 'tile_size' in config:
            self.log(f"  Tile size: {config['tile_size']}", 'INFO')
        return config_path
    
    def run_s2p(self, config_path):
        self.log("STEP 4: Running s2p Processing", 'SECTION')
        
        if not shutil.which('s2p'):
            self.log("s2p not found", 'ERROR')
            return False
        
        cmd = ['s2p', str(config_path)]
        return self.run_command(cmd, "Running s2p")
    
    def run(self):
        print("="*70)
        print("ASTER Complete Workflow - AUTO UTM DETECTION")
        print("="*70)
        print(f"Input: {self.hdf_file}")
        print(f"Scene: {self.scene_name}")
        print(f"UTM: {'Provided: ' + self.utm_zone if self.utm_zone_provided else 'Auto-detect'}")
        print(f"Enhancement: {self.enhance}")
        print(f"Quality: {self.quality}")
        print("="*70)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simplified workflow:
        # 1. Extract HDF
        # 2. Generate RPC files (includes MicMac calibration + optional enhancement)
        # 3. Create s2p config
        # 4. Run s2p
        steps = [
            self.extract_hdf,
            self.generate_rpc_files,
        ]
        
        for step in steps:
            if not step():
                return False
        
        config_path = self.create_s2p_config()
        if not config_path or not self.run_s2p(config_path):
            return False
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETED")
        print("="*70)
        print(f"UTM Zone: {self.utm_zone}")
        print(f"DEM: {self.output_dir}/s2p_output/dsm.tif")
        print("="*70)
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description='ASTER Complete Workflow - Uses MicMac calibration + optional s2p GPU processing',
        epilog="""
Examples:
    # Basic usage (auto UTM, MicMac calibrated images, no enhancement)
    python3 aster_complete_workflow.py scene.hdf
    
    # With custom s2p parameters
    python3 aster_complete_workflow.py scene.hdf --config my_s2p_params.json
    
    # With enhancement for snow/ice
    python3 aster_complete_workflow.py scene.hdf --enhance always
    
    # Manual UTM zone
    python3 aster_complete_workflow.py scene.hdf --utm-zone "11 +north"
        """
    )
    
    parser.add_argument('hdf_file', help='ASTER L1A HDF file')
    parser.add_argument('--utm-zone', help='UTM zone (auto-detected if not provided)')
    parser.add_argument('--quality', choices=['fast', 'balanced', 'best'], default='balanced',
                       help='s2p quality preset (default: balanced)')
    parser.add_argument('--enhance', choices=['none', 'auto', 'always'], default='none',
                       help='Image enhancement: none=use MicMac calibrated, auto=enhance if snow, always=always enhance (default: none)')
    parser.add_argument('--config', help='Custom s2p config JSON file (paths will be auto-updated)')
    parser.add_argument('--output-dir', help='Output directory (default: scene name)')
    parser.add_argument('--resolution', type=int, default=30, help='DEM resolution in meters (default: 30)')
    parser.add_argument('--skip-extraction', action='store_true', help='Skip HDF extraction if already done')
    parser.add_argument('--skip-enhancement', action='store_true', help='Deprecated: use --enhance=none instead')
    
    args = parser.parse_args()
    
    workflow = ASTERWorkflow(
        hdf_file=args.hdf_file,
        utm_zone=args.utm_zone,
        quality=args.quality,
        enhance=args.enhance,
        config=args.config,
        output_dir=args.output_dir,
        resolution=args.resolution,
        skip_extraction=args.skip_extraction,
        skip_enhancement=args.skip_enhancement
    )
    
    sys.exit(0 if workflow.run() else 1)


if __name__ == '__main__':
    main()

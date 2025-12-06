#!/usr/bin/env python3
"""
MMASTER Workflow for Unzipped ASTER L1A Data
This script executes the MMASTER workflow starting from an unzipped ASTER L1A directory.
MicMac will generate the XML files from the ASTER data files.

Requirements:
- MicMac (IncludeALGLIB branch) installed and in PATH
- GDAL installed
- Python 3.6+

Usage:
    python mmaster_workflow.py -d <aster_directory> -z <utm_zone> [options]
"""

import os
import sys
import subprocess
import argparse
import glob
import shutil
from pathlib import Path
# No time-based delays are used


class MMASTERWorkflow:
    def __init__(self, aster_dir, utm_zone, **kwargs):
        """
        Initialize MMASTER workflow for unzipped ASTER L1A data.
        
        Args:
            aster_dir: Path to unzipped ASTER L1A directory
            utm_zone: UTM zone (e.g., "4 +north")
            **kwargs: Additional options (zoomf, resterr, corthr, etc.)
        """
        # Resolve the absolute path to the data directory
        self.aster_dir = Path(aster_dir).resolve()
        self.utm_zone = utm_zone
        
        # Extract scene name from directory name
        self.scene_name = self.aster_dir.name
        
        # Default parameters
        self.zoomf = kwargs.get('zoomf', 1)
        self.resterr = kwargs.get('resterr', 30)
        self.corthr = kwargs.get('corthr', 0.7)
        self.szw = kwargs.get('szw', 5)
        self.water_mask = kwargs.get('water_mask', False)
        self.do_ply = kwargs.get('do_ply', False)
        self.do_angle = kwargs.get('do_angle', False)
        self.nocor_dem = kwargs.get('nocor_dem', False)
        self.fit_version = kwargs.get('fit_version', 2)
        self.no_fill_holes = kwargs.get('no_fill_holes', False)
        
        # File suffixes (will be created by MicMac)
        self.n_suffix = "_3N"
        self.b_suffix = "_3B"
        self.n_xml = "_3N.xml"
        self.b_xml = "_3B.xml"
        self.n_tif = "_3N.tif"
        self.b_tif = "_3B.tif"
        
        # Set projection
        self.proj = f"+proj=utm +zone={self.utm_zone} +datum=WGS84 +units=m +no_defs"
        
    def check_dependencies(self):
        """Check if required tools are installed."""
        try:
            subprocess.run(['mm3d', '-help'], 
                          capture_output=True, check=True)
            print("✓ MicMac found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ MicMac not found. Please install MicMac (IncludeALGLIB branch)")
            return False
            
        try:
            subprocess.run(['gdal_translate', '--version'], 
                          capture_output=True, check=True)
            print("✓ GDAL found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("✗ GDAL not found. Please install GDAL")
            return False
            
        return True
    
    def check_aster_files(self):
        """Check for ASTER L1A data files."""
        if not self.aster_dir.exists():
            print(f"✗ ASTER directory not found: {self.aster_dir}")
            return False
        
        # List all files in the directory
        all_files = list(self.aster_dir.glob("*"))
        print(f"\n✓ Found {len(all_files)} files in {self.aster_dir.name}")
        
        # Show file types
        extensions = {}
        for f in all_files:
            if f.is_file():
                ext = f.suffix.lower() if f.suffix else "no extension"
                extensions[ext] = extensions.get(ext, 0) + 1
        
        print("  File types:")
        for ext, count in sorted(extensions.items()):
            print(f"    {ext}: {count}")
        
        # Check for common ASTER file patterns
        hdf_files = list(self.aster_dir.glob("*.hdf"))
        tif_files = list(self.aster_dir.glob("*ImageData*.tif")) + \
                    list(self.aster_dir.glob("*VNIR*.tif")) + \
                    list(self.aster_dir.glob("*.tif"))
        dat_files = list(self.aster_dir.glob("*.dat"))
        
        if hdf_files:
            print(f"  Found {len(hdf_files)} HDF file(s)")
        if tif_files:
            print(f"  Found {len(tif_files)} TIF file(s)")
        if dat_files:
            print(f"  Found {len(dat_files)} DAT file(s)")
        
        # We need at least some data files
        if not (hdf_files or tif_files or dat_files or all_files):
            print("✗ No recognizable ASTER data files found")
            return False
        
        return True
    
    def list_directory_contents(self):
        """List and analyze directory contents."""
        print("\n=== Directory Contents ===")
        files = sorted(Path.cwd().glob("*"))
        for f in files[:20]:  # Show first 20 files
            if f.is_file():
                size_mb = f.stat().st_size / (1024*1024)
                print(f"  {f.name} ({size_mb:.2f} MB)")
        if len(files) > 20:
            print(f"  ... and {len(files)-20} more files")
    
    def run_command(self, cmd, cwd=None, show_output=True):
        """
        Execute a shell command, streaming output to prevent pipe buffer deadlock.
        """
        cwd = cwd or Path.cwd() 
        cmd_str = ' '.join(cmd) if isinstance(cmd, list) else cmd
        print(f"\n> {cmd_str}")
        
        try:
            # Set stdout=None to allow output to stream directly to sys.stdout,
            # preventing the hang. We capture stderr for clean error messages.
            subprocess.run(
                cmd,
                cwd=cwd,
                shell=isinstance(cmd, str),
                check=True,
                stdout=None, 
                stderr=subprocess.PIPE,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Command failed with return code {e.returncode}")
            if e.stderr:
                print(f"Error output:\n{e.stderr}")
            return False

    def move_micmac_output(self):
        """
        Checks the parent directory for MicMac generated XML/TIF files (which 
        ASTERGT2MM often places there) and moves them into the current CWD 
        (the granule directory).
        """
        parent_dir = Path.cwd().parent
        scene_base_name = self.scene_name
        
        # Patterns for the files MicMac typically creates in the parent dir
        file_patterns = [
            f'{scene_base_name}*.tif',
            f'{scene_base_name}*.xml',
            f'FalseColor_{scene_base_name}*.tif',
            f'FalseColor_{scene_base_name}*.xml',
        ]
        
        moved_files = []
        for pattern in file_patterns:
            for f in parent_dir.glob(pattern):
                if f.is_file():
                    target_path = Path.cwd() / f.name
                    # Check if the file name is already present in CWD to avoid errors
                    if not target_path.exists():
                        try:
                            shutil.move(str(f), str(target_path))
                            moved_files.append(f.name)
                        except Exception as e:
                            print(f"⚠ Warning: Could not move {f.name}: {e}")
                            
        if moved_files:
            print(f"✓ Moved {len(moved_files)} MicMac output files from parent directory to {Path.cwd().name}:")
            for name in moved_files:
                print(f"    - {name}")
            return True
        else:
            print("⚠ Warning: No expected XML/TIF files were found in the parent directory to move.")
            return False

    def convert_aster_to_micmac(self):
        """
        Convert ASTER data to MicMac format.
        This step creates the XML and TIF files needed for processing.
        """
        print("\n=== Converting ASTER to MicMac format ===")
        print(f"Working directory: {Path.cwd()}") # Use CWD for output
        
        # Check if XML files already exist in the CWD
        existing_xml = list(Path.cwd().glob("*_3N.xml")) + list(Path.cwd().glob("*_3B.xml"))
        if len(existing_xml) >= 2:
            print("✓ XML files already exist, skipping ASTER conversion")
            return True
        
        # Use the full scene name for MicMac conversion
        scene_id_to_use = self.scene_name
        
        # Try 1: Full scene name (matches the successful manual command)
        cmd = ['mm3d', 'SateLib', 'ASTERGT2MM', scene_id_to_use]
        
        if self.run_command(cmd):
            print("✓ ASTER conversion successful")
            # --- NEW: Move files from parent directory into CWD ---
            self.move_micmac_output()
            # ----------------------------------------------------
            return True
        
        # Try 2: Wildcard pattern
        print("\nTrying with wildcard pattern...")
        cmd = ['mm3d', 'SateLib', 'ASTERGT2MM', 'AST_L1A*']
        if self.run_command(cmd):
            print("✓ ASTER conversion successful")
            # --- NEW: Move files from parent directory into CWD ---
            self.move_micmac_output()
            # ----------------------------------------------------
            return True
        
        print("\n✗ All ASTER conversion attempts failed")
        print("Please ensure all ASTER L1A data files are present in the directory")
        return False
    
    def verify_micmac_output(self):
        """Verify that MicMac created the expected XML and TIF files, now located in CWD."""
        print("\n=== Verifying MicMac output === (Single Attempt)")

        # Search in the current working directory (self.aster_dir)
        cwd = Path.cwd()
        
        # Search for XML files with exact patterns
        xml_files = list(cwd.glob("*_3N.xml")) + \
                    list(cwd.glob("*_3B.xml")) + \
                    list(cwd.glob("*.xml"))
        
        xml_files = list(set(xml_files))
        tif_files = list(cwd.glob("*.tif"))
        
        # Look for the required 3N and 3B candidates (case-insensitive and multiple pattern checks)
        n_xml_candidates = [f for f in xml_files if f.name.endswith('_3N.xml') or '_3n.xml' in f.name.lower()]
        b_xml_candidates = [f for f in xml_files if f.name.endswith('_3B.xml') or '_3b.xml' in f.name.lower()]
        
        print(f"\nXML files found: {len(xml_files)}")
        for xml in xml_files:
            print(f"  {xml.name}")

        print(f"TIF files found: {len(tif_files)}")
        for tif in tif_files[:10]:
            print(f"  {tif.name}")
        
        # Final check
        if n_xml_candidates and b_xml_candidates:
            # Update the file names based on actual files found
            self.n_xml_file = n_xml_candidates[0].name
            self.b_xml_file = b_xml_candidates[0].name

            # Find corresponding TIF files 
            n_tif_candidates = [f for f in tif_files if f.name.endswith('_3N.tif') or '3N.tif' in f.name or '3n.tif' in f.name.lower()]
            b_tif_candidates = [f for f in tif_files if f.name.endswith('_3B.tif') or '3B.tif' in f.name or '3b.tif' in f.name.lower()]

            if n_tif_candidates and b_tif_candidates:
                self.n_tif_file = n_tif_candidates[0].name
                self.b_tif_file = b_tif_candidates[0].name

            print(f"\n✓ Found files for nadir and backward bands:")
            print(f"    Nadir (3N) XML: {self.n_xml_file}")
            print(f"    Backward (3B) XML: {self.b_xml_file}")
            if hasattr(self, 'n_tif_file'):
                print(f"    Nadir (3N) TIF: {self.n_tif_file}")
                print(f"    Backward (3B) TIF: {self.b_tif_file}")
            return True
        else:
            print(f"\n✗ Could not find 3N and 3B XML files.")
            return False
    
    def generate_grids(self):
        """Generate grid files for ASTER bands."""
        print("\n=== Generating grids ===")
        
        if not hasattr(self, 'n_xml_file') or not hasattr(self, 'b_xml_file'):
            print("✗ XML file names not set. Cannot generate grids.")
            return False
        
        # Use actual XML file names which are in the CWD
        commands = [
            ['mm3d', 'SateLib', 'Aster2Grid', self.b_xml_file, '20', 
             self.proj, 'HMin=-500', 'HMax=9000', 'expDIMAP=1', 'expGrid=1'],
            ['mm3d', 'SateLib', 'Aster2Grid', self.n_xml_file, '20', 
             self.proj, 'HMin=-500', 'HMax=9000', 'expDIMAP=1', 'expGrid=1'],
        ]
        
        # Also try to generate grid for false color composite if it exists
        false_color_xmls = list(Path.cwd().glob('*FalseColor*.xml'))
        if false_color_xmls:
            self.false_color_xml = false_color_xmls[0].name
            commands.append([
                'mm3d', 'SateLib', 'Aster2Grid', self.false_color_xml, '20', 
                self.proj, 'HMin=-500', 'HMax=9000', 'expDIMAP=1', 'expGrid=1'
            ])
        
        for cmd in commands:
            if not self.run_command(cmd):
                print(f"⚠ Grid generation failed for: {cmd[3]}")
                # Continue anyway - MicMac might still work
        
        return True
    
    def run_correlation_mini(self):
        """Run initial correlation at reduced resolution."""
        print("\n=== Running initial correlation (Mini) ===")
        
        if not hasattr(self, 'n_tif_file') or not hasattr(self, 'b_tif_file'):
            print("✗ TIF file names not set. Cannot run correlation.")
            return False
        
        # Determine the shared prefix dynamically
        base_name = self.n_tif_file.replace('_3N.tif', '').replace('_3n.tif', '')
        pattern = f'{base_name}_3[NB].tif'
        imMNT_pattern = f'{base_name}_3[NB].tif'
        
        print(f"Using pattern: {pattern}")
        
        cmd = [
            'mm3d', 'Malt', 'Ortho', pattern,
            'GRIBin', f'ImMNT={imMNT_pattern}',
            'MOri=GRID', 'ZMoy=2500', 'ZInc=2500', 'ZoomF=8', 'ZoomI=32',
            'ResolTerrain=30', 'NbVI=2', 'EZA=1', 'Regul=0.1',
            f'DefCor={self.corthr}', 'DoOrtho=0', 'DirMEC=MEC-Mini'
        ]
        
        return self.run_command(cmd)
    
    def estimate_dem_stats(self):
        """Estimate DEM statistics from initial correlation."""
        print("\n=== Estimating DEM statistics ===")
        
        # Read initial DEM from the CWD
        dem_file = Path.cwd() / "MEC-Mini" / "Z_Num9_DeZoom8_STD-MALT.tif"
        if not dem_file.exists():
            print(f"⚠ Initial DEM not found: {dem_file.name}")
            print("  Using default values: mean=2500, inc=3000")
            return 2500, 3000
        
        # Use gdalinfo to get statistics
        try:
            result = subprocess.run(
                ['gdalinfo', '-stats', str(dem_file)],
                capture_output=True,
                text=True,
                check=True
            )
            
            mean = None
            for line in result.stdout.split('\n'):
                if 'STATISTICS_MEAN' in line:
                    mean = float(line.split('=')[1])
                    break
            
            if mean is None:
                print("⚠ Could not extract DEM mean, using default 2500")
                mean = 2500
            
            inc = 3000  # Default increment
            print(f"  DEM Mean: {mean:.2f}, Increment: {inc}")
            return mean, inc
            
        except subprocess.CalledProcessError:
            print("⚠ Could not get DEM stats, using defaults")
            return 2500, 3000
    
    def run_full_correlation(self, mean, inc):
        """Run full resolution correlation."""
        print("\n=== Running full correlation ===")
        
        # Generate refined grids
        nblvl = 20
        commands = [
            ['mm3d', 'SateLib', 'Aster2Grid', self.b_xml_file, str(nblvl),
             self.proj, f'HMin={mean-inc}', f'HMax={mean+inc}', 'expDIMAP=1', 'expGrid=1'],
            ['mm3d', 'SateLib', 'Aster2Grid', self.n_xml_file, str(nblvl),
             self.proj, f'HMin={mean-inc}', f'HMax={mean+inc}', 'expDIMAP=1', 'expGrid=1'],
        ]
        
        # Check if false color exists
        if hasattr(self, 'false_color_xml'):
            commands.append([
                'mm3d', 'SateLib', 'Aster2Grid', self.false_color_xml, str(nblvl),
                self.proj, f'HMin={mean-inc}', f'HMax={mean+inc}', 'expDIMAP=1', 'expGrid=1'
            ])
        
        for cmd in commands:
            self.run_command(cmd)
        
        # Test orientation
        print("\n--- Testing orientation ---")
        cmd = [
            'mm3d', 'MMTestOrient', self.b_tif_file, self.n_tif_file,
            'GRIBin', 'PB=1', 'MOri=GRID', 'ZoomF=1',
            f'ZInc={inc}', f'ZMoy={mean}'
        ]
        self.run_command(cmd)
        
        # Final correlation
        print("\n--- Running final correlation ---")
        
        # Use the same pattern as Mini correlation
        base_name = self.n_tif_file.replace('_3N.tif', '').replace('_3n.tif', '')
        pattern = f'{base_name}_3[NB].tif'
        imMNT_pattern = f'{base_name}_3[NB].tif'

        
        cmd = [
            'mm3d', 'Malt', 'Ortho', pattern,
            'GRIBin', f'ImMNT={imMNT_pattern}',
            'MOri=GRID',
            f'ZInc={inc}', f'ZMoy={mean}', f'ZoomF={self.zoomf}', 'ZoomI=32',
            f'ResolTerrain={self.resterr}', 'NbVI=2', 'EZA=1', 'DefCor=0',
            'Regul=0.1', f'SzW={self.szw}'
        ]
        
        # Add false color orthophoto if it exists
        if hasattr(self, 'false_color_xml'):
            # Look for the TIF file corresponding to the FalseColor XML
            false_color_tifs = list(Path.cwd().glob('*FalseColor*.tif'))
            if false_color_tifs:
                cmd.append(f'ImOrtho={false_color_tifs[0].name}')
                cmd.append('ResolOrtho=2')
        
        return self.run_command(cmd)
    
    def apply_parallax_correction(self):
        """Apply parallax correction to ASTER imagery."""
        print("\n=== Applying parallax correction ===")
        
        parallax_file = Path.cwd() / "GeoI-Px" / "Px2_Num16_DeZoom1_Geom-Im.tif"
        
        if not parallax_file.exists():
            print(f"⚠ Parallax file not found: {parallax_file.name}")
            print("  Skipping parallax correction")
            return True
        
        if not hasattr(self, 'b_tif_file'):
            print("⚠ Backward TIF file name not set. Skipping parallax correction.")
            return True
        
        cmd = [
            'mm3d', 'SateLib', 'ApplyParallaxCor',
            self.b_tif_file, 'GeoI-Px/Px2_Num16_DeZoom1_Geom-Im.tif', # Use relative path from CWD
            f'FitASTER={self.fit_version}', 'ExportFitASTER=1',
            f'ASTERSceneName={self.scene_name}'
        ]
        
        return self.run_command(cmd)
    
    def final_processing(self):
        """Generate final DEM and orthophoto products."""
        print("\n=== Final processing ===")
        
        # Generate angle maps if requested
        if self.do_angle:
            if hasattr(self, 'b_xml_file') and hasattr(self, 'n_xml_file'):
                cmd = [
                    'mm3d', 'SateLib', 'GenTracksFromXML',
                    self.b_xml_file, self.n_xml_file
                ]
                self.run_command(cmd)
        
        # Reproject final DEM
        # Look for files in CWD (self.aster_dir)
        dem_file = Path.cwd() / "MEC-Malt" / "Z_Num9_DeZoom1_STD-MALT.tif"
        output_dem = Path.cwd() / f'{self.scene_name}_DEM.tif'
        
        if dem_file.exists():
            # Choose resampling method based on no_fill_holes flag
            if self.no_fill_holes:
                # Use nearest neighbor to avoid interpolation/extrapolation
                resample_method = 'near'
                print("✓ Using nearest neighbor resampling (no hole filling)")
            else:
                # Use cubic spline (original behavior)
                resample_method = 'cubicspline'
                print("✓ Using cubic spline resampling (default)")
            
            cmd = [
                'gdal_translate', '-tr', str(self.resterr), str(self.resterr),
                '-r', resample_method, '-a_srs', self.proj, 
                '-a_nodata', '-9999',  # Set nodata value explicitly
                '-co', 'COMPRESS=LZW',
                '-co', 'TILED=YES', 
                '-co', 'BIGTIFF=YES',
            ]
            
            # Only add NODATA to output if we're preserving holes
            if self.no_fill_holes:
                cmd.extend(['-co', 'NODATA=-9999'])
            
            cmd.extend([str(dem_file), str(output_dem)])
            
            if self.run_command(cmd):
                print(f"✓ DEM created: {output_dem.name}")
        else:
            print(f"⚠ Final DEM not found: {dem_file.name}")
        
        # Reproject orthophoto
        ortho_file = Path.cwd() / "Ortho-MEC-Malt" / "Orthophotomosaic.tif"
        output_ortho = Path.cwd() / f'{self.scene_name}_Ortho.tif'
        
        if ortho_file.exists():
            cmd = [
                'gdal_translate', '-tr', '15', '15', '-r', 'bilinear',
                '-a_srs', self.proj, '-co', 'COMPRESS=LZW',
                '-co', 'TILED=YES', '-co', 'BIGTIFF=YES', '-co', 'PHOTOMETRIC=RGB',
                str(ortho_file), str(output_ortho)
            ]
            if self.run_command(cmd):
                print(f"✓ Orthophoto created: {output_ortho.name}")
        else:
            print(f"⚠ Orthophoto not found: {ortho_file.name}")
        
        return True
    
    def run(self):
        """Execute the complete MMASTER workflow."""
        print("="*70)
        print("MMASTER Workflow for ASTER L1A Data")
        print("="*70)
        print(f"ASTER directory: {self.aster_dir}")
        print(f"Scene name: {self.scene_name}")
        print(f"UTM Zone: {self.utm_zone}")
        print(f"Resolution: {self.resterr}m")
        print(f"Fill holes: {'No (nearest neighbor)' if self.no_fill_holes else 'Yes (cubic spline)'}")
        print("="*70)
        
        # Store original CWD
        original_cwd = Path.cwd()
        
        # FIX: Change directory to the granule directory (self.aster_dir)
        print(f"Changing current working directory from {original_cwd.name} to {self.aster_dir.name}")
        try:
            os.chdir(self.aster_dir)
            self.aster_dir = Path.cwd()
        except OSError as e:
            print(f"✗ Failed to change directory: {e}")
            return False
        
        # Check dependencies
        if not self.check_dependencies():
            os.chdir(original_cwd)
            return False
        
        # Check ASTER files
        if not self.check_aster_files():
            pass
        
        # Show directory contents (now inside the granule directory)
        self.list_directory_contents()
        
        # Convert ASTER to MicMac format (creates XML and TIF files, then moves them to CWD)
        if not self.convert_aster_to_micmac():
            print("\n" + "="*70)
            print("WORKFLOW FAILED: Could not convert ASTER data to MicMac format")
            print("="*70)
            os.chdir(original_cwd)
            return False
        
        # Verify MicMac created the necessary files (checking in the current CWD)
        if not self.verify_micmac_output():
            print("\n⚠ Warning: Some expected files may be missing, but continuing...")
        
        # Generate grids
        if not self.generate_grids():
            print("\n⚠ Warning: Grid generation had issues, but continuing...")
        
        # Run initial correlation
        print("\nStarting correlation process (this may take a while)...")
        if not self.run_correlation_mini():
            print("\n✗ Initial correlation failed")
            os.chdir(original_cwd)
            return False
        
        # Estimate DEM statistics
        mean, inc = self.estimate_dem_stats()
        
        # Run full correlation
        if not self.run_full_correlation(mean, inc):
            print("\n✗ Full correlation failed")
            os.chdir(original_cwd)
            return False
        
        # Apply parallax correction
        self.apply_parallax_correction()
        
        # Final processing
        self.final_processing()
        
        # Restore original working directory
        os.chdir(original_cwd)
        
        print("\n" + "="*70)
        print("WORKFLOW COMPLETED")
        print("="*70)
        print(f"\nOutput files are in: {self.aster_dir}")
        print(f"  - DEM: {self.scene_name}_DEM.tif")
        print(f"  - Orthophoto: {self.scene_name}_Ortho.tif")
        print("="*70)
        return True


def main():
    parser = argparse.ArgumentParser(
        description='MMASTER workflow for ASTER L1A data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python mmaster_workflow.py -d AST_L1A_00408242025175328_20251011031216 -z "11 +north"
    python mmaster_workflow.py -d /path/to/aster_data -z "4 +north" -t 30 -f 1 -a
    python mmaster_workflow.py -d /path/to/aster_data -z "4 +north" --no-fill-holes
        """
    )
    
    parser.add_argument('-d', '--directory', required=True,
                       help='Path to unzipped ASTER L1A directory')
    parser.add_argument('-z', '--zone', required=True,
                       help='UTM zone (e.g., "4 +north" or "11 +south")')
    parser.add_argument('-f', '--zoomf', type=int, default=1,
                       help='Zoom factor (default: 1)')
    parser.add_argument('-t', '--resterr', type=int, default=30,
                       help='Resolution terrain in meters (default: 30)')
    parser.add_argument('-c', '--corthr', type=float, default=0.7,
                       help='Correlation threshold (default: 0.7)')
    parser.add_argument('-q', '--szw', type=int, default=5,
                       help='Correlation window size parameter (default: 5)')
    parser.add_argument('-w', '--water_mask', action='store_true',
                       help='Use water mask')
    parser.add_argument('-a', '--angle', action='store_true',
                       help='Generate track angle maps')
    parser.add_argument('-n', '--nocor', action='store_true',
                       help='Skip no-correlation DEM')
    parser.add_argument('-i', '--fitversion', type=int, default=2,
                       choices=[1, 2],
                       help='Fit version for parallax correction (default: 2)')
    parser.add_argument('--no-fill-holes', action='store_true',
                       help='Do not fill holes or interpolate beyond actual data in final DEM. '
                            'Uses nearest neighbor resampling instead of cubic spline.')
    
    args = parser.parse_args()
    
    # Create workflow instance
    workflow = MMASTERWorkflow(
        aster_dir=args.directory,
        utm_zone=args.zone,
        zoomf=args.zoomf,
        resterr=args.resterr,
        corthr=args.corthr,
        szw=args.szw,
        water_mask=args.water_mask,
        do_angle=args.angle,
        nocor_dem=args.nocor,
        fit_version=args.fitversion,
        no_fill_holes=args.no_fill_holes
    )
    
    # Run workflow
    success = workflow.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

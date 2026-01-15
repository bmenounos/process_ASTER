#!/usr/bin/env python3
"""
ASTER Batch Processor
Process multiple ASTER HDF granules and collect final DEMs with proper naming.

Usage:
    python3 aster_batch_processor.py /path/to/hdf_files/*.hdf [options]
    python3 aster_batch_processor.py --input-dir /path/to/hdf_files [options]
    python3 aster_batch_processor.py --file-list granules.txt [options]

The script will:
1. Process each HDF granule in its own directory (named by granule ID)
2. Extract date/time from granule name (e.g., AST_L1A_00408152015201917)
3. Rename final DEM to include date_time string
4. Move all final DEMs to a consolidated output directory
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import re
from datetime import datetime
import shutil


class ASTERBatchProcessor:
    """Batch processor for multiple ASTER granules"""
    
    def __init__(self, **kwargs):
        self.work_dir = Path(kwargs.get('work_dir', 'aster_processing')).resolve()
        self.final_dem_dir = Path(kwargs.get('output_dir', 'final_dems')).resolve()
        self.quality = kwargs.get('quality', 'balanced')
        self.enhance = kwargs.get('enhance', 'none')
        self.resolution = kwargs.get('resolution', 30)
        self.custom_config = kwargs.get('config')
        self.utm_zone = kwargs.get('utm_zone')
        self.keep_intermediates = kwargs.get('keep_intermediates', False)
        self.workflow_script = kwargs.get('workflow_script', 'aster_complete_workflow.py')
        
        # Track processing status
        self.processed = []
        self.failed = []
        self.skipped = []
        
    def extract_datetime_from_granule(self, granule_name):
        """
        Extract date and time from ASTER granule name.
        
        Format: AST_L1A_00408152015201917
                         MMDDYYYYHHMMSS
        
        Returns: datetime_string (e.g., "20150815_201917") or None
        """
        # Match pattern: digits after last underscore or in position
        # ASTER L1A format: AST_L1A_00MMDDYYYYHHMMSS
        match = re.search(r'AST_L1A_(\d{14,16})', granule_name)
        
        if match:
            digits = match.group(1)
            
            # Handle different digit counts
            if len(digits) == 14:
                # MMDDYYYYHHMMSS
                mm = digits[0:2]
                dd = digits[2:4]
                yyyy = digits[4:8]
                hh = digits[8:10]
                mi = digits[10:12]
                ss = digits[12:14]
            elif len(digits) == 16:
                # PPMMDDYYYYHHMMSS (with path/row prefix)
                mm = digits[2:4]
                dd = digits[4:6]
                yyyy = digits[6:10]
                hh = digits[10:12]
                mi = digits[12:14]
                ss = digits[14:16]
            else:
                print(f"  ⚠ Warning: Unexpected digit count ({len(digits)}) in {granule_name}")
                return None
            
            try:
                # Validate date
                dt = datetime(int(yyyy), int(mm), int(dd), int(hh), int(mi), int(ss))
                return f"{yyyy}{mm}{dd}_{hh}{mi}{ss}"
            except ValueError as e:
                print(f"  ⚠ Warning: Invalid date in {granule_name}: {e}")
                return None
        
        print(f"  ⚠ Warning: Could not extract date/time from {granule_name}")
        return None
    
    def find_final_dem(self, processing_dir):
        """
        Find the final (non-filtered) DEM in the processing directory.
        Looks for: s2p_output/dsm.tif
        """
        s2p_output = processing_dir / 's2p_output'
        
        if not s2p_output.exists():
            print(f"  ⚠ Warning: s2p_output directory not found in {processing_dir}")
            return None
        
        # Look for dsm.tif (the main DEM output from s2p)
        dem_path = s2p_output / 'dsm.tif'
        
        if dem_path.exists():
            return dem_path
        
        # Alternative: look for any TIF that looks like a DEM
        tif_files = list(s2p_output.glob('*.tif'))
        dem_candidates = [f for f in tif_files if 'dsm' in f.name.lower() or 'dem' in f.name.lower()]
        
        if dem_candidates:
            return dem_candidates[0]
        
        print(f"  ⚠ Warning: No DEM file found in {s2p_output}")
        return None
    
    def process_granule(self, hdf_file):
        """Process a single ASTER granule"""
        hdf_path = Path(hdf_file).resolve()
        granule_name = hdf_path.stem
        
        print(f"\n{'='*70}")
        print(f"Processing: {granule_name}")
        print(f"{'='*70}")
        
        # Create granule-specific directory
        granule_dir = self.work_dir / granule_name
        granule_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"  Working directory: {granule_dir}")
        
        # Build command for workflow script
        cmd = [
            'python3',
            str(self.workflow_script),
            str(hdf_path),
            '--output-dir', str(granule_dir),
            '--quality', self.quality,
            '--enhance', self.enhance,
            '--resolution', str(self.resolution)
        ]
        
        if self.utm_zone:
            cmd.extend(['--utm-zone', self.utm_zone])
        
        if self.custom_config:
            cmd.extend(['--config', str(self.custom_config)])
        
        print(f"  Running: {' '.join(cmd)}")
        
        # Run workflow
        try:
            # Don't capture output - let it stream directly to console
            # This allows us to see progress in real-time
            result = subprocess.run(cmd, check=True)
            
            # Find and rename DEM
            dem_path = self.find_final_dem(granule_dir)
            
            if not dem_path:
                print(f"  ✗ Failed: Could not find final DEM")
                self.failed.append(granule_name)
                return False
            
            # Extract datetime and create new filename
            datetime_str = self.extract_datetime_from_granule(granule_name)
            
            if datetime_str:
                new_dem_name = f"ASTER_DEM_{datetime_str}.tif"
            else:
                # Fallback to granule name if datetime extraction fails
                new_dem_name = f"ASTER_DEM_{granule_name}.tif"
            
            # Copy DEM to final directory
            self.final_dem_dir.mkdir(parents=True, exist_ok=True)
            final_dem_path = self.final_dem_dir / new_dem_name
            
            print(f"  → Copying DEM: {dem_path.name}")
            print(f"             to: {final_dem_path}")
            
            shutil.copy2(dem_path, final_dem_path)
            
            # Clean up intermediate files if requested
            if not self.keep_intermediates:
                print(f"  → Cleaning intermediate files...")
                # Keep only the essential outputs
                try:
                    # Remove large intermediate directories but keep logs
                    for subdir in ['calibrated', 's2p_output']:
                        subdir_path = granule_dir / subdir
                        if subdir_path.exists():
                            # Keep only key files
                            for item in subdir_path.iterdir():
                                if item.is_file() and item.suffix not in ['.json', '.log', '.txt']:
                                    item.unlink()
                except Exception as e:
                    print(f"  ⚠ Warning: Cleanup failed: {e}")
            
            print(f"  ✓ Success: {new_dem_name}")
            self.processed.append(granule_name)
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed: Workflow error")
            print(f"  Return code: {e.returncode}")
            self.failed.append(granule_name)
            return False
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            self.failed.append(granule_name)
            return False
    
    def run(self, hdf_files):
        """Process all HDF files"""
        print("\n" + "="*70)
        print("ASTER BATCH PROCESSOR")
        print("="*70)
        print(f"Work directory: {self.work_dir}")
        print(f"Output directory: {self.final_dem_dir}")
        print(f"Quality: {self.quality}")
        print(f"Enhancement: {self.enhance}")
        print(f"Resolution: {self.resolution}m")
        print(f"Granules to process: {len(hdf_files)}")
        print("="*70)
        
        # Create directories
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.final_dem_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each granule
        for i, hdf_file in enumerate(hdf_files, 1):
            print(f"\n[{i}/{len(hdf_files)}]")
            self.process_granule(hdf_file)
        
        # Summary
        print("\n" + "="*70)
        print("BATCH PROCESSING COMPLETE")
        print("="*70)
        print(f"✓ Processed: {len(self.processed)}")
        print(f"✗ Failed: {len(self.failed)}")
        if self.skipped:
            print(f"⊘ Skipped: {len(self.skipped)}")
        print(f"\nFinal DEMs location: {self.final_dem_dir}")
        print("="*70)
        
        # List final DEMs
        dem_files = sorted(self.final_dem_dir.glob('*.tif'))
        if dem_files:
            print(f"\nFinal DEMs ({len(dem_files)}):")
            for dem in dem_files:
                size_mb = dem.stat().st_size / (1024*1024)
                print(f"  {dem.name} ({size_mb:.1f} MB)")
        
        if self.failed:
            print(f"\nFailed granules:")
            for name in self.failed:
                print(f"  - {name}")
        
        return len(self.failed) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Batch process multiple ASTER HDF granules',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process all HDF files in a directory
    python3 aster_batch_processor.py /data/aster/*.hdf
    
    # Process from directory with quality preset
    python3 aster_batch_processor.py --input-dir /data/aster --quality best
    
    # Process with custom s2p config
    python3 aster_batch_processor.py /data/*.hdf --config my_s2p.json
    
    # Keep intermediate files
    python3 aster_batch_processor.py /data/*.hdf --keep-intermediates
    
    # Process from file list
    python3 aster_batch_processor.py --file-list granules.txt --enhance always

Output:
    - Each granule processed in: <work-dir>/<granule-name>/
    - Final DEMs collected in: <output-dir>/ASTER_DEM_<datetime>.tif
        """
    )
    
    # Input specification (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('hdf_files', nargs='*', help='HDF files to process')
    input_group.add_argument('--input-dir', help='Directory containing HDF files')
    input_group.add_argument('--file-list', help='Text file with list of HDF files (one per line)')
    
    # Processing options
    parser.add_argument('--work-dir', default='aster_processing',
                       help='Working directory for processing (default: aster_processing)')
    parser.add_argument('--output-dir', default='final_dems',
                       help='Output directory for final DEMs (default: final_dems)')
    parser.add_argument('--workflow-script', default='aster_complete_workflow.py',
                       help='Path to workflow script (default: aster_complete_workflow.py)')
    
    # Workflow parameters
    parser.add_argument('--quality', choices=['fast', 'balanced', 'best'], default='balanced',
                       help='Processing quality preset (default: balanced)')
    parser.add_argument('--enhance', choices=['none', 'auto', 'always'], default='none',
                       help='Image enhancement (default: none)')
    parser.add_argument('--resolution', type=int, default=30,
                       help='DEM resolution in meters (default: 30)')
    parser.add_argument('--utm-zone', help='UTM zone (e.g., "11 +north") - auto-detected if not provided')
    parser.add_argument('--config', help='Custom s2p config JSON file')
    
    # Management options
    parser.add_argument('--keep-intermediates', action='store_true',
                       help='Keep intermediate processing files (uses more disk space)')
    
    args = parser.parse_args()
    
    # Gather HDF files
    hdf_files = []
    
    if args.hdf_files:
        hdf_files = args.hdf_files
    elif args.input_dir:
        input_path = Path(args.input_dir)
        if not input_path.is_dir():
            print(f"Error: {input_path} is not a directory")
            sys.exit(1)
        hdf_files = list(input_path.glob('*.hdf')) + list(input_path.glob('*.HDF'))
    elif args.file_list:
        list_path = Path(args.file_list)
        if not list_path.is_file():
            print(f"Error: {list_path} does not exist")
            sys.exit(1)
        with open(list_path) as f:
            hdf_files = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Validate files exist
    valid_files = []
    for hdf in hdf_files:
        hdf_path = Path(hdf)
        if hdf_path.is_file():
            valid_files.append(hdf_path)
        else:
            print(f"Warning: File not found: {hdf}")
    
    if not valid_files:
        print("Error: No valid HDF files found")
        sys.exit(1)
    
    # Create processor and run
    processor = ASTERBatchProcessor(
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        workflow_script=args.workflow_script,
        quality=args.quality,
        enhance=args.enhance,
        resolution=args.resolution,
        utm_zone=args.utm_zone,
        config=args.config,
        keep_intermediates=args.keep_intermediates
    )
    
    success = processor.run(valid_files)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

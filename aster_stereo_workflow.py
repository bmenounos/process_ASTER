#!/usr/bin/env python3
"""
ASTER Stereo Pipeline Workflow
Processes ASTER L1A data through NASA Ames Stereo Pipeline to produce DEMs.

Workflow:
1. Extract ASTER L1A HDF files using aster_hdf2geotiff.py
2. Generate camera models and imagery with aster2asp
3. Group scenes by acquisition date
4. Mosaic and map project scenes for each day (Band 3N and 3B separately)
5. Remove jitter from mosaics
6. Run parallel_stereo on left/right strips
7. Generate final 25m DEM with point2dem
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import json
import shutil
import logging
import traceback

class ASTERStereoWorkflow:
    """Manages ASTER stereo processing workflow."""
    
    def __init__(self, work_dir, asp_bin=None, num_threads=8, log_file=None):
        """
        Initialize workflow manager.
        
        Args:
            work_dir: Working directory containing ASTER data
            asp_bin: Path to ASP bin directory (if not in PATH)
            num_threads: Number of parallel threads for processing
            log_file: Path to log file (default: work_dir/aster_workflow_YYYYMMDD_HHMMSS.log)
        """
        self.work_dir = Path(work_dir).resolve()
        self.asp_bin = Path(asp_bin) if asp_bin else None
        self.num_threads = num_threads
        
        # Setup logging
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = self.work_dir / f"aster_workflow_{timestamp}.log"
        else:
            log_file = Path(log_file)
        
        self.log_file = log_file
        self.setup_logging()
        
        # Log initialization
        self.logger.info("="*80)
        self.logger.info("ASTER Stereo Workflow Initialized")
        self.logger.info("="*80)
        self.logger.info(f"Working directory: {self.work_dir}")
        self.logger.info(f"Log file: {self.log_file}")
        self.logger.info(f"ASP bin directory: {self.asp_bin if self.asp_bin else 'Using PATH'}")
        self.logger.info(f"Number of threads: {self.num_threads}")
        self.logger.info(f"Python version: {sys.version}")
        self.logger.info("="*80)
        
        # Create subdirectories
        self.extracted_dir = self.work_dir / "01_extracted"
        self.asp_dir = self.work_dir / "02_asp_products"
        self.mosaics_dir = self.work_dir / "03_mosaics"
        self.jitter_dir = self.work_dir / "04_jitter_removed"
        self.stereo_dir = self.work_dir / "05_stereo"
        self.dem_dir = self.work_dir / "06_final_dems"
        
        for d in [self.extracted_dir, self.asp_dir, self.mosaics_dir, 
                  self.jitter_dir, self.stereo_dir, self.dem_dir]:
            d.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Created/verified directory: {d}")
    
    def setup_logging(self):
        """Setup logging configuration."""
        # Create logger
        self.logger = logging.getLogger('ASTERWorkflow')
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler (detailed logging)
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler (less verbose)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_section(self, title):
        """Log a section header."""
        separator = "="*80
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info(title)
        self.logger.info(separator)
    
    def run_command(self, cmd, desc=""):
        """Execute a shell command with error handling and logging."""
        if desc:
            self.log_section(desc)
        
        cmd_str = ' '.join(str(c) for c in cmd)
        self.logger.info(f"Executing command: {cmd_str}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            if result.stdout:
                self.logger.debug(f"stdout:\n{result.stdout}")
                # Only print to console if it's short
                if len(result.stdout) < 500:
                    print(result.stdout)
            
            if result.stderr:
                self.logger.debug(f"stderr:\n{result.stderr}")
            
            self.logger.info("Command completed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Command failed with return code {e.returncode}")
            self.logger.error(f"Command: {cmd_str}")
            if e.stdout:
                self.logger.error(f"stdout:\n{e.stdout}")
            if e.stderr:
                self.logger.error(f"stderr:\n{e.stderr}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error executing command: {str(e)}")
            self.logger.error(f"Command: {cmd_str}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False
    
    def get_asp_command(self, tool):
        """Get full path to ASP tool."""
        if self.asp_bin:
            return str(self.asp_bin / tool)
        return tool
    
    def step1_extract_hdf_files(self, hdf_pattern=".hdf"):
        """
        Step 1: Extract ASTER L1A HDF files.
        
        Args:
            hdf_pattern: Glob pattern for HDF files
        """
        self.log_section("STEP 1: Extracting ASTER L1A HDF files")
        
        hdf_files = list(self.work_dir.glob(hdf_pattern))
        
        if not hdf_files:
            self.logger.error(f"No HDF files found matching pattern: {hdf_pattern}")
            return []
        
        self.logger.info(f"Found {len(hdf_files)} HDF files to process")
        for hdf_file in hdf_files:
            self.logger.info(f"  - {hdf_file.name}")
        
        extracted_dirs = []
        success_count = 0
        fail_count = 0
        
        for i, hdf_file in enumerate(hdf_files, 1):
            self.logger.info(f"\n[{i}/{len(hdf_files)}] Extracting: {hdf_file.name}")
            
            # Extract granule name for output directory
            granule_name = hdf_file.stem
            output_dir = self.extracted_dir / granule_name
            
            self.logger.debug(f"Output directory: {output_dir}")
            
            # Run extraction script
            cmd = [
                "python3",
                "aster_hdf2geotiff.py",
                str(hdf_file),
                str(output_dir)
            ]
            
            if self.run_command(cmd, f"Extracting {hdf_file.name}"):
                extracted_dirs.append(output_dir)
                success_count += 1
                self.logger.info(f"Successfully extracted to: {output_dir}")
            else:
                fail_count += 1
                self.logger.error(f"Failed to extract: {hdf_file.name}")
        
        self.logger.info(f"\nExtraction summary:")
        self.logger.info(f"  Total files: {len(hdf_files)}")
        self.logger.info(f"  Successful: {success_count}")
        self.logger.info(f"  Failed: {fail_count}")
        
        return extracted_dirs
    
    def step2_run_aster2asp(self, extracted_dirs=None):
        """
        Step 2: Run aster2asp to generate camera models and ASP-compatible imagery.
        
        Args:
            extracted_dirs: List of directories with extracted data (or auto-detect)
        """
        self.log_section("STEP 2: Running aster2asp")
        
        if extracted_dirs is None:
            extracted_dirs = [d for d in self.extracted_dir.iterdir() if d.is_dir()]
            self.logger.info(f"Auto-detected {len(extracted_dirs)} extracted directories")
        
        if not extracted_dirs:
            self.logger.error("No extracted directories found.")
            return {}
        
        asp_products = defaultdict(list)
        success_count = 0
        fail_count = 0
        
        for i, ext_dir in enumerate(extracted_dirs, 1):
            self.logger.info(f"\n[{i}/{len(extracted_dirs)}] Processing: {ext_dir.name}")
            
            # Verify the directory contains required ASTER data
            band3n_tif = list(ext_dir.glob("*.VNIR_Band3N.ImageData.tif"))
            band3b_tif = list(ext_dir.glob("*.VNIR_Band3B.ImageData.tif"))
            
            self.logger.debug(f"Band3N files found: {len(band3n_tif)}")
            self.logger.debug(f"Band3B files found: {len(band3b_tif)}")
            
            if not band3n_tif or not band3b_tif:
                self.logger.warning(f"Missing Band3N or Band3B in {ext_dir.name}")
                self.logger.debug(f"Band3N: {band3n_tif}")
                self.logger.debug(f"Band3B: {band3b_tif}")
                fail_count += 1
                continue
            
            self.logger.debug(f"Using Band3N: {band3n_tif[0].name}")
            self.logger.debug(f"Using Band3B: {band3b_tif[0].name}")
            
            # Output prefix for this scene
            scene_prefix = ext_dir.name
            output_prefix = self.asp_dir / scene_prefix
            
            # Run aster2asp - it expects the directory containing extracted ASTER data
            cmd = [
                self.get_asp_command("aster2asp"),
                str(ext_dir),  # Pass the directory, not individual files
                "-o", str(output_prefix)
            ]
            
            if self.run_command(cmd, f"Running aster2asp for {scene_prefix}"):
                # Extract date from granule name
                date = self.extract_date_from_granule(scene_prefix)
                self.logger.debug(f"Extracted date: {date}")
                
                product_info = {
                    'prefix': output_prefix,
                    'left': f"{output_prefix}-Band3N.tif",
                    'right': f"{output_prefix}-Band3B.tif",
                    'left_cam': f"{output_prefix}-Band3N.tsai",
                    'right_cam': f"{output_prefix}-Band3B.tsai"
                }
                
                # Verify output files were created
                if Path(product_info['left']).exists() and Path(product_info['right']).exists():
                    asp_products[date].append(product_info)
                    success_count += 1
                    self.logger.info(f"Successfully processed scene for date {date}")
                    self.logger.debug(f"Created: {Path(product_info['left']).name}")
                    self.logger.debug(f"Created: {Path(product_info['right']).name}")
                else:
                    fail_count += 1
                    self.logger.error(f"Output files not created for {scene_prefix}")
                    self.logger.debug(f"Expected: {product_info['left']}")
                    self.logger.debug(f"Expected: {product_info['right']}")
            else:
                fail_count += 1
                self.logger.error(f"Failed to process: {scene_prefix}")
        
        self.logger.info(f"\naster2asp summary:")
        self.logger.info(f"  Total scenes: {len(extracted_dirs)}")
        self.logger.info(f"  Successful: {success_count}")
        self.logger.info(f"  Failed: {fail_count}")
        self.logger.info(f"  Unique dates: {len(asp_products)}")
        
        for date, scenes in asp_products.items():
            self.logger.info(f"  Date {date}: {len(scenes)} scenes")
        
        return asp_products
    
    def extract_date_from_granule(self, granule_name):
        """
        Extract acquisition date from ASTER granule name using the format:
        AST_L1A_CCCMMDDYYYYHHMMSS_...
        
        where CCC = Version/Offset, MM = Month (01-12), DD = Day (01-31), 
        and YYYY = Full Year.
        
        Args:
            granule_name (str): The name of the ASTER granule.
            
        Returns:
            str: The date in YYYYMMDD format, or "unknown" on failure.
        """
        try:
            # Extract date portion (format: AST_L1A_CCCMMDDYYYYHHMMSS_...)
            parts = granule_name.split('_')
            
            if len(parts) >= 3:
                date_time_str = parts[2]
                
                # Expected length: CCC(3) + MMDDYYYYHHMMSS(12) = 15 characters
                if len(date_time_str) >= 15:
                    # Parse: CCCMMDDYYYYHHMMSS
                    # We extract MMDDYYYY (characters 3 through 10)
                    date_part = date_time_str[3:11]  # MMDDYYYY
                    
                    # Convert to standard date using MMDDYYYY
                    # Format string is "%m%d%Y"
                    dt = datetime.strptime(date_part, "%m%d%Y")
                    formatted_date = dt.strftime("%Y%m%d")
                    
                    self.logger.debug(f"Parsed date from {granule_name}: MMDDYYYY={date_part} -> {formatted_date}")
                    return formatted_date
                else:
                    self.logger.warning(f"Date string too short in {granule_name} (Expected >= 15 chars): {date_time_str}")
        except Exception as e:
            self.logger.warning(f"Could not extract date from {granule_name}: {e}")
            self.logger.debug(f"Traceback:\n{traceback.format_exc()}")
        
        return "unknown"    

    def step3_mosaic_and_mapproject(self, asp_products):
        """
        Step 3: Mosaic and map project scenes by date.
        
        Args:
            asp_products: Dictionary of ASP products grouped by date
        """
        self.log_section("STEP 3: Mosaicking and map projecting")
        
        mosaicked_products = {}
        
        for date, scenes in asp_products.items():
            if not scenes:
                continue
            
            self.logger.info(f"\nProcessing date: {date} ({len(scenes)} scenes)")
            
            # Create date-specific directory
            date_dir = self.mosaics_dir / date
            date_dir.mkdir(exist_ok=True)
            self.logger.debug(f"Date directory: {date_dir}")
            
            # Separate left (3N) and right (3B) images
            left_images = [Path(s['left']) for s in scenes]
            right_images = [Path(s['right']) for s in scenes]
            
            self.logger.debug(f"Left images: {[img.name for img in left_images]}")
            self.logger.debug(f"Right images: {[img.name for img in right_images]}")
            
            # Mosaic left images (Band3N)
            left_mosaic = date_dir / f"{date}_Band3N_mosaic.tif"
            self.logger.info(f"Creating left mosaic: {left_mosaic.name}")
            self.mosaic_images(left_images, left_mosaic)
            
            # Mosaic right images (Band3B)
            right_mosaic = date_dir / f"{date}_Band3B_mosaic.tif"
            self.logger.info(f"Creating right mosaic: {right_mosaic.name}")
            self.mosaic_images(right_images, right_mosaic)
            
            # Map project mosaics
            # Use average camera model or first camera for initial projection
            ref_dem = None  # Can specify reference DEM if available
            
            left_mapped = date_dir / f"{date}_Band3N_mapped.tif"
            right_mapped = date_dir / f"{date}_Band3B_mapped.tif"
            
            # For ASTER, we'll use the mosaics directly if no reference DEM
            # In practice, you'd use mapproject with camera models
            # This is a simplified approach - see notes below
            
            mosaicked_products[date] = {
                'left_mosaic': left_mosaic,
                'right_mosaic': right_mosaic,
                'left_mapped': left_mapped if left_mapped.exists() else left_mosaic,
                'right_mapped': right_mapped if right_mapped.exists() else right_mosaic,
                'cameras': [Path(s['left_cam']) for s in scenes]
            }
            
            self.logger.info(f"Completed mosaicking for date {date}")
        
        self.logger.info(f"\nMosaicking summary:")
        self.logger.info(f"  Dates processed: {len(mosaicked_products)}")
        
        return mosaicked_products
    
    def mosaic_images(self, image_list, output_mosaic):
        """
        Mosaic multiple images using GDAL.
        
        Args:
            image_list: List of image paths to mosaic
            output_mosaic: Output mosaic path
        """
        if not image_list:
            self.logger.warning("Empty image list for mosaicking")
            return False
        
        self.logger.info(f"Mosaicking {len(image_list)} images")
        
        if len(image_list) == 1:
            # Just copy if single image
            shutil.copy(image_list[0], output_mosaic)
            self.logger.info(f"Single image, copied to: {output_mosaic}")
            return True
        
        # Use gdal_merge.py or similar
        cmd = [
            "gdal_merge.py",
            "-o", str(output_mosaic),
            "-co", "COMPRESS=LZW",
            "-co", "TILED=YES"
        ] + [str(img) for img in image_list]
        
        success = self.run_command(cmd, f"Mosaicking {len(image_list)} images")
        
        if success and output_mosaic.exists():
            file_size_mb = output_mosaic.stat().st_size / (1024 * 1024)
            self.logger.info(f"Mosaic created: {output_mosaic.name} ({file_size_mb:.2f} MB)")
        
        return success
    
    def step4_remove_jitter(self, mosaicked_products):
        """
        Step 4: Remove jitter from mosaicked images.
        
        Args:
            mosaicked_products: Dictionary of mosaicked products by date
        """
        self.log_section("STEP 4: Removing jitter")
        
        jitter_removed_products = {}
        
        for date, products in mosaicked_products.items():
            self.logger.info(f"\nProcessing date: {date}")
            
            date_dir = self.jitter_dir / date
            date_dir.mkdir(exist_ok=True)
            self.logger.debug(f"Jitter output directory: {date_dir}")
            
            left_dejitter = date_dir / f"{date}_Band3N_dejitter.tif"
            right_dejitter = date_dir / f"{date}_Band3B_dejitter.tif"
            
            # Run jitter_solve on left image
            self.logger.info(f"Processing left image: {products['left_mapped'].name}")
            self.run_jitter_solve(
                products['left_mapped'],
                left_dejitter
            )
            
            # Run jitter_solve on right image
            self.logger.info(f"Processing right image: {products['right_mapped'].name}")
            self.run_jitter_solve(
                products['right_mapped'],
                right_dejitter
            )
            
            jitter_removed_products[date] = {
                'left': left_dejitter,
                'right': right_dejitter,
                'cameras': products['cameras']
            }
            
            self.logger.info(f"Completed jitter removal for date {date}")
        
        self.logger.info(f"\nJitter removal summary:")
        self.logger.info(f"  Dates processed: {len(jitter_removed_products)}")
        
        return jitter_removed_products
    
    def run_jitter_solve(self, input_image, output_image):
        """
        Run jitter_solve on an image.
        
        Note: ASP's jitter_solve requires camera models. This is a placeholder
        for the actual jitter removal process which may involve bundle_adjust
        or other ASP tools depending on your specific needs.
        """
        self.logger.warning(f"Jitter removal not yet implemented - copying image")
        self.logger.debug(f"Input: {input_image}")
        self.logger.debug(f"Output: {output_image}")
        
        # For now, just copy the image (you'll need to implement actual jitter removal)
        try:
            shutil.copy(input_image, output_image)
            self.logger.debug(f"Copied {input_image.name} to {output_image.name}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to copy image: {e}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            return False
    
    def step5_run_parallel_stereo(self, jitter_removed_products):
        """
        Step 5: Run parallel_stereo on left/right strips.
        
        Args:
            jitter_removed_products: Dictionary of jitter-removed products by date
        """
        self.log_section("STEP 5: Running parallel_stereo")
        
        stereo_products = {}
        success_count = 0
        fail_count = 0
        
        for date, products in jitter_removed_products.items():
            self.logger.info(f"\nProcessing date: {date}")
            
            date_dir = self.stereo_dir / date
            date_dir.mkdir(exist_ok=True)
            self.logger.debug(f"Stereo output directory: {date_dir}")
            
            output_prefix = date_dir / f"{date}_stereo"
            
            self.logger.info(f"Left image: {products['left'].name}")
            self.logger.info(f"Right image: {products['right'].name}")
            self.logger.debug(f"Output prefix: {output_prefix}")
            
            # Run parallel_stereo
            # Note: This assumes you have appropriate camera models
            # You may need to adjust stereo parameters for ASTER
            
            cmd = [
                self.get_asp_command("parallel_stereo"),
                str(products['left']),
                str(products['right']),
                str(output_prefix),
                "--stereo-algorithm", "asp_mgm",  # or asp_sgm, asp_bm
                "--subpixel-mode", "3",
                "--threads", str(self.num_threads),
                "--corr-seed-mode", "1"
            ]
            
            # Add camera models if available
            if products['cameras']:
                # For mosaicked data, this is complex - may need virtual cameras
                # This is a simplified version
                self.logger.debug(f"Camera models available: {len(products['cameras'])}")
                pass
            
            if self.run_command(cmd, f"Running parallel_stereo for {date}"):
                point_cloud = f"{output_prefix}-PC.tif"
                stereo_products[date] = {
                    'prefix': output_prefix,
                    'point_cloud': point_cloud
                }
                success_count += 1
                self.logger.info(f"Successfully completed stereo for {date}")
                
                # Check if point cloud was created
                if Path(point_cloud).exists():
                    file_size_mb = Path(point_cloud).stat().st_size / (1024 * 1024)
                    self.logger.info(f"Point cloud size: {file_size_mb:.2f} MB")
            else:
                fail_count += 1
                self.logger.error(f"Failed stereo processing for {date}")
        
        self.logger.info(f"\nStereo processing summary:")
        self.logger.info(f"  Total dates: {len(jitter_removed_products)}")
        self.logger.info(f"  Successful: {success_count}")
        self.logger.info(f"  Failed: {fail_count}")
        
        return stereo_products
    
    def step6_generate_dem(self, stereo_products, resolution=25.0):
        """
        Step 6: Generate final DEM using point2dem.
        
        Args:
            stereo_products: Dictionary of stereo products by date
            resolution: Output DEM resolution in meters
        """
        self.log_section(f"STEP 6: Generating {resolution}m DEMs")
        
        dem_products = {}
        success_count = 0
        fail_count = 0
        
        for date, products in stereo_products.items():
            self.logger.info(f"\nProcessing date: {date}")
            self.logger.debug(f"Point cloud: {products['point_cloud']}")
            
            output_dem = self.dem_dir / f"{date}_DEM_{int(resolution)}m.tif"
            
            cmd = [
                self.get_asp_command("point2dem"),
                str(products['point_cloud']),
                "-o", str(output_dem.parent / output_dem.stem),
                "--tr", str(resolution),
                "--threads", str(self.num_threads),
                "--errorimage"
            ]
            
            if self.run_command(cmd, f"Generating DEM for {date}"):
                error_image = str(output_dem).replace('.tif', '-IntersectionErr.tif')
                dem_products[date] = {
                    'dem': output_dem,
                    'error': error_image
                }
                success_count += 1
                self.logger.info(f"Successfully generated DEM for {date}")
                
                # Log DEM file size
                if output_dem.exists():
                    file_size_mb = output_dem.stat().st_size / (1024 * 1024)
                    self.logger.info(f"DEM size: {file_size_mb:.2f} MB")
                
                # Log error image if it exists
                if Path(error_image).exists():
                    err_size_mb = Path(error_image).stat().st_size / (1024 * 1024)
                    self.logger.info(f"Error image size: {err_size_mb:.2f} MB")
            else:
                fail_count += 1
                self.logger.error(f"Failed to generate DEM for {date}")
        
        self.logger.info(f"\nDEM generation summary:")
        self.logger.info(f"  Total dates: {len(stereo_products)}")
        self.logger.info(f"  Successful: {success_count}")
        self.logger.info(f"  Failed: {fail_count}")
        
        return dem_products
    
    def run_full_workflow(self, hdf_pattern=".hdf", dem_resolution=25.0):
        """
        Execute the complete workflow.
        
        Args:
            hdf_pattern: Pattern for finding HDF files
            dem_resolution: Final DEM resolution in meters
        """
        workflow_start = datetime.now()
        
        self.log_section("ASTER STEREO PIPELINE WORKFLOW - START")
        self.logger.info(f"Working directory: {self.work_dir}")
        self.logger.info(f"Threads: {self.num_threads}")
        self.logger.info(f"Target DEM resolution: {dem_resolution}m")
        self.logger.info(f"HDF pattern: {hdf_pattern}")
        self.logger.info(f"Start time: {workflow_start.strftime('%Y-%m-%d %H:%M:%S')}")
        
        try:
            # Step 1: Extract HDF files
            extracted_dirs = self.step1_extract_hdf_files(hdf_pattern)
            if not extracted_dirs:
                self.logger.error("No files extracted. Exiting workflow.")
                return None
            
            # Step 2: Run aster2asp
            asp_products = self.step2_run_aster2asp(extracted_dirs)
            if not asp_products:
                self.logger.error("No ASP products generated. Exiting workflow.")
                return None
            
            # Step 3: Mosaic and map project
            mosaicked_products = self.step3_mosaic_and_mapproject(asp_products)
            if not mosaicked_products:
                self.logger.error("No mosaics created. Exiting workflow.")
                return None
            
            # Step 4: Remove jitter
            jitter_removed = self.step4_remove_jitter(mosaicked_products)
            
            # Step 5: Run stereo
            stereo_products = self.step5_run_parallel_stereo(jitter_removed)
            if not stereo_products:
                self.logger.error("Stereo processing failed. Exiting workflow.")
                return None
            
            # Step 6: Generate DEMs
            dem_products = self.step6_generate_dem(stereo_products, dem_resolution)
            
            # Workflow completion summary
            workflow_end = datetime.now()
            duration = workflow_end - workflow_start
            
            self.log_section("WORKFLOW COMPLETE")
            self.logger.info(f"End time: {workflow_end.strftime('%Y-%m-%d %H:%M:%S')}")
            self.logger.info(f"Total duration: {duration}")
            self.logger.info(f"Generated DEMs for {len(dem_products)} dates:")
            
            for date, products in dem_products.items():
                self.logger.info(f"  {date}: {products['dem']}")
                if Path(products['dem']).exists():
                    size_mb = Path(products['dem']).stat().st_size / (1024 * 1024)
                    self.logger.info(f"    Size: {size_mb:.2f} MB")
            
            self.logger.info(f"\nLog file saved to: {self.log_file}")
            
            return dem_products
            
        except Exception as e:
            self.logger.error(f"Workflow failed with exception: {str(e)}")
            self.logger.error(f"Traceback:\n{traceback.format_exc()}")
            
            workflow_end = datetime.now()
            duration = workflow_end - workflow_start
            self.logger.error(f"Failed after: {duration}")
            self.logger.info(f"\nLog file saved to: {self.log_file}")
            
            raise


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="ASTER Stereo Pipeline Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all HDF files in current directory
  python aster_stereo_workflow.py .
  
  # Process with custom ASP installation
  python aster_stereo_workflow.py /data/aster --asp-bin /opt/StereoPipeline/bin
  
  # Custom resolution and threads
  python aster_stereo_workflow.py /data/aster --resolution 30 --threads 16
  
  # Specify custom log file
  python aster_stereo_workflow.py /data/aster --log-file /logs/aster_processing.log
        """
    )
    
    parser.add_argument(
        "work_dir",
        help="Working directory containing ASTER HDF files"
    )
    
    parser.add_argument(
        "--asp-bin",
        help="Path to ASP bin directory (if not in PATH)"
    )
    
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of parallel threads (default: 8)"
    )
    
    parser.add_argument(
        "--resolution",
        type=float,
        default=25.0,
        help="Output DEM resolution in meters (default: 25.0)"
    )
    
    parser.add_argument(
        "--hdf-pattern",
        default=".hdf",
        help="Glob pattern for HDF files (default: *.hdf)"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file (default: work_dir/aster_workflow_YYYYMMDD_HHMMSS.log)"
    )
    
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip HDF extraction step (use existing extracted data)"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize workflow
        workflow = ASTERStereoWorkflow(
            work_dir=args.work_dir,
            asp_bin=args.asp_bin,
            num_threads=args.threads,
            log_file=args.log_file
        )
        
        # Run workflow
        if args.skip_extract:
            workflow.logger.info("Skipping extraction step, using existing data...")
            extracted_dirs = [d for d in workflow.extracted_dir.iterdir() if d.is_dir()]
            asp_products = workflow.step2_run_aster2asp(extracted_dirs)
            mosaicked = workflow.step3_mosaic_and_mapproject(asp_products)
            jitter_removed = workflow.step4_remove_jitter(mosaicked)
            stereo_products = workflow.step5_run_parallel_stereo(jitter_removed)
            workflow.step6_generate_dem(stereo_products, args.resolution)
        else:
            workflow.run_full_workflow(
                hdf_pattern=args.hdf_pattern,
                dem_resolution=args.resolution
            )
        
        workflow.logger.info("Script completed successfully")
        return 0
        
    except Exception as e:
        print(f"\nFATAL ERROR: {str(e)}", file=sys.stderr)
        print(f"Check log file for details: {args.log_file or 'aster_workflow_*.log'}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

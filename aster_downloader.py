import requests
import os
from datetime import datetime
from pathlib import Path
import json
import boto3
from botocore.exceptions import NoCredentialsError

class ASTERDaytimeDownloader:
    """
    Download ASTER L1A V004 HDF files (daytime only) from AWS S3 using NASA Earthdata credentials.
    
    You'll need to create a NASA Earthdata account at: https://urs.earthdata.nasa.gov/
    """

    def __init__(self, username, password):
        """
        Initialize downloader with NASA Earthdata credentials.

        Args:
            username: NASA Earthdata username
            password: NASA Earthdata password
        """
        self.username = username
        self.password = password
        self.version = '004'
        
        # V004 Collection ID (cloud-hosted on AWS)
        self.collection_id = 'C3306888985-LPCLOUD'
        
        self.cmr_search_url = "https://cmr.earthdata.nasa.gov/search/granules.json"
        self.s3_credentials_url = "https://data.lpdaac.earthdatacloud.nasa.gov/s3credentials"
        self.s3_bucket = "lp-prod-protected"
        self.s3_client = None
        
        print(f"Initialized ASTER L1A V004 Daytime-only downloader")
        print(f"Collection ID: {self.collection_id}")

    def get_s3_credentials(self):
        """
        Get temporary S3 credentials from NASA Earthdata.
        
        Returns:
            Dictionary with AWS credentials
        """
        try:
            response = requests.get(
                self.s3_credentials_url,
                auth=(self.username, self.password)
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error getting S3 credentials: {e}")
            return None

    def init_s3_client(self):
        """
        Initialize S3 client with temporary credentials.
        """
        if self.s3_client is not None:
            return True
            
        print("Getting temporary S3 credentials...")
        credentials = self.get_s3_credentials()
        
        if not credentials:
            print("Failed to get S3 credentials")
            return False
        
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials['accessKeyId'],
                aws_secret_access_key=credentials['secretAccessKey'],
                aws_session_token=credentials['sessionToken'],
                region_name='us-west-2'
            )
            print("✓ S3 client initialized")
            return True
        except Exception as e:
            print(f"Error initializing S3 client: {e}")
            return False

    def load_geojson_polygon(self, geojson_path):
        """
        Load polygon coordinates from GeoJSON file.

        Args:
            geojson_path: Path to GeoJSON file

        Returns:
            List of [lon, lat] coordinate pairs
        """
        with open(geojson_path, 'r') as f:
            data = json.load(f)
        
        coords = data['features'][0]['geometry']['coordinates'][0]
        return coords

    def polygon_to_cmr_format(self, coords):
        """
        Convert polygon coordinates to CMR API format.

        Args:
            coords: List of [lon, lat] pairs

        Returns:
            Comma-separated string of lon,lat pairs
        """
        coord_pairs = [f"{lon},{lat}" for lon, lat in coords]
        return ",".join(coord_pairs)

    def is_daytime_granule(self, granule):
        """
        Check if a granule is a daytime acquisition.
        
        Args:
            granule: Granule metadata dictionary
            
        Returns:
            Boolean indicating if granule is daytime
        """
        # Check for day/night flag in granule metadata
        # ASTER uses 'Day' or 'Night' in the additional attributes
        day_night_flag = granule.get('day_night_flag', '').upper()
        
        if day_night_flag == 'DAY':
            return True
        elif day_night_flag == 'NIGHT':
            return False
        
        # Alternative: check in data_granule
        data_granule = granule.get('data_granule', {})
        day_night = data_granule.get('day_night_flag', '').upper()
        
        if day_night == 'DAY':
            return True
        elif day_night == 'NIGHT':
            return False
        
        # If not explicitly marked, assume daytime (most ASTER data is daytime)
        # But log a warning
        granule_id = granule.get('title', 'Unknown')
        print(f"  ⚠ Warning: Could not determine day/night flag for {granule_id}, assuming daytime")
        return True

    def search_granules(self, polygon=None, bbox=None, start_date=None, end_date=None, 
                        max_results=100, max_cloud_cover=None): # ADDED max_cloud_cover
        """
        Search for ASTER L1A V004 granules within AOI and date range.

        Args:
            polygon: List of [lon, lat] coordinate pairs for polygon search
            bbox: Bounding box as [min_lon, min_lat, max_lon, max_lat]
            start_date: Start date as string 'YYYY-MM-DD'
            end_date: End date as string 'YYYY-MM-DD'
            max_results: Maximum number of results to return
            max_cloud_cover: Maximum percentage of cloud cover (0-100)

        Returns:
            List of granule metadata dictionaries
        """
        params = {
            'collection_concept_id': self.collection_id,
            'page_size': max_results,
            'sort_key': '-start_date'
        }

        if polygon:
            params['polygon'] = self.polygon_to_cmr_format(polygon)
            print(f"Searching with polygon ({len(polygon)} points)...")
        elif bbox:
            bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            params['bounding_box'] = bbox_str
            print(f"Searching with bounding box: {bbox}")
        else:
            raise ValueError("Either polygon or bbox must be provided")

        if start_date and end_date:
            params['temporal'] = f"{start_date}T00:00:00Z,{end_date}T23:59:59Z"
            print(f"  Date range: {start_date} to {end_date}")

        # --- ADDED CLOUD COVER FILTER ---
        if max_cloud_cover is not None:
            # CMR accepts cloud_cover as a range: min,max. We want 0 up to max_cloud_cover
            if 0 <= max_cloud_cover <= 100:
                params['cloud_cover'] = f"0,{max_cloud_cover}"
                print(f"  Filter: Max Cloud Cover: {max_cloud_cover}%")
            else:
                print(f"  Warning: Invalid max_cloud_cover value ({max_cloud_cover}). Skipping filter.")
        # --- END CLOUD COVER FILTER ---

        print(f"  Collection: ASTER L1A V004")
        print(f"  Filter: Daytime granules only")
        
        response = requests.get(self.cmr_search_url, params=params)
        response.raise_for_status()

        data = response.json()
        granules = data.get('feed', {}).get('entry', [])

        print(f"Found {len(granules)} total granules (before daytime filter)")
        
        # Filter for daytime granules only
        daytime_granules = [g for g in granules if self.is_daytime_granule(g)]
        
        print(f"Filtered to {len(daytime_granules)} granules for download")
        
        # Debug: print first granule info
        if daytime_granules:
            first = daytime_granules[0]
            print(f"  First granule: {first.get('title', 'Unknown')}")
            # Optional: Print cloud cover if available (CMR usually includes it if requested)
            cloud_cover_value = first.get('cloud_cover', 'N/A')
            print(f"  Cloud Cover: {cloud_cover_value}%")
            print(f"  Has links: {len(first.get('links', []))} links")
        
        return daytime_granules

    def extract_s3_urls(self, granule):
        """
        Extract S3 URLs from granule metadata.
        
        Args:
            granule: Granule metadata dictionary
            
        Returns:
            List of S3 URLs (https:// format for direct download)
        """
        s3_urls = []
        links = granule.get('links', [])
        
        for link in links:
            href = link.get('href', '')
            
            # Look for direct S3 access URLs (LPCLOUD uses HTTPS URLs to S3)
            if 'lpdaac.earthdatacloud.nasa.gov' in href and '.hdf' in href:
                s3_urls.append(href)
            # Also check for s3:// URLs
            elif href.startswith('s3://'):
                s3_urls.append(href)
        
        return s3_urls

    def download_from_url(self, url, output_dir='aster_data'):
        """
        Download a file from HTTPS URL with authentication.

        Args:
            url: Download URL
            output_dir: Directory to save downloaded files

        Returns:
            Path to downloaded file or None if failed
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Extract filename from URL
        filename = os.path.basename(url.split('?')[0])
        output_path = os.path.join(output_dir, filename)

        # Skip if file already exists
        if os.path.exists(output_path):
            print(f"  ✓ Already exists: {filename}")
            return output_path

        print(f"  Downloading: {filename}")
        try:
            # Create session with authentication
            session = requests.Session()
            session.auth = (self.username, self.password)

            # Download file with streaming
            response = session.get(url, stream=True, allow_redirects=True)
            response.raise_for_status()

            # Write file in chunks
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            print(f"    Progress: {percent:.1f}%", end='\r')
            
            print(f"  ✓ Downloaded: {filename}                    ")
            return output_path

        except Exception as e:
            print(f"  ✗ Error downloading {filename}: {str(e)}")
            if os.path.exists(output_path):
                os.remove(output_path)
            return None

    def download_granules(self, geojson_path, start_date, end_date, 
                         output_dir='aster_data', max_results=100, max_cloud_cover=None): # ADDED max_cloud_cover
        """
        Search for and download ASTER L1A V004 daytime granules.

        Args:
            geojson_path: Path to GeoJSON file defining AOI
            start_date: Start date as string 'YYYY-MM-DD'
            end_date: End date as string 'YYYY-MM-DD'
            output_dir: Directory to save downloaded files
            max_results: Maximum number of granules to search for
            max_cloud_cover: Maximum percentage of cloud cover (0-100)

        Returns:
            List of paths to downloaded files
        """
        # Load polygon from GeoJSON
        polygon = self.load_geojson_polygon(geojson_path)
        print(f"Loaded polygon from {geojson_path}")

        # Search for granules (will be filtered for daytime and cloud cover)
        granules = self.search_granules(
            polygon=polygon, 
            start_date=start_date, 
            end_date=end_date, 
            max_results=max_results,
            max_cloud_cover=max_cloud_cover # PASSED THROUGH
        )

        if not granules:
            print("No daytime granules found for the specified criteria.")
            return []

        downloaded_files = []
        print(f"\n{'='*60}")
        print(f"Starting download of {len(granules)} daytime granules...")
        print(f"{'='*60}")

        for i, granule in enumerate(granules, 1):
            granule_id = granule.get('title', 'Unknown')
            # Optional: Display cloud cover during download loop
            cloud_cover_display = granule.get('cloud_cover')
            cc_str = f" [CC: {cloud_cover_display}%]" if cloud_cover_display is not None else ""
            print(f"\n[{i}/{len(granules)}] Processing: {granule_id}{cc_str}")

            # Get download URLs
            urls = self.extract_s3_urls(granule)

            if not urls:
                print(f"  ✗ No download URLs found")
                # Debug: print all links
                links = granule.get('links', [])
                print(f"  Available links ({len(links)}):")
                for link in links[:3]:  # Show first 3
                    print(f"    - {link.get('rel', 'N/A')}: {link.get('href', 'N/A')[:80]}")
                continue

            # Download each URL (usually just one HDF file per granule)
            for url in urls:
                file_path = self.download_from_url(url, output_dir)
                if file_path:
                    downloaded_files.append(file_path)

        print(f"\n{'='*60}")
        print(f"Download complete: {len(downloaded_files)}/{len(granules)} files downloaded")
        print(f"Output directory: {os.path.abspath(output_dir)}")
        print(f"{'='*60}")

        return downloaded_files


# Example usage
if __name__ == "__main__":
    # NOTE: Replace with your NASA Earthdata credentials
    # Register at: https://urs.earthdata.nasa.gov/
    USERNAME = "your_earthdata_username"
    PASSWORD = "your_earthdata_password"

    # Define parameters
    GEOJSON_FILE = "aoi_polygon.geojson"
    START_DATE = "2025-07-01"
    END_DATE = "2025-09-30"
    OUTPUT_DIR = "aster_daytime_data"
    
    # --- NEW PARAMETER: Set max cloud cover (e.g., 10 for less than 10% clouds) ---
    MAX_CLOUD_COVER = 10 

    # Create downloader instance
    downloader = ASTERDaytimeDownloader(USERNAME, PASSWORD)

    # Download daytime granules
    downloaded_files = downloader.download_granules(
        geojson_path=GEOJSON_FILE,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir=OUTPUT_DIR,
        max_results=50,
        max_cloud_cover=MAX_CLOUD_COVER # PASS CLOUD COVER FILTER
    )

    # Print summary
    if downloaded_files:
        print(f"\nSuccessfully downloaded {len(downloaded_files)} daytime granules:")
        for file in downloaded_files[:10]:  # Show first 10
            print(f"  - {file}")
        if len(downloaded_files) > 10:
            print(f"  ... and {len(downloaded_files) - 10} more")
    else:
        print("\nNo files were downloaded.")

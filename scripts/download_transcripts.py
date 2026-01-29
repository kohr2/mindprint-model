#!/usr/bin/env python3
"""
Download Bob Loukas video transcripts from Box.com.

Downloads transcripts from Box.com folder and saves them locally as YYYY-MM-DD.txt files.
Uses JWT authentication via BOX_CONFIG_JSON environment variable.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import jwt FIRST before box_sdk_gen to ensure it's available
try:
    import jwt  # PyJWT - must be imported before box_sdk_gen
except ImportError:
    print("ERROR: PyJWT not installed. Run: pip install PyJWT")
    sys.exit(1)

try:
    from box_sdk_gen.client import BoxClient
    from box_sdk_gen.box.jwt_auth import JWTConfig, BoxJWTAuth
    # Patch jwt into box_sdk_gen.internal.utils if it's None
    import box_sdk_gen.internal.utils as utils
    if utils.jwt is None:
        utils.jwt = jwt
        # Also need cryptography modules
        try:
            from cryptography.hazmat.backends import default_backend
            from cryptography.hazmat.primitives import serialization
            utils.default_backend = default_backend
            utils.serialization = serialization
        except ImportError:
            pass
except ImportError:
    try:
        # Fallback to old boxsdk API
        from boxsdk import BoxClient, BoxJwtAuth, JwtConfig
        BoxJWTAuth = BoxJwtAuth  # Alias for compatibility
    except ImportError:
        print("ERROR: boxsdk not installed. Run: pip install boxsdk")
        sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class TranscriptDownloader:
    """Download transcripts from Box.com."""

    def __init__(self, box_config_json: str, folder_id: str):
        """
        Initialize the downloader.

        Args:
            box_config_json: JSON string containing Box JWT config
            folder_id: Box folder ID containing transcripts
        """
        self.folder_id = folder_id
        self.client = self._initialize_client(box_config_json)

    def _initialize_client(self, box_config_json: str) -> BoxClient:
        """Initialize Box client with JWT authentication."""
        try:
            config = json.loads(box_config_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid BOX_CONFIG_JSON: {e}")

        # Extract JWT config
        box_app_settings = config.get("boxAppSettings", {})
        app_auth = box_app_settings.get("appAuth", {})

        # Use box_sdk_gen API (new SDK)
        try:
            from box_sdk_gen.box.jwt_auth import JWTConfig, BoxJWTAuth
            
            jwt_config = JWTConfig(
                client_id=box_app_settings.get("clientID"),
                client_secret=box_app_settings.get("clientSecret"),
                jwt_key_id=app_auth.get("publicKeyID"),
                private_key=app_auth.get("privateKey"),
                private_key_passphrase=app_auth.get("passphrase", ""),
                enterprise_id=config.get("enterpriseID"),
            )
            
            auth = BoxJWTAuth(config=jwt_config)
            return BoxClient(auth=auth)
        except (ImportError, AttributeError, TypeError) as e:
            # Fallback to old boxsdk API if available
            try:
                jwt_config = JwtConfig(
                    client_id=box_app_settings.get("clientID"),
                    client_secret=box_app_settings.get("clientSecret"),
                    jwt_key_id=app_auth.get("publicKeyID"),
                    private_key=app_auth.get("privateKey"),
                    private_key_passphrase=app_auth.get("passphrase", ""),
                    enterprise_id=config.get("enterpriseID"),
                )
                
                auth = BoxJWTAuth(jwt_config)
                return BoxClient(auth)
            except Exception:
                raise ValueError(f"Failed to initialize Box client: {e}")

    def list_transcripts(self, limit: Optional[int] = None) -> List[Dict]:
        """
        List all transcript files in the folder.

        Args:
            limit: Maximum number of files to return

        Returns:
            List of file metadata dictionaries
        """
        try:
            # Use new Box SDK API with managers
            folders_manager = self.client.folders
            folder = folders_manager.get_folder_by_id(folder_id=self.folder_id)
            
            # Get items in folder
            items_response = folders_manager.get_folder_items(folder_id=self.folder_id, limit=limit)
            items = items_response.entries if hasattr(items_response, 'entries') else []

            transcripts = []
            for item in items:
                # Check if it's a file and ends with .txt
                if hasattr(item, 'type') and item.type == 'file' and hasattr(item, 'name') and item.name.endswith(".txt"):
                    transcripts.append({
                        "id": item.id,
                        "name": item.name,
                        "size": getattr(item, 'size', 0),
                        "modified_at": getattr(item, 'modified_at', '').isoformat() if hasattr(getattr(item, 'modified_at', None), 'isoformat') else str(getattr(item, 'modified_at', '')),
                        "created_at": getattr(item, 'created_at', '').isoformat() if hasattr(getattr(item, 'created_at', None), 'isoformat') else str(getattr(item, 'created_at', '')),
                    })

            return sorted(transcripts, key=lambda x: x["modified_at"], reverse=True)

        except Exception as e:
            logger.error(f"Failed to list transcripts: {e}")
            raise

    def download_transcript(self, file_id: str, output_path: Path) -> bool:
        """
        Download a transcript file.

        Args:
            file_id: Box file ID
            output_path: Local path to save the file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Use new Box SDK API with downloads manager
            downloads_manager = self.client.downloads
            file_content = downloads_manager.download_file(file_id=file_id)
            
            # Write to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                if hasattr(file_content, 'read'):
                    f.write(file_content.read())
                elif isinstance(file_content, bytes):
                    f.write(file_content)
                else:
                    # Try to get content as bytes
                    f.write(bytes(file_content))

            logger.info(f"Downloaded: {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"Failed to download {file_id}: {e}")
            return False

    def parse_filename_date(self, filename: str) -> Optional[str]:
        """
        Extract date from filename.

        Expected formats:
        - YYYY-MM-DD.txt
        - session_YYYY-MM-DD.txt
        - YYYYMMDD.txt
        - YYYY-MM-DDTHH-MM-SSZ.txt (ISO timestamp format)
        - YYYY-MM-DDTHH-MM-SSZ_F.txt (ISO timestamp with _F suffix)

        Returns:
            Date string in YYYY-MM-DD format, or None if not parseable
        """
        # Remove .txt extension
        base = filename.replace(".txt", "")
        
        # Remove _F suffix if present (e.g., 2019-02-19T14-28-05Z_F -> 2019-02-19T14-28-05Z)
        if base.endswith("_F"):
            base = base[:-2]

        # Try YYYY-MM-DD format
        if len(base) == 10 and base[4] == "-" and base[7] == "-":
            try:
                datetime.strptime(base, "%Y-%m-%d")
                return base
            except ValueError:
                pass

        # Try ISO timestamp format: YYYY-MM-DDTHH-MM-SSZ
        if 'T' in base and base.count('-') >= 2:
            # Extract date part (YYYY-MM-DD) before 'T'
            date_part = base.split('T')[0]
            if len(date_part) == 10:
                try:
                    datetime.strptime(date_part, "%Y-%m-%d")
                    return date_part
                except ValueError:
                    pass

        # Try session_YYYY-MM-DD format
        if base.startswith("session_"):
            date_part = base.replace("session_", "")
            if len(date_part) == 10 and date_part[4] == "-" and date_part[7] == "-":
                try:
                    datetime.strptime(date_part, "%Y-%m-%d")
                    return date_part
                except ValueError:
                    pass

        # Try YYYYMMDD format
        if len(base) == 8 and base.isdigit():
            try:
                date_obj = datetime.strptime(base, "%Y%m%d")
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                pass

        return None

    def download_all(
        self,
        output_dir: Path,
        count: Optional[int] = None,
        skip_existing: bool = True,
    ) -> Dict:
        """
        Download all transcripts.

        Args:
            output_dir: Directory to save transcripts
            count: Maximum number of transcripts to download
            skip_existing: Skip files that already exist locally

        Returns:
            Dictionary with download statistics
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        # List transcripts
        logger.info(f"Listing transcripts from folder {self.folder_id}...")
        transcripts = self.list_transcripts(limit=count)

        if not transcripts:
            logger.warning("No transcripts found in folder")
            return {
                "total": 0,
                "downloaded": 0,
                "skipped": 0,
                "failed": 0,
                "files": [],
            }

        logger.info(f"Found {len(transcripts)} transcript files")

        stats = {
            "total": len(transcripts),
            "downloaded": 0,
            "skipped": 0,
            "failed": 0,
            "files": [],
        }

        metadata = []

        for transcript in transcripts:
            filename = transcript["name"]
            file_id = transcript["id"]

            # Parse date from filename
            date_str = self.parse_filename_date(filename)
            if not date_str:
                logger.warning(f"Could not parse date from filename: {filename}")
                # Use original filename without extension
                output_filename = filename
            else:
                output_filename = f"{date_str}.txt"

            output_path = raw_dir / output_filename

            # Skip if exists
            if skip_existing and output_path.exists():
                logger.debug(f"Skipping existing file: {output_filename}")
                stats["skipped"] += 1
                metadata.append({
                    "filename": output_filename,
                    "date": date_str,
                    "box_id": file_id,
                    "box_name": filename,
                    "size": transcript["size"],
                    "modified_at": transcript["modified_at"],
                    "status": "skipped",
                })
                continue

            # Download
            success = self.download_transcript(file_id, output_path)
            if success:
                stats["downloaded"] += 1
                status = "downloaded"
            else:
                stats["failed"] += 1
                status = "failed"

            metadata.append({
                "filename": output_filename,
                "date": date_str,
                "box_id": file_id,
                "box_name": filename,
                "size": transcript["size"],
                "modified_at": transcript["modified_at"],
                "status": status,
            })

        # Save metadata
        metadata_path = output_dir / "transcripts_metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Metadata saved to {metadata_path}")

        return stats


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Bob Loukas transcripts from Box.com",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 50 most recent transcripts
  python scripts/download_transcripts.py --count 50

  # Download all transcripts (no limit)
  python scripts/download_transcripts.py

  # Download to custom directory
  python scripts/download_transcripts.py --output-dir data/bob_loukas/transcripts

Environment Variables:
  BOX_CONFIG_JSON: JSON string containing Box JWT configuration
  BOX_TRANSCRIPTS_FOLDER_ID: Box folder ID (default: 319911287224)
""",
    )

    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Maximum number of transcripts to download (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/bob_loukas/transcripts",
        help="Output directory for transcripts (default: data/bob_loukas/transcripts)",
    )
    parser.add_argument(
        "--folder-id",
        type=str,
        default=None,
        help="Box folder ID (default: from BOX_TRANSCRIPTS_FOLDER_ID env or 319911287224)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download existing files",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get Box config from environment
    box_config_json = os.environ.get("BOX_CONFIG_JSON")
    if not box_config_json:
        logger.error("BOX_CONFIG_JSON environment variable is required")
        logger.error("Copy it from mindprint-agent's .env file")
        return 1

    # Get folder ID
    folder_id = args.folder_id or os.environ.get("BOX_TRANSCRIPTS_FOLDER_ID", "319911287224")

    # Initialize downloader
    try:
        downloader = TranscriptDownloader(box_config_json, folder_id)
    except Exception as e:
        logger.error(f"Failed to initialize Box client: {e}")
        return 1

    # Download transcripts
    output_dir = Path(args.output_dir)
    stats = downloader.download_all(
        output_dir=output_dir,
        count=args.count,
        skip_existing=not args.no_skip,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"Total files found:  {stats['total']}")
    print(f"Downloaded:         {stats['downloaded']}")
    print(f"Skipped:            {stats['skipped']}")
    print(f"Failed:             {stats['failed']}")
    print(f"Output directory:   {output_dir.absolute()}")
    print("=" * 60 + "\n")

    return 0 if stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Download images from Qdrant examples repository to data/<uuid>/images folder
"""

import os
import urllib.request
from pathlib import Path
from uuid import uuid4

# Image files to download
IMAGES = [
    "image-1.png",
    "image-2.png",
    "image-3.png",
    "image-4.png",
    "image-5.png",
]

BASE_URL = (
    "https://raw.githubusercontent.com/qdrant/examples/master/multimodal-search/images"
)


def download_images():
    """Download all images to data/<uuid>/images folder"""

    # Generate UUID for folder
    folder_uuid = str(uuid4())

    # Create data structure: data/<uuid>/images
    data_dir = Path("data") / folder_uuid / "images"
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading images to: {data_dir}")
    print(f"UUID: {folder_uuid}\n")

    # Download each image
    for image_name in IMAGES:
        url = f"{BASE_URL}/{image_name}"
        output_path = data_dir / image_name

        try:
            print(f"Downloading {image_name}...", end=" ")
            urllib.request.urlretrieve(url, output_path)
            file_size = os.path.getsize(output_path)
            print(f"✓ ({file_size} bytes)")
        except Exception as e:
            print(f"✗ Error: {e}")

    print(f"\n✓ All images downloaded successfully to: {data_dir}")
    return folder_uuid


if __name__ == "__main__":
    uuid_folder = download_images()

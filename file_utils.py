"""
This script contains file IO related utility functions.

Copyright (c) 2024 Hideyuki Inada
"""
import os
import logging
from typing import List

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
logger = logging.getLogger(__name__)

def get_image_file_list_in_directory(dir_path: str) -> List[str]:
    """
    Scans the specified directory for image files (jpg and png) and returns a list of their paths,
    sorted by creation time.

    Args:
        dir_path (str): The path to the directory to scan for image files.

    Returns:
        list: A list of full paths to the image files, sorted by creation time.

    Example:
        >>> get_image_file_list_in_directory("/path/to/directory")
        ['/path/to/directory/image1.png', '/path/to/directory/image2.jpg', ...]
    """
    # Scan outputs directory for generated images
    files = os.listdir(dir_path)
    # Full path, file name, creation time
    file_paths = [
        (os.path.join(dir_path, f), f, os.path.getctime(os.path.join(dir_path, f)))
        for f in files if f.lower().endswith("jpg") or f.lower().endswith("png")
    ]

    # Sort by creation date
    file_paths = sorted(file_paths, key=lambda e: e[2])
    if len(file_paths) == 0:
        logger.warn("Directory does not contain any image files")

    # Remove base name, creation time
    file_paths = [e[0] for e in file_paths]
    return file_paths

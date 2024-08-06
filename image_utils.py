"""
This script provides functions to read metadata from JPG and PNG image files, 
and to read and process images using OpenCV and Pillow.

Copyright (c) 2024 Hideyuki Inada
"""
import os
import datetime
from typing import Dict, Tuple

import cv2 as cv
from PIL import Image

def read_jpg_meta_data(file_name: str) -> Dict[str, str]:
    """
    Reads metadata from a JPG image file.

    Args:
        file_name (str): The path to the JPG image file.

    Returns:
        dict: A dictionary containing the metadata of the image.
    """
    # Open the image file
    image = Image.open(file_name)

    # Get the EXIF data
    exifdata = image.getexif()

    metadata = dict()
    # Iterate over all EXIF data fields
    for tag_id in exifdata:
        # Get the tag name, instead of the human unreadable tag id
        tag = Image.ExifTags.TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)

        # Decode bytes if necessary
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        metadata[tag] = data

    return metadata

def read_png_meta_data(file_name: str) -> Dict[str, str]:
    """
    Reads metadata from a PNG image file.

    Args:
        file_name (str): The path to the PNG image file.

    Returns:
        dict: A dictionary containing the metadata of the image.
    """
    # Open the image file
    image = Image.open(file_name)
    return image.info

def read_image(file_path: str) -> Tuple[Image.Image, str, str]:
    """
    Reads an image file, retrieves its metadata, and returns the image along with 
    a description and metadata in HTML format.

    Args:
        file_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the image, description, and metadata in HTML format.

    Example:
        >>> img, description, metadata_html = read_image("example.jpg")
        >>> print(description)
        Name: example.jpg, Time created: 2024-06-12 15:45:30, Height: 1080, Width: 1920
    """
    img = cv.imread(file_path)[:,:,::-1]  # BGR to RGB
    creation_time = os.path.getctime(file_path)

    # Convert creation time to a datetime object in UTC
    utc_time = datetime.datetime.fromtimestamp(creation_time, tz=datetime.timezone.utc)

    # Convert UTC time to local time
    local_time = utc_time.astimezone()

    # Format the local time to a human-readable string without decimal points
    local_time_str = local_time.strftime('%Y-%m-%d %H:%M:%S')
    
    # Get the row width and height
    height = img.shape[0]
    width = img.shape[1]

    # Read metadata
    metadata = read_png_meta_data(file_path) if file_path.lower().endswith(".png") else read_jpg_meta_data(file_path)
    
    # Format metadata as an HTML table
    metadata_str = "<table>"
    for k, v in metadata.items():
        metadata_str += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    metadata_str += "</table>"

    return img, f"Name: {os.path.basename(file_path)}, Time created: {local_time_str}, " + \
                f"Height: {height}, Width: {width}", metadata_str

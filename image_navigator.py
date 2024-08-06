"""
This script provides an ImageNavigator class to navigate through images in a directory.
It supports actions such as next, previous, first, last, skip forward, skip backward, and delete.

Classes:
    ImageNavigator

Copyright (c) 2024 Hideyuki Inada
"""

import os
from PIL import Image
from typing import List, Callable, Tuple, Optional

class ImageNavigator:
    """
    A class to navigate through image files in a directory.

    Methods:
        blank_image_callback() -> Tuple[Image.Image, str, str]
        update_file_paths(file_paths: List[str]) -> None
        next_button_handler(arg: None = None) -> Tuple[Image.Image, str, str]
        prev_button_handler(arg: None = None) -> Tuple[Image.Image, str, str]
        first_button_handler(arg: None = None) -> Tuple[Image.Image, str, str]
        last_button_handler(arg: None = None) -> Tuple[Image.Image, str, str]
        skip_forward_button_handler(arg: None = None) -> Tuple[Image.Image, str, str]
        skip_backward_button_handler(arg: None = None) -> Tuple[Image.Image, str, str]
        delete_button_handler(arg: None = None) -> Tuple[Image.Image, str, str]
    """

    def __init__(self, app=None, file_paths: List[str] = None, skip_num: int = 5, callback: Callable = None):
        """
        Initializes the ImageNavigator with the provided application context, file paths, skip number, and callback.

        Args:
            app: The application context.
            file_paths (List[str]): The list of image file paths.
            skip_num (int): The number of images to skip when using skip forward/backward handlers.
            callback (Callable): The callback function to execute when navigating to an image.
        """
        self.app = app
        self.callback = callback
        self.file_paths = file_paths
        self.current_index = 0
        self.skip_num = skip_num

        # Create a blank image to display when there is no file in the outputs directory
        width, height = 1024, 1024
        color = (255, 255, 255)  # RGB color for white
        self.blank_image = Image.new('RGB', (width, height), color)

    def blank_image_callback(self) -> Tuple[Image.Image, str, str]:
        """
        Returns a blank image and empty file and metadata information.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the blank image, file information, and metadata information.
        """
        return self.blank_image, "", ""

    def update_file_paths(self, file_paths: List[str], new_current_index: Optional[int]=None) -> None:
        """
        Updates the list of file paths.

        Args:
            file_paths (List[str]): The new list of image file paths.
            new_current_index (Optional[int]): The new current index.
        """
        self.file_paths = file_paths
        if new_current_index:
            self.current_index = new_current_index

    def next_button_handler(self, arg: None = None) -> Tuple[Image.Image, str, str]:
        """
        Handles the next button action to navigate to the next image.

        Args:
            arg: Optional argument.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the image, file information, and metadata information.
        """
        if len(self.file_paths) == 0:
            return self.blank_image_callback()

        if self.current_index + 1 < len(self.file_paths):
            self.current_index += 1
        else:
            self.current_index = 0

        return self.callback(self.file_paths[self.current_index])

    def prev_button_handler(self, arg: None = None) -> Tuple[Image.Image, str, str]:
        """
        Handles the previous button action to navigate to the previous image.

        Args:
            arg: Optional argument.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the image, file information, and metadata information.
        """
        if len(self.file_paths) == 0:
            return self.blank_image_callback()

        if self.current_index - 1 >= 0:
            self.current_index -= 1
        else:
            self.current_index = len(self.file_paths) - 1

        return self.callback(self.file_paths[self.current_index])

    def first_button_handler(self, arg: None = None) -> Tuple[Image.Image, str, str]:
        """
        Handles the first button action to navigate to the first image.

        Args:
            arg: Optional argument.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the image, file information, and metadata information.
        """
        if len(self.file_paths) == 0:
            return self.blank_image_callback()

        self.current_index = 0
        return self.callback(self.file_paths[self.current_index])

    def last_button_handler(self, arg: None = None) -> Tuple[Image.Image, str, str]:
        """
        Handles the last button action to navigate to the last image.

        Args:
            arg: Optional argument.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the image, file information, and metadata information.
        """
        if len(self.file_paths) == 0:
            return self.blank_image_callback()

        self.current_index = len(self.file_paths) - 1
        return self.callback(self.file_paths[self.current_index])

    def skip_forward_button_handler(self, arg: None = None) -> Tuple[Image.Image, str, str]:
        """
        Handles the skip forward button action to navigate forward by the skip number.

        Args:
            arg: Optional argument.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the image, file information, and metadata information.
        """
        if len(self.file_paths) == 0:
            return self.blank_image_callback()

        if self.current_index + self.skip_num < len(self.file_paths):
            self.current_index += self.skip_num
        else:
            self.current_index = len(self.file_paths) - 1

        return self.callback(self.file_paths[self.current_index])

    def skip_backward_button_handler(self, arg: None = None) -> Tuple[Image.Image, str, str]:
        """
        Handles the skip backward button action to navigate backward by the skip number.

        Args:
            arg: Optional argument.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the image, file information, and metadata information.
        """
        if len(self.file_paths) == 0:
            return self.blank_image_callback()

        if self.current_index - self.skip_num >= 0:
            self.current_index -= self.skip_num
        else:
            self.current_index = 0

        return self.callback(self.file_paths[self.current_index])

    def delete_button_handler(self, arg: None = None) -> Tuple[Image.Image, str, str]:
        """
        Handles the delete button action to delete the current image file.

        Args:
            arg: Optional argument.

        Returns:
            Tuple[Image.Image, str, str]: A tuple containing the next image, file information, and metadata information.
        """
        if len(self.file_paths) == 0:
            return self.blank_image_callback()

        path_to_remove = self.file_paths[self.current_index]
        print(f"Removing {path_to_remove}")
        os.remove(path_to_remove)
        self.file_paths.pop(self.current_index)

        # Tell the main app to rescan directory
        self.app.rescan_output_directory()

        # Deleted all the images
        if len(self.file_paths) == 0:
            self.current_index = 0
            return self.blank_image_callback()
        # index was the last one, but that is no longer valid
        # original indices: 0, 1, 2
        # delete image at 2
        # new indices: 0, 1
        # last valid index = 1
        elif self.current_index > len(self.file_paths) - 1:
            self.current_index -= 1
        return self.callback(self.file_paths[self.current_index])

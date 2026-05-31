import cv2
import numpy as np
from pathlib import Path


def to_grayscale(image):
    """
    Convert an RGB image to grayscale.

    Parameters
    ----------
    image : np.ndarray
        RGB image.

    Returns
    -------
    np.ndarray
        Grayscale image.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def generate_canny_sketch(image, threshold1=50, threshold2=150):
    """
    Generate a sketch-like image using Canny edge detection.

    Parameters
    ----------
    image : np.ndarray
        RGB image.
    threshold1 : int
        Lower Canny threshold.
    threshold2 : int
        Upper Canny threshold.

    Returns
    -------
    np.ndarray
        Binary edge image with white edges on black background.
    """
    gray = to_grayscale(image)
    edges = cv2.Canny(gray, threshold1, threshold2)

    return edges


def generate_sobel_sketch(image):
    """
    Generate an edge image using Sobel gradient magnitude.

    Parameters
    ----------
    image : np.ndarray
        RGB image.

    Returns
    -------
    np.ndarray
        Edge intensity image.
    """
    gray = to_grayscale(image)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    if magnitude.max() > 0:
        magnitude = 255 * magnitude / magnitude.max()

    return magnitude.astype(np.uint8)


def generate_laplacian_sketch(image):
    """
    Generate an edge image using the Laplacian operator.

    Parameters
    ----------
    image : np.ndarray
        RGB image.

    Returns
    -------
    np.ndarray
        Edge intensity image.
    """
    gray = to_grayscale(image)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.absolute(laplacian)

    if laplacian.max() > 0:
        laplacian = 255 * laplacian / laplacian.max()

    return laplacian.astype(np.uint8)
    


def save_sketch(sketch, output_path):
    """
    Save a sketch image to disk.

    Parameters
    ----------
    sketch : np.ndarray
        Sketch or edge image.
    output_path : str or Path
        Output file path.

    Returns
    -------
    Path
        Path to the saved sketch.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path), sketch)

    return output_path
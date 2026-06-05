import cv2
import numpy as np
from skimage.feature import hog


def load_grayscale_image(path):
    """
    Load an image from disk as grayscale

    Parameters
    ----------
    path: str or Path
        Path to the image file

    Returns
    -------
    np.ndarray or None
        Grayscale image array, or None if loading fails
    
    """
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None
    
    return image


def extract_hog_features(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False,
):
    """
    Extract HOG features from graysclale image

    Parameters
    ----------
    image: np.ndarray
        Graysclae image
    orientations: int
        Number of orientation bins
    pixels_per_cells: tuple
        Size of each cell in pixels
    cells_per_block: tuple
        Number of cells in each normalization block
    visualize: bool
        If True, also return a HIG visualization image


    Returns
    -------
    np.ndarray or tuple
        HOG feature vector
        If visualize=True, returns (features, hog_image)
    """


    return hog(
        image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        block_norm="L2-Hys",
        visualize=visualize,
        feature_vector=True,

    )
    

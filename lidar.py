import laspy
import numpy as np
from numpy.random import normal
import matplotlib.pyplot as plt

from MvFIF_v8 import MvFIF
from datetime import datetime

from sklearn.neighbors import KDTree


def load_lidar_data(file_path):
    """
    Loads LiDAR data from a LAS/LAZ file using laspy.

    Args:
        file_path (str): The path to the LiDAR file.

    Returns:
        tuple: A tuple containing:
            - points (ndarray): The X, Y, Z coordinates of LiDAR points as a (num_points, 3) array.
            - las_data (laspy.LasData): The full LiDAR data object.
    """
    print(f"Loading LiDAR data from {file_path}...")
    try:
        with laspy.open(file_path) as lidar_file:
            las_data = lidar_file.read()
            points = np.vstack([las_data.x, las_data.y, las_data.z]).T
    except Exception as e:
        raise RuntimeError(f"Failed to load LiDAR data: {e}")

    return points, las_data


def classify_ground_points(las, IMF, las_data, min_threshold=None, max_threshold=None, intensity_threshold=60):
    """
    Classifies ground and non-ground points based on the residual from MvFIF.

    Args:
        las (ndarray): LiDAR points (X, Y, Z).
        IMF (ndarray): IMF from MvFIF.
        threshold (float, optional): Threshold for elevation difference to distinguish ground from non-ground.
            If None, computed automatically.

    Returns:
        ndarray: Array of classifications (1 for non-ground, 2 for ground).
    """
    num_points = las.shape[0]

    # Extract the residual (trend) for Z-coordinate
    high_freq = IMF[1]
    high_freq_z = high_freq[2, :]  # Shape: (num_points,)

    normalized_z = normalize_z(las)
    flattened_z = IMF[-1, 2, :]
    # Compute the absolute difference between original Z and residual Z
    elevation_difference = np.abs(normalized_z - high_freq_z)
    intensity = las_data.intensity

    # Compute threshold if not provided
    if min_threshold is None:
        # Set threshold as mean plus standard deviation
        min_threshold = elevation_difference.mean() - 1.5*elevation_difference.std()
        max_threshold = elevation_difference.mean() + elevation_difference.std()
    print(f"Flattened Z stats: min={flattened_z.min()}, max={flattened_z.max()}, mean={flattened_z.mean()}, std={flattened_z.std()}")
    print(f"Elevation difference stats: min={elevation_difference.min()}, max={elevation_difference.max()}, mean={elevation_difference.mean()}, std={elevation_difference.std()}")
    print(f"Using min threshold={min_threshold} for classification.")

    # Classify points: 2 for ground (difference less than threshold), 1 for non-ground
    flattened_thresh = np.percentile(flattened_z, 95)
    classification = np.ones(num_points, dtype=np.uint8)  # Default to non-ground
    ground_mask = (elevation_difference >= min_threshold) & (elevation_difference <= max_threshold) & (intensity > intensity_threshold) & (flattened_z <= flattened_thresh)
    classification[ground_mask] = 2  #Label ground points
    #classification = classify_by_relative_elevation(las, classification)

    return classification


def normalize_z(las):
    """
    Normalize the Z values of LiDAR points to the range [0, 1].

    Args:
        las (ndarray): LiDAR points (X, Y, Z).

    Returns:
        ndarray: Normalized Z values.
    """
    z_min = np.min(las[:, 2])
    z_max = np.max(las[:, 2])

    # Normalize Z between 0 and 1
    normalized_z = (las[:, 2] - z_min) / (z_max - z_min)
    return normalized_z

def save_classified_las(las_data, classification, output_file):
    """
    Saves classified LiDAR data into a new LAS file.

    Args:
        las_data (laspy.LasData): The original LAS data.
        classification (ndarray): Ground/non-ground classification for each point.
        output_file (str): Output file path for the classified LAS data.
    """
    print(f"Saving classified LiDAR data to {output_file}...")

    try:
        new_las = laspy.LasData(las_data.header)
        new_las.points = las_data.points.copy()
        new_las.classification = classification
        new_las.write(output_file)
    except Exception as e:
        raise RuntimeError(f"Failed to save classified LAS data: {e}")

    print(f"Classified LAS file successfully saved as {output_file}.")




def main(lidar_file_path, output_file):
    """
    Main function to process and classify LiDAR points as ground/non-ground.

    Args:
        lidar_file_path (str): Path to the input LiDAR file.
        output_file (str): Path to save the classified output LAS file.
    """
    # Load LiDAR data
    las, las_data = load_lidar_data(lidar_file_path)

    # Parameters for MvFIF
    delta = 0.01
    alpha = 0.025
    NumSteps = 10
    ExtPoints = 5
    NIMFs = 2 # Number of IMFs to extract
    MaxInner = 100

    # Run MvFIF to compute IMFs and residual
    print("Running MvFIF on the LiDAR data...")
    start_time = datetime.now()
    IMF, _ = MvFIF(las.T, delta, alpha, NumSteps, ExtPoints, NIMFs, MaxInner)

    if IMF is None:
        print("No IMFs were extracted. Exiting...")
        return

    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"MvFIF completed in {elapsed_time:.2f} seconds")

    # Classify ground and non-ground points using the residual
    classification = classify_ground_points(las, IMF, las_data)

    #finished_classification = classify_by_relative_elevation(las, classification)

    # Save the classified points
    save_classified_las(las_data, classification, output_file)


if __name__ == "__main__":
    lidar_file_path = 'golf.las'  # Input LiDAR file
    output_file = 'classified_golf.las'  # Output classified LAS file
    main(lidar_file_path, output_file)

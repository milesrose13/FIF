import laspy
import numpy as np
from MvFIF_v8 import MvFIF
from datetime import datetime


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


def classify_ground_points(las, imfs, num_low_freq_imfs=None, num_high_freq_imfs=1, threshold=None):
    """
    Classifies ground and non-ground points based on the ratio of low-frequency to high-frequency energy.

    Args:
        las (ndarray): LiDAR points (X, Y, Z).
        imfs (ndarray): IMFs from MvFIF.
        num_low_freq_imfs (int, optional): Number of low-frequency IMFs to consider.
            If None, uses all IMFs except the specified high-frequency ones.
        num_high_freq_imfs (int, optional): Number of high-frequency IMFs to consider.
            Defaults to 1.
        threshold (float, optional): Threshold for the energy ratio.
            If None, computed automatically.

    Returns:
        ndarray: Array of classifications (1 for non-ground, 2 for ground).
    """
    total_imfs = imfs.shape[0]
    if num_low_freq_imfs is None:
        num_low_freq_imfs = total_imfs - num_high_freq_imfs

    # Ensure that the number of IMFs is sufficient
    if num_low_freq_imfs <= 0 or num_high_freq_imfs <= 0:
        raise ValueError("Invalid number of high-frequency or low-frequency IMFs specified.")

    # Extract high-frequency IMFs
    high_freq_imfs = imfs[:num_high_freq_imfs, :, :]  # Shape: (num_high_freq_imfs, 3, num_points)

    # Extract low-frequency IMFs
    low_freq_imfs = imfs[-num_low_freq_imfs:, :, :]  # Shape: (num_low_freq_imfs, 3, num_points)

    # Compute energy of high-frequency IMFs for each point
    high_freq_energy = np.sum(high_freq_imfs ** 2, axis=(0, 1))  # Shape: (num_points,)

    # Compute energy of low-frequency IMFs for each point
    low_freq_energy = np.sum(low_freq_imfs ** 2, axis=(0, 1))  # Shape: (num_points,)

    # Compute the ratio of low-frequency energy to high-frequency energy
    # Adding a small epsilon to the denominator to avoid division by zero
    epsilon = 1e-8
    energy_ratio = low_freq_energy / (high_freq_energy + epsilon)

    # If no threshold is provided, compute it automatically
    if threshold is None:
        # For example, set threshold as the median of the energy ratios
        threshold = np.median(energy_ratio)

    # Classify points: 2 for ground (high energy ratio), 1 for non-ground
    classification = np.ones(las.shape[0], dtype=np.uint8)
    ground_mask = energy_ratio > threshold
    classification[ground_mask] = 2

    return classification



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
    alpha = 0.5
    NumSteps = 10
    ExtPoints = 5
    NIMFs = 5
    MaxInner = 100

    # Run MvFIF to compute IMFs
    print("Running MvFIF on the LiDAR data...")
    start_time = datetime.now()
    IMF, _ = MvFIF(las.T, delta, alpha, NumSteps, ExtPoints, NIMFs, MaxInner)

    if IMF is None:
        print("No IMFs were extracted. Exiting...")
        return

    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"MvFIF completed in {elapsed_time:.2f} seconds")

    # Classify ground and non-ground points
    classification = classify_ground_points(las, IMF)

    # Save the classified points
    save_classified_las(las_data, classification, output_file)


if __name__ == "__main__":
    lidar_file_path = 'golf.las'  # Input LiDAR file
    output_file = 'classified_golf.las'  # Output classified LAS file
    main(lidar_file_path, output_file)

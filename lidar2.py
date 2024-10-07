import laspy
import numpy as np
from numpy.random import shuffle

from FIF import FIF
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
    except Exception as e:
        raise RuntimeError(f"Failed to load LiDAR data: {e}")

    return las_data


import laspy

import laspy


def save_classified_las(las_data, zIMF, output_file):
    """
    Saves LiDAR data into a new LAS file with updated Z-coordinates (zIMF).

    Args:
        las_data (laspy.LasData): The original LAS data.
        zIMF (ndarray): The modified Z-coordinates.
        output_file (str): Path to save the output LAS file.
    """
    print(f"Saving modified LiDAR data to {output_file}...")

    try:
        # Open a new LAS file for writing with the original header
        with laspy.open(output_file, mode="w", header=las_data.header) as writer:
            # Update the Z-coordinates in the points
            las_data.z = zIMF

            # Write the updated points to the new file
            writer.write_points(las_data.points)

    except Exception as e:
        raise RuntimeError(f"Failed to save modified LAS data: {e}")

    print(f"Modified LAS file successfully saved as {output_file}.")


def main(lidar_file_path, output_file):
    """
    Main function to process and classify LiDAR points as ground/non-ground.

    Args:
        lidar_file_path (str): Path to the input LiDAR file.
        output_file (str): Path to save the classified output LAS file.
    """
    # Load LiDAR data
    las_data = load_lidar_data(lidar_file_path)

    # Access Z-coordinates properly
    z = las_data.z

    # Parameters for MvFIF
    delta = 0.1
    alpha = 0.5
    num_steps = 10
    ext_points = 5
    n_imfs = 5  # Number of IMFs to extract
    max_inner = 80

    # Run MvFIF to compute IMFs and residual
    print("Running FIF on the LiDAR data...")
    start_time = datetime.now()
    IMF, _ = FIF(z, delta, alpha, num_steps, ext_points, n_imfs, max_inner)
    IMF_4 =IMF[3]
    print(f"IMF2 stats: min={IMF_4.min()}, max={IMF_4.max()}, mean={IMF_4.mean()}, std={IMF_4.std()}")

    if IMF is None:
        print("No IMFs were extracted. Exiting...")
        return

    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"FIF completed in {elapsed_time:.2f} seconds")

    # Classify ground and non-ground points using the residual

    # Save the classified points
    for i in range(5):
        imf_i = IMF[i]  # Select the i-th IMF
        output_file_i = f"{i}_{output_file}"  # Prepend the index to the output filename
        save_classified_las(las_data, imf_i, output_file_i)


if __name__ == "__main__":
    lidar_file_path = 'golf.las'  # Input LiDAR file
    output_file = 'IMF_golf.las'  # Output classified LAS file
    main(lidar_file_path, output_file)

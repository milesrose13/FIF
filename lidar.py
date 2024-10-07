import laspy
import numpy as np
from datetime import datetime
from FIF import FIF


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


def classify_points(IMF, intensity, offset):
    num_points = len(intensity)
    classification = np.ones(num_points, dtype=np.uint8)  # Default to non-ground

    # Thresholds (adjust as needed)
    INT_threshold = 15
    IMF1_threshold = np.percentile(IMF[0], 50+offset)
    IMF2_threshold = np.percentile(IMF[1], 50-offset)
    IMF3_threshold = np.percentile(IMF[2], 50+offset)
    IMF4_threshold = np.percentile(IMF[3], 50+offset)

    # Create masks
    IMF1_mask = IMF[0] <= IMF1_threshold
    IMF2_mask = IMF[1] >= IMF2_threshold
    IMF3_mask = IMF[2] <= IMF3_threshold
    IMF4_mask = IMF[3] <= IMF4_threshold
    INT_mask = intensity >= INT_threshold

    # Combined ground mask
    ground_mask = IMF1_mask & IMF2_mask & IMF3_mask & IMF4_mask & INT_mask

    # Update classification
    classification[ground_mask] = 2  # Ground points

    return classification

def main(input_las_path, output_file, ground_output_las_path):
    # Load the LiDAR data
    las = laspy.read(input_las_path)
    z = las.z
    intensity = las.intensity

    # Parameters for FIF (adjust as needed)
    delta = 0.1
    alpha = 0.5
    num_steps = 10
    ext_points = 5
    n_imfs = 5
    max_inner = 80

    # Step 1: Run FIF on the entire dataset
    print("Running FIF on the entire dataset...")
    IMF_full, _ = FIF(z, delta, alpha, num_steps, ext_points, n_imfs, max_inner)

    # Step 2: Classify all points
    classification_full = classify_points(IMF_full, intensity, 35)

    # Step 3: Extract ground points from the first classification
    ground_mask_first = classification_full == 2
    ground_z = z[ground_mask_first]
    ground_intensity = intensity[ground_mask_first]

    # Step 4: Run FIF on ground points only
    print("Running FIF on ground points...")
    IMF_ground, _ = FIF(ground_z, delta, alpha, num_steps, ext_points, n_imfs, max_inner)

    # Step 5: Classify ground points again
    classification_ground = classify_points(IMF_ground, ground_intensity, 30)

    # Step 6: Update the classification in the full dataset
    ground_indices_first = np.where(ground_mask_first)[0]
    classification_full[ground_indices_first] = classification_ground

    # Step 7: Extract final ground points based on the second classification
    final_ground_mask = classification_full == 2
    ground_points = las.points[final_ground_mask]

    # Step 8: Extract non-ground points from the first classification
    non_ground_mask = classification_full == 1
    non_ground_points = las.points[non_ground_mask]

    # Step 9: Create a new LAS file with non-ground points
    new_las = laspy.LasData(header=las.header)
    new_las.points = non_ground_points.copy()
    new_las.write(output_file)
    print(f"Non-ground LAS file saved as {output_file}")

    # Step 10: Append ground points to the new LAS file using LasAppender
    with laspy.open(output_file, mode="a") as las_append:
        las_append.append_points(ground_points)

    print(f"Combined LAS file saved as {output_file}")



if __name__ == "__main__":
    lidar_file_path = 'downtown_wack.las'  # Input LiDAR file
    output_file = 'classified_downtown_wack.las'  # Output classified LAS file
    ground_only_output_file = 'ground_downtown_wack.las'
    main(lidar_file_path, output_file, ground_only_output_file)

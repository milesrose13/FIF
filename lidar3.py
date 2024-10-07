import laspy
import numpy as np
from datetime import datetime
from FIF import FIF

def load_lidar_data(file_path):
    print(f"Loading LiDAR data from {file_path}...")
    try:
        with laspy.open(file_path) as lidar_file:
            las_data = lidar_file.read()
            points = np.vstack([las_data.x, las_data.y, las_data.z]).T
    except Exception as e:
        raise RuntimeError(f"Failed to load LiDAR data: {e}")
    return points, las_data

def classify_ground_points(IMF_1, IMF_2, IMF_3, intensity):
    INT_threshold = 15
    IMF1_threshold = np.percentile(IMF_1, 75)
    IMF2_threshold = np.percentile(IMF_2, 15)
    IMF3_threshold = np.percentile(IMF_3, 75)

    IMF1_mask = IMF_1 <= IMF1_threshold
    IMF2_mask = IMF_2 >= IMF2_threshold
    IMF3_mask = IMF_3 <= IMF3_threshold
    INT_mask = intensity >= INT_threshold

    ground_mask = IMF1_mask & IMF2_mask & IMF3_mask & INT_mask
    return ground_mask.astype(np.uint8) * 2 + (1 - ground_mask).astype(np.uint8)

def save_classified_las(las_data, classification, output_file):
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
    las, las_data = load_lidar_data(lidar_file_path)
    z = las_data.z

    # Optimized Parameters for FIF
    delta = 0.05
    alpha = 0.5
    num_steps = 5
    ext_points = 3
    n_imfs = 4  # Reduced number of IMFs for faster performance
    max_inner = 50  # Reduced for faster convergence

    print("Running FIF on the LiDAR data...")
    start_time = datetime.now()
    IMF, _ = FIF(z, delta, alpha, num_steps, ext_points, n_imfs, max_inner)

    if IMF is None:
        print("No IMFs were extracted. Exiting...")
        return

    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"FIF completed in {elapsed_time:.2f} seconds")

    # Classify ground and non-ground points using IMF variance and intensity
    classification = classify_ground_points(IMF[0], IMF[1], IMF[2], las_data.intensity)

    save_classified_las(las_data, classification, output_file)

if __name__ == "__main__":
    lidar_file_path = 'little_mountain.las'
    output_file = 'classified_little_mountain.las'
    main(lidar_file_path, output_file)


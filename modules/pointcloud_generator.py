import numpy as np
import cv2
import open3d as o3d
from PIL import Image


def load_depth_map(file_path):
    try:
        with Image.open(file_path) as img:
            depth_map = np.array(img)
            if len(depth_map.shape) > 2:
                depth_map = np.mean(depth_map, axis=2).astype(np.uint8)
            
            height, width = depth_map.shape
            size = min(height, width)
            
            start_y = (height - size) // 2
            start_x = (width - size) // 2
            depth_map = depth_map[start_y:start_y+size, start_x:start_x+size]
            
            return depth_map.astype(np.float32) / 255.0  # Normalize to [0,1]
    except Exception as e:
        print(f"Failed to load: {file_path}")
        print(f"Error: {str(e)}")
        return None


def create_mask_from_depth(depth_map, threshold_low=0.1, threshold_high=0.9):
    mask = (depth_map > threshold_low) & (depth_map < threshold_high)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_uint8 = mask.astype(np.uint8) * 255
    
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
    
    if num_labels > 1:
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_uint8 = (labels == largest_component).astype(np.uint8) * 255
    
    return mask_uint8 > 0


def create_point_cloud_from_depth(depth_map, view):
    if depth_map is None:
        return None
    
    mask = create_mask_from_depth(depth_map, threshold_low=0.2, threshold_high=0.95)
    
    size = depth_map.shape[0]
    y, x = np.mgrid[0:size, 0:size]
    
    step = 1
    x = x[::step, ::step]
    y = y[::step, ::step]
    depth_map = depth_map[::step, ::step]
    mask = mask[::step, ::step]
    
    x = x - size/2
    y = y - size/2
    
    scale = 100
    
    if view == "front":
        points = np.stack([x, -y, depth_map * scale], axis=-1)
    elif view == "right":
        points = np.stack([depth_map * scale * 2.5, -y, -x], axis=-1)
    elif view == "left":
        points = np.stack([-depth_map * scale * 2.5, -y, x], axis=-1)
    elif view == "back":
        points = np.stack([-x, -y, -depth_map * scale], axis=-1)

    valid_points = points[mask]
    
    valid_depths = depth_map[mask]
    depth_threshold = 0.01
    final_valid_mask = valid_depths > depth_threshold
    valid_points = valid_points[final_valid_mask]
    
    print(f"{view} view: {np.sum(mask)} points before mask, {len(valid_points)} points after")
    
    if len(valid_points) > 100000:
        indices = np.random.choice(len(valid_points), 100000, replace=False)
        valid_points = valid_points[indices]
    
    if len(valid_points) < 100:
        print(f"Warning: Too few valid points in {view} view ({len(valid_points)} points)")
        return None
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    colors = {
        "front": [1, 0, 0],
        "right": [0, 1, 0],
        "left": [0, 0, 1],
        "back": [1, 1, 0]
    }
    
    pcd.paint_uniform_color(colors[view])
    
    print(f"  {view} view outlier removal started...")
    if len(valid_points) > 500:
        pcd = remove_noise_from_pointcloud(pcd, method="statistical", verbose=True)
    else:
        print(f"  Skipping outlier removal due to few points: {len(valid_points)} points")
    
    return pcd


def remove_noise_from_pointcloud(pcd, method="statistical", verbose=True):
    if pcd is None or len(pcd.points) == 0:
        return pcd
    
    original_count = len(pcd.points)
    cleaned_pcd = pcd
    
    if method in ["statistical", "all"]:
        cl, ind = cleaned_pcd.remove_statistical_outlier(
            nb_neighbors=10,
            std_ratio=3.0
        )
        cleaned_pcd = cl
        if verbose:
            stat_removed = original_count - len(cleaned_pcd.points)
            print(f"  Statistical outlier removal: {stat_removed} extreme outliers removed")
    
    return cleaned_pcd


def preprocess_for_icp(pcd, aggressive=False):
    if pcd is None:
        return None
    
    print(f"  ICP preprocessing started: {len(pcd.points)} points")
    
    if aggressive:
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=8, std_ratio=1.5)
        pcd = cl
    else:
        pcd = remove_noise_from_pointcloud(pcd, method="statistical", verbose=False)
    
    if len(pcd.points) > 50:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=15)
    
    print(f"  ICP preprocessing complete: {len(pcd.points)} points")
    return pcd
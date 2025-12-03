import numpy as np
import os
import cv2
import open3d as o3d
from modules.pointcloud_generator import (
    load_depth_map, 
    create_point_cloud_from_depth,
    create_mask_from_depth
)
from modules.fpfh_alignment import align_point_clouds_fpfh


def process_depth_maps(views_dict, debug_save=True, debug_dir="output/debug"):
    """
    Process depth maps to generate point clouds.
    
    Args:
        views_dict (dict): Dictionary of depth map file paths by view
        debug_save (bool): Whether to save debug files
        debug_dir (str): Debug file save directory
        
    Returns:
        dict: Dictionary of point clouds by view
    """
    point_clouds = {}
    
    for view_name, file_path in views_dict.items():
        print(f"\nProcessing {view_name} view...")
        depth_map = load_depth_map(file_path)
        
        if depth_map is not None:
            # Save mask for debugging
            if debug_save:
                mask = create_mask_from_depth(depth_map, threshold_low=0.2, threshold_high=0.95)
                
                os.makedirs(debug_dir, exist_ok=True)
                mask_path = os.path.join(debug_dir, f"{view_name}_mask.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                print(f"Mask saved: {mask_path}")
            
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    return point_clouds


def align_point_clouds(point_clouds, use_point_to_plane_icp=True):
    """
    Align point clouds
    
    Args:
        point_clouds (dict): Dictionary of point clouds by view
        use_point_to_plane_icp (bool): True for Point-to-Plane ICP, False for Point-to-Point ICP
    
    Returns:
        tuple: (aligned_clouds, view_names) - List of aligned point clouds and view names
    """
    icp_mode = "Point-to-Plane" if use_point_to_plane_icp else "Point-to-Point"
    print(f"\n=== FPFH-based Point Cloud Alignment ({icp_mode} ICP) ===")
    
    # Start alignment with front view as reference
    aligned_clouds = [point_clouds["front"]]
    front_target = point_clouds["front"]
    view_names = ["front"]
    
    # Align left and right views to front
    left_aligned = None
    right_aligned = None
    
    if "left" in point_clouds:
        print("\nAligning left view to front...")
        params_align = {
            'voxel_coarse': 5.0,
            'voxel_list': [20.0, 10.0, 5.0],
            'ransac_iter': 20000,
            'use_cpd': False,
            'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
            'fitness_threshold_accept': 0.02,
            'force_cpd': False,
            'allow_rotation': False,
            'allow_small_rotation': True,
            'use_point_to_plane_icp': use_point_to_plane_icp
        }
        left_aligned = align_point_clouds_fpfh(point_clouds["left"], front_target, params=params_align)
        aligned_clouds.append(left_aligned)
        view_names.append("left")
    
    if "right" in point_clouds:
        print("\nAligning right view to front...")
        params_align = {
            'voxel_coarse': 3.0,  # Restored to 3.0: more global RANSAC features
            'voxel_list': [10.0, 5.0, 2.5, 1.0],  # Very fine 4-stage multi-scale
            'ransac_iter': 100000,  # Maintain high RANSAC iteration count
            'use_cpd': False,
            'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
            'fitness_threshold_accept': 0.02,
            'force_cpd': False,
            'allow_rotation': True,  # Right view may need rotation
            'allow_small_rotation': True,
            'use_point_to_plane_icp': use_point_to_plane_icp
        }
        right_aligned = align_point_clouds_fpfh(point_clouds["right"], front_target, params=params_align)
        aligned_clouds.append(right_aligned)
        view_names.append("right")
    
    # Align back view only to left/right point clouds (alignment results)
    if "back" in point_clouds:
        print("\nAligning back view to left/right accumulated cloud...")
        side_target = o3d.geometry.PointCloud()
        st_points = []
        st_colors = []

        # Left alignment result
        if left_aligned is not None and len(left_aligned.points) > 0:
            st_points.extend(np.asarray(left_aligned.points))
            if left_aligned.has_colors():
                st_colors.extend(np.asarray(left_aligned.colors))

        # Right alignment result
        if right_aligned is not None and len(right_aligned.points) > 0:
            st_points.extend(np.asarray(right_aligned.points))
            if right_aligned.has_colors():
                st_colors.extend(np.asarray(right_aligned.colors))

        if len(st_points) == 0:
            print("  Left/right target is empty, skipping back alignment.")
        else:
            side_target.points = o3d.utility.Vector3dVector(np.array(st_points))
            if len(st_colors) == len(st_points) and len(st_colors) > 0:
                side_target.colors = o3d.utility.Vector3dVector(np.array(st_colors))

            # Align back to left/right target
            params_align = {
                'voxel_coarse': 5.0,
                'voxel_list': [25.0, 12.0, 6.0],
                'ransac_iter': 30000,
                'use_cpd': False,
                'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
                'fitness_threshold_accept': 0.02,
                'force_cpd': False,
                'allow_rotation': False,
                'allow_small_rotation': True,
                'use_point_to_plane_icp': use_point_to_plane_icp
            }
            back_aligned = align_point_clouds_fpfh(point_clouds["back"], side_target, params=params_align)
            aligned_clouds.append(back_aligned)
            view_names.append("back")
    
    return aligned_clouds, view_names


def merge_and_clean_pointclouds(aligned_clouds):
    """
    Merge aligned point clouds and remove outliers.
    
    Args:
        aligned_clouds (list): List of aligned point clouds
        
    Returns:
        o3d.geometry.PointCloud: Merged and cleaned point cloud
    """
    print(f"\n=== Final Merge and Outlier Removal ===")
    
    # Merge all point clouds into one
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in aligned_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    print(f"Merged point count: {len(merged_cloud.points)}")
    
    # Remove only extreme outliers (minimize downsampling)
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.5)  # Very lenient criteria
    merged_cloud = cl
    print(f"After extreme outlier removal: {len(merged_cloud.points)} points")
    
    # Recalculate normal vectors
    merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    return merged_cloud

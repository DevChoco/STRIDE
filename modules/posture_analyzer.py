import numpy as np
import open3d as o3d
from modules.skeleton_parser import (
    detect_landmarks_with_ai,
    create_skeleton_from_pointcloud,
    calculate_spine_angles,
    create_skeleton_visualization,
    print_angles
)


def analyze_posture(merged_cloud, front_image_path):
    print("\n=== Skeleton Generation and Posture Analysis ===")
    
    front_landmarks = detect_landmarks_with_ai(front_image_path)
    
    if front_landmarks:
        print("AI landmark detection successful! Generating accurate skeleton reflecting individual body features.")
        for name, landmark in front_landmarks.items():
            print(f"  {name}: x={landmark['x']:.1f}, y={landmark['y']:.1f}, visibility={landmark['visibility']:.3f}")
    else:
        print("AI landmark detection failed, using default anatomical proportions.")
    skeleton_points = create_skeleton_from_pointcloud(merged_cloud, front_landmarks)
    angles = calculate_spine_angles(skeleton_points)
    skeleton_pcd, skeleton_cylinders = create_skeleton_visualization(skeleton_points)
    
    print_angles(angles)
    
    return skeleton_points, angles, skeleton_pcd, skeleton_cylinders


def analyze_posture_from_lod_meshes(lod_meshes, front_image_path):
    print("\n=== Multi-Resolution Ensemble Skeleton Prediction ===")
    
    front_landmarks = detect_landmarks_with_ai(front_image_path)
    
    if front_landmarks:
        print("AI landmark detection successful! Reflecting individual body features.")
    else:
        print("AI landmark detection failed, using default anatomical proportions.")
    all_predictions = {}
    lod_order = ["ultra_low", "low", "medium", "default", "high", "ultra_high"]
    
    for lod_level in lod_order:
        if lod_level not in lod_meshes or lod_meshes[lod_level] is None:
            print(f"  ⚠ {lod_level} LOD mesh not found. Skipping.")
            continue
        
        print(f"\n  [{lod_level.upper()} LOD] Predicting skeleton...")
        mesh = lod_meshes[lod_level]
        
        pcd = mesh.sample_points_uniformly(number_of_points=50000)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        
        skeleton_points = create_skeleton_from_pointcloud(pcd, front_landmarks)
        angles = calculate_spine_angles(skeleton_points)
        
        all_predictions[lod_level] = {
            'skeleton_points': skeleton_points,
            'angles': angles
        }
        print(f"    ✓ {lod_level} LOD prediction complete: {len(skeleton_points)} joints")
    
    print("\n=== Median-based Optimal Skeleton Selection ===")
    
    if not all_predictions:
        raise ValueError("No prediction results.")
    joint_names = set()
    for pred in all_predictions.values():
        joint_names.update(pred['skeleton_points'].keys())
    
    voted_skeleton_points = {}
    
    for joint_name in joint_names:
        joint_predictions = []
        for pred in all_predictions.values():
            if joint_name in pred['skeleton_points'] and pred['skeleton_points'][joint_name] is not None:
                joint_predictions.append(pred['skeleton_points'][joint_name])
        
        if not joint_predictions:
            voted_skeleton_points[joint_name] = None
            continue
        
        joint_predictions = np.array(joint_predictions)
        median_point = np.median(joint_predictions, axis=0)
        distances = np.linalg.norm(joint_predictions - median_point, axis=1)
        closest_idx = np.argmin(distances)
        voted_skeleton_points[joint_name] = joint_predictions[closest_idx]
    angle_names = set()
    for pred in all_predictions.values():
        if pred['angles']:
            angle_names.update(pred['angles'].keys())
    
    voted_angles = {}
    for angle_name in angle_names:
        angle_predictions = []
        for pred in all_predictions.values():
            if pred['angles'] and angle_name in pred['angles'] and pred['angles'][angle_name] is not None:
                angle_predictions.append(pred['angles'][angle_name])
        
        if not angle_predictions:
            voted_angles[angle_name] = None
            continue
        
        angle_predictions = np.array(angle_predictions)
        median_angle = np.median(angle_predictions)
        distances = np.abs(angle_predictions - median_angle)
        closest_idx = np.argmin(distances)
        voted_angles[angle_name] = angle_predictions[closest_idx]
    
    skeleton_pcd, skeleton_cylinders = create_skeleton_visualization(voted_skeleton_points)
    
    print("\nFinal voting results:")
    print(f"  Total joint count: {len(voted_skeleton_points)}")
    print(f"  Participating LOD models: {len(all_predictions)}")
    print_angles(voted_angles)
    
    return voted_skeleton_points, voted_angles, skeleton_pcd, skeleton_cylinders, all_predictions, None

import numpy as np
import os
import cv2
import open3d as o3d
import copy
import json

from modules.pointcloud_processor import (
    process_depth_maps,
    align_point_clouds,
    merge_and_clean_pointclouds
)
from modules.posture_analyzer import (
    analyze_posture,
    analyze_posture_from_lod_meshes
)
from modules.skeleton_parser import (
    create_skeleton_visualization
)
from modules.data_exporter import (
    save_skeleton_data,
    generate_xray_snapshot
)
from modules.visualization import (
    visualize_results
)
from modules.mesh_generator import create_and_save_mesh
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, rc

try:
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    else:
        font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/NanumGothic.ttf").get_name()
        rc('font', family=font_name)
except:
    try:
        rc('font', family='Malgun Gothic')
    except:
        pass

plt.rcParams['axes.unicode_minus'] = False


def main():
    """Main execution function"""
    print("="*60)
    print("     Modular 3D Posture Analysis System")
    print("     FPFH Alignment + Skeleton Parsing")
    print("="*60)
    
    USE_POINT_TO_PLANE_ICP = True
    
    print(f"\n[ICP Mode Configuration]")
    if USE_POINT_TO_PLANE_ICP:
        print("Using Point-to-Plane ICP")
    else:
        print("Using Point-to-Point ICP")
        
    nn = "16"
    gender = "female"

    views = {
        "front": rf"Data\DepthMap\{gender}_front\DepthMap{nn}.bmp",
        "right": rf"Data\DepthMap\{gender}_R\DepthMap{nn}.bmp",
        "left": rf"Data\DepthMap\{gender}_L\DepthMap{nn}.bmp",
        "back": rf"Data\DepthMap\{gender}_back\DepthMap{nn}.bmp"
    }
    
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output", "3d_models")
        debug_dir = os.path.join(script_dir, "output", "debug")
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        
        print(f"\n[Output Path Configuration]")
        print(f"3D model save path: {output_dir}")
        print(f"Debug file path: {debug_dir}")
        
        point_clouds = process_depth_maps(views, debug_save=True, debug_dir=debug_dir)
        
        if not point_clouds:
            print("Point cloud generation failed. Exiting program.")
            return
        
        aligned_clouds, view_names = align_point_clouds(point_clouds, use_point_to_plane_icp=USE_POINT_TO_PLANE_ICP)
        
        merged_cloud = merge_and_clean_pointclouds(aligned_clouds)
        
        print("\n=== Mesh Generation and LOD Optimization ===")
        print("Converting point cloud to high-quality mesh and generating multi-resolution LODs...")
        
        try:
            mesh, saved_files = create_and_save_mesh(
                merged_cloud, 
                output_dir,
                "body_mesh_fpfh",
                create_lod=True,
                reduction_ratio=0.2,
                optimization_level="high_quality",
                enable_quality_analysis=True,
                enable_hole_filling=False
            )
        except TypeError:
            print("Switching to basic mesh generation mode...")
            mesh, saved_files = create_and_save_mesh(merged_cloud, output_dir, "body_mesh_fpfh")
        
        if saved_files:
            print(f"\nMesh files saved:")
            for file_path in saved_files:
                print(f"  {file_path}")
        
        print("\n=== Loading LOD Meshes ===")
        lod_meshes = {}
        lod_levels = ["ultra_low", "low", "medium", "default", "high", "ultra_high"]
        
        for lod_level in lod_levels:
            if lod_level == "default":
                mesh_path = os.path.join(output_dir, "body_mesh_fpfh.obj")
            else:
                mesh_path = os.path.join(output_dir, f"body_mesh_fpfh_{lod_level}.obj")
            
            if os.path.exists(mesh_path):
                try:
                    loaded_mesh = o3d.io.read_triangle_mesh(mesh_path)
                    if len(loaded_mesh.vertices) > 0:
                        lod_meshes[lod_level] = loaded_mesh
                        print(f"  ✓ {lod_level} LOD loaded: {len(loaded_mesh.vertices)} vertices")
                    else:
                        print(f"  ⚠ {lod_level} LOD mesh is empty")
                except Exception as e:
                    print(f"  ✗ {lod_level} LOD load failed: {e}")
            else:
                print(f"  ⚠ {lod_level} LOD file not found: {mesh_path}")
        
        if len(lod_meshes) >= 3:
            print(f"\nMulti-resolution ensemble mode: Using {len(lod_meshes)} LOD models")
            skeleton_points, angles, skeleton_pcd, skeleton_cylinders, all_predictions, _ = analyze_posture_from_lod_meshes(
                lod_meshes, views["front"]
            )
        else:
            print(f"\nSingle mode: Using basic point cloud due to insufficient LOD models")
            skeleton_points, angles, skeleton_pcd, skeleton_cylinders = analyze_posture(
                merged_cloud, views["front"]
            )
            all_predictions = None
        
        print("\n=== Saving Skeleton Data ===")
        lod_levels = ["ultra_low", "low", "medium", "default", "high", "ultra_high"]
        for lod_level in lod_levels:
            if all_predictions and lod_level in all_predictions:
                individual_skeleton = all_predictions[lod_level]['skeleton_points']
                individual_angles = all_predictions[lod_level]['angles']
                individual_pcd, individual_cylinders = create_skeleton_visualization(individual_skeleton)
                save_skeleton_data(individual_skeleton, individual_pcd, individual_cylinders, 
                                 output_dir, merged_cloud, lod_level=f"{lod_level}_individual", 
                                 angles=individual_angles)
            
            save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, 
                             output_dir, merged_cloud, lod_level=lod_level, angles=angles)
        
        save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, 
                         output_dir, merged_cloud, lod_level="voted_ensemble", angles=angles)
        
        xray_path = os.path.join(debug_dir, "xray_overlay.png")
        generate_xray_snapshot(mesh, skeleton_pcd, skeleton_cylinders, xray_path)

        visualize_results(merged_cloud, mesh, skeleton_pcd, skeleton_cylinders)
        
        print("\n="*30)
        print("     3D Posture Analysis Complete!")
        print("="*30)
        
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
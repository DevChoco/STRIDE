"""
포인트 클라우드 처리 모듈
포인트 클라우드 생성, 정렬, 병합 등의 처리를 담당합니다.
"""

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
    뎁스맵을 처리하여 포인트 클라우드를 생성합니다.
    
    Args:
        views_dict (dict): 뷰별 뎁스맵 파일 경로 딕셔너리
        debug_save (bool): 디버그 파일 저장 여부
        debug_dir (str): 디버그 파일 저장 경로
        
    Returns:
        dict: 뷰별 포인트 클라우드 딕셔너리
    """
    point_clouds = {}
    
    for view_name, file_path in views_dict.items():
        print(f"\n{view_name} 뷰 처리 중...")
        depth_map = load_depth_map(file_path)
        
        if depth_map is not None:
            # 디버깅을 위해 마스크 저장
            if debug_save:
                mask = create_mask_from_depth(depth_map, threshold_low=0.2, threshold_high=0.95)
                
                os.makedirs(debug_dir, exist_ok=True)
                mask_path = os.path.join(debug_dir, f"{view_name}_mask.png")
                cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))
                print(f"마스크 저장됨: {mask_path}")
            
            pcd = create_point_cloud_from_depth(depth_map, view_name)
            if pcd is not None:
                # 법선 벡터 계산
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
                point_clouds[view_name] = pcd
    
    return point_clouds


def align_point_clouds(point_clouds, use_point_to_plane_icp=True):
    """
    포인트 클라우드 정렬
    
    Args:
        point_clouds (dict): 뷰별 포인트 클라우드 딕셔너리
        use_point_to_plane_icp (bool): True면 Point-to-Plane ICP, False면 Point-to-Point ICP
    
    Returns:
        tuple: (aligned_clouds, view_names) - 정렬된 포인트 클라우드 리스트와 뷰 이름 리스트
    """
    icp_mode = "Point-to-Plane" if use_point_to_plane_icp else "Point-to-Point"
    print(f"\n=== FPFH 기반 포인트 클라우드 정렬 단계 ({icp_mode} ICP) ===")
    
    # 정면을 기준으로 정렬 시작
    aligned_clouds = [point_clouds["front"]]
    front_target = point_clouds["front"]
    view_names = ["front"]
    
    # 좌측과 우측을 정면과 정렬
    left_aligned = None
    right_aligned = None
    
    if "left" in point_clouds:
        print("\n좌측 뷰를 정면과 정렬...")
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
        print("\n우측 뷰를 정면과 정렬...")
        params_align = {
            'voxel_coarse': 3.0,  # 2.0에서 3.0으로 복원: 더 global한 RANSAC 특징
            'voxel_list': [10.0, 5.0, 2.5, 1.0],  # 매우 세밀한 4단계 멀티스케일
            'ransac_iter': 100000,  # 높은 RANSAC 반복 횟수 유지
            'use_cpd': False,
            'cpd_params': {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40},
            'fitness_threshold_accept': 0.02,
            'force_cpd': False,
            'allow_rotation': True,  # 우측은 회전이 필요할 수 있음
            'allow_small_rotation': True,
            'use_point_to_plane_icp': use_point_to_plane_icp
        }
        right_aligned = align_point_clouds_fpfh(point_clouds["right"], front_target, params=params_align)
        aligned_clouds.append(right_aligned)
        view_names.append("right")
    
    # 후면은 좌/우 포인트 클라우드(정렬 결과)에만 정렬
    if "back" in point_clouds:
        print("\n후면 뷰를 좌/우 누적 클라우드에 정렬...")
        side_target = o3d.geometry.PointCloud()
        st_points = []
        st_colors = []

        # 좌측 정렬 결과
        if left_aligned is not None and len(left_aligned.points) > 0:
            st_points.extend(np.asarray(left_aligned.points))
            if left_aligned.has_colors():
                st_colors.extend(np.asarray(left_aligned.colors))

        # 우측 정렬 결과
        if right_aligned is not None and len(right_aligned.points) > 0:
            st_points.extend(np.asarray(right_aligned.points))
            if right_aligned.has_colors():
                st_colors.extend(np.asarray(right_aligned.colors))

        if len(st_points) == 0:
            print("  좌/우 타겟이 비어 있어 후면 정렬을 건너뜁니다.")
        else:
            side_target.points = o3d.utility.Vector3dVector(np.array(st_points))
            if len(st_colors) == len(st_points) and len(st_colors) > 0:
                side_target.colors = o3d.utility.Vector3dVector(np.array(st_colors))

            # 후면을 좌/우 타겟에 정렬
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
    정렬된 포인트 클라우드를 병합하고 이상치를 제거합니다.
    
    Args:
        aligned_clouds (list): 정렬된 포인트 클라우드 리스트
        
    Returns:
        o3d.geometry.PointCloud: 병합되고 정리된 포인트 클라우드
    """
    print(f"\n=== 최종 병합 및 이상치 제거 ===")
    
    # 모든 포인트 클라우드를 하나로 합치기
    merged_cloud = o3d.geometry.PointCloud()
    points = []
    colors = []
    for pcd in aligned_clouds:
        points.extend(np.asarray(pcd.points))
        colors.extend(np.asarray(pcd.colors))
    merged_cloud.points = o3d.utility.Vector3dVector(np.array(points))
    merged_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
    
    print(f"병합된 포인트 수: {len(merged_cloud.points)}")
    
    # 극단적인 이상치만 제거 (다운샘플링 최소화)
    cl, ind = merged_cloud.remove_statistical_outlier(nb_neighbors=15, std_ratio=3.5)  # 매우 관대한 기준
    merged_cloud = cl
    print(f"극단적 이상치 제거 후: {len(merged_cloud.points)} 포인트")
    
    # 법선 벡터 재계산
    merged_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
    
    return merged_cloud

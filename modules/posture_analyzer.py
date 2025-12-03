"""
자세 분석 모듈
포인트 클라우드 및 LOD 메시에서 스켈레톤을 생성하고 자세를 분석합니다.
"""

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
    """
    포인트 클라우드에서 스켈레톤을 생성하고 자세를 분석합니다.
    
    Args:
        merged_cloud (o3d.geometry.PointCloud): 병합된 포인트 클라우드
        front_image_path (str): 정면 이미지 경로 (AI 랜드마크 검출용)
        
    Returns:
        tuple: (skeleton_points, angles, skeleton_pcd, skeleton_cylinders)
    """
    print("\n=== 스켈레톤 생성 및 자세 분석 ===")
    
    # AI 기반 랜드마크 검출
    front_landmarks = detect_landmarks_with_ai(front_image_path)
    
    if front_landmarks:
        print("AI 랜드마크 검출 성공! 개인별 신체 특징을 반영한 정확한 스켈레톤을 생성합니다.")
        for name, landmark in front_landmarks.items():
            print(f"  {name}: x={landmark['x']:.1f}, y={landmark['y']:.1f}, visibility={landmark['visibility']:.3f}")
    else:
        print("AI 랜드마크 검출 실패, 기본 해부학적 비율을 사용합니다.")
    
    # 스켈레톤 생성 및 각도 분석
    skeleton_points = create_skeleton_from_pointcloud(merged_cloud, front_landmarks)
    angles = calculate_spine_angles(skeleton_points)
    skeleton_pcd, skeleton_cylinders = create_skeleton_visualization(skeleton_points)
    
    # 각도 분석 결과 출력
    print_angles(angles)
    
    return skeleton_points, angles, skeleton_pcd, skeleton_cylinders


def analyze_posture_from_lod_meshes(lod_meshes, front_image_path):
    """
    여러 LOD 메시에서 스켈레톤을 예측하고 보팅 방식으로 최적의 골격을 선택합니다.
    
    Args:
        lod_meshes (dict): LOD 레벨별 메시 딕셔너리 {"ultra_low": mesh, "low": mesh, ...}
        front_image_path (str): 정면 이미지 경로 (AI 랜드마크 검출용)
        
    Returns:
        tuple: (voted_skeleton_points, voted_angles, skeleton_pcd, skeleton_cylinders, all_predictions)
    """
    print("\n=== 다중 해상도 앙상블 기반 스켈레톤 예측 ===")
    
    # AI 기반 랜드마크 검출
    front_landmarks = detect_landmarks_with_ai(front_image_path)
    
    if front_landmarks:
        print("AI 랜드마크 검출 성공! 개인별 신체 특징을 반영합니다.")
    else:
        print("AI 랜드마크 검출 실패, 기본 해부학적 비율을 사용합니다.")
    
    # 각 LOD 메시에서 독립적으로 스켈레톤 예측
    all_predictions = {}
    lod_order = ["ultra_low", "low", "medium", "default", "high", "ultra_high"]
    
    for lod_level in lod_order:
        if lod_level not in lod_meshes or lod_meshes[lod_level] is None:
            print(f"  ⚠ {lod_level} LOD 메시가 없습니다. 건너뜁니다.")
            continue
        
        print(f"\n  [{lod_level.upper()} LOD] 스켈레톤 예측 중...")
        mesh = lod_meshes[lod_level]
        
        # 메시를 포인트 클라우드로 변환
        pcd = mesh.sample_points_uniformly(number_of_points=50000)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5, max_nn=30))
        
        # 스켈레톤 생성
        skeleton_points = create_skeleton_from_pointcloud(pcd, front_landmarks)
        angles = calculate_spine_angles(skeleton_points)
        
        all_predictions[lod_level] = {
            'skeleton_points': skeleton_points,
            'angles': angles
        }
        print(f"    ✓ {lod_level} LOD 예측 완료: {len(skeleton_points)} 관절점")
    
    # 간단한 중앙값 기반 보팅으로 최종 스켈레톤 선택
    print("\n=== 중앙값 기반 최적 골격 선택 ===")
    
    if not all_predictions:
        raise ValueError("예측 결과가 없습니다.")
    
    # 모든 관절점 이름 수집
    joint_names = set()
    for pred in all_predictions.values():
        joint_names.update(pred['skeleton_points'].keys())
    
    voted_skeleton_points = {}
    
    # 각 관절점에 대해 중앙값 선택
    for joint_name in joint_names:
        joint_predictions = []
        for pred in all_predictions.values():
            if joint_name in pred['skeleton_points'] and pred['skeleton_points'][joint_name] is not None:
                joint_predictions.append(pred['skeleton_points'][joint_name])
        
        if not joint_predictions:
            voted_skeleton_points[joint_name] = None
            continue
        
        # 중앙값 계산 후 가장 가까운 예측 선택
        joint_predictions = np.array(joint_predictions)
        median_point = np.median(joint_predictions, axis=0)
        distances = np.linalg.norm(joint_predictions - median_point, axis=1)
        closest_idx = np.argmin(distances)
        voted_skeleton_points[joint_name] = joint_predictions[closest_idx]
    
    # 각도 보팅
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
        
        # 중앙값 선택
        angle_predictions = np.array(angle_predictions)
        median_angle = np.median(angle_predictions)
        distances = np.abs(angle_predictions - median_angle)
        closest_idx = np.argmin(distances)
        voted_angles[angle_name] = angle_predictions[closest_idx]
    
    # 최종 스켈레톤 시각화 객체 생성
    skeleton_pcd, skeleton_cylinders = create_skeleton_visualization(voted_skeleton_points)
    
    print("\n최종 보팅 결과:")
    print(f"  총 관절점 수: {len(voted_skeleton_points)}")
    print(f"  참여 LOD 모델 수: {len(all_predictions)}")
    print_angles(voted_angles)
    
    return voted_skeleton_points, voted_angles, skeleton_pcd, skeleton_cylinders, all_predictions, None
    return voted_skeleton_points, voted_angles, skeleton_pcd, skeleton_cylinders, all_predictions, voting_report

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
matplotlib.use('Agg')  # GUI 없이 그래프 생성
from matplotlib import font_manager, rc

# 한글 폰트 설정
try:
    # Windows의 맑은 고딕 폰트 사용
    font_path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(font_path):
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
    else:
        # 맑은 고딕이 없으면 나눔고딕 시도
        font_name = font_manager.FontProperties(fname="C:/Windows/Fonts/NanumGothic.ttf").get_name()
        rc('font', family=font_name)
except:
    # 폰트를 찾을 수 없으면 시스템 기본 폰트 사용
    try:
        rc('font', family='Malgun Gothic')
    except:
        pass

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False


def main():

    """메인 실행 함수"""
    print("="*60)
    print("     모듈화된 3D 자세 분석 시스템")
    print("     FPFH 정렬 + 스켈레톤 파싱")
    print("="*60)
    
    # ============================================================
    # ICP 설정: Point-to-Plane vs Point-to-Point
    # ============================================================
    # True: Point-to-Plane ICP (더 정밀, 법선 벡터 기반, 평면에 수직 방향 최적화)
    # False: Point-to-Point ICP (기본, 점 간 거리 최소화)
    USE_POINT_TO_PLANE_ICP = True  # ========================================================================================================================
    
    print(f"\n[ICP 모드 설정]")
    if USE_POINT_TO_PLANE_ICP:
        print("Point-to-Plane ICP 사용")
    else:
        print("Point-to-Point ICP 사용")
        
    # 입력 이미지 경로 설정
    nn = "16"
    gen = "여"

    views = {
        "front": rf"Data\DepthMap\{gen}_정면\DepthMap{nn}.bmp",
        "right": rf"Data\DepthMap\{gen}_R\DepthMap{nn}.bmp",
        "left": rf"Data\DepthMap\{gen}_L\DepthMap{nn}.bmp",
        "back": rf"Data\DepthMap\{gen}_후면\DepthMap{nn}.bmp"
    }
    
    try:
        # 현재 스크립트의 디렉토리를 기준으로 절대 경로 설정
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "output", "3d_models")
        debug_dir = os.path.join(script_dir, "output", "debug")
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(debug_dir, exist_ok=True)
        
        print(f"\n[출력 경로 설정]")
        print(f"3D 모델 저장 경로: {output_dir}")
        print(f"디버그 파일 경로: {debug_dir}")
        
        # 뎁스맵 처리 및 포인트 클라우드 생성
        point_clouds = process_depth_maps(views, debug_save=True, debug_dir=debug_dir)
        
        if not point_clouds:
            print("포인트 클라우드 생성 실패. 프로그램을 종료합니다.")
            return
        
        # 2단계: FPFH 기반 포인트 클라우드 정렬
        aligned_clouds, view_names = align_point_clouds(point_clouds, use_point_to_plane_icp=USE_POINT_TO_PLANE_ICP)
        
        # 3단계: 포인트 클라우드 병합 및 정리
        merged_cloud = merge_and_clean_pointclouds(aligned_clouds)
        
        # 4단계: 메시 생성 및 버텍스 리덕션
        print("\n=== 메시 생성 및 LOD 최적화 ===")
        print("포인트 클라우드를 고품질 메시로 변환하고 다중 해상도 LOD를 생성합니다...")
        
        try:
            mesh, saved_files = create_and_save_mesh(
                merged_cloud, 
                output_dir,  # 절대 경로 사용
                "body_mesh_fpfh",
                create_lod=True,
                reduction_ratio=0.2,  # 80% 버텍스 감소
                optimization_level="high_quality",  # 고품질 최적화
                enable_quality_analysis=True,
                enable_hole_filling=False  # 홀 채우기 비활성화
            )
        except TypeError:
            # 기존 함수 시그니처와 호환되지 않는 경우 기본 호출
            print("기본 메시 생성 모드로 전환...")
            mesh, saved_files = create_and_save_mesh(merged_cloud, output_dir, "body_mesh_fpfh")
        
        if saved_files:
            print(f"\n메시 파일이 저장되었습니다:")
            for file_path in saved_files:
                print(f"  {file_path}")
        
        # 4.5단계: 생성된 LOD 메시 로드
        print("\n=== LOD 메시 로드 ===")
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
                        print(f"  ✓ {lod_level} LOD 로드 완료: {len(loaded_mesh.vertices)} 버텍스")
                    else:
                        print(f"  ⚠ {lod_level} LOD 메시가 비어있습니다.")
                except Exception as e:
                    print(f"  ✗ {lod_level} LOD 로드 실패: {e}")
            else:
                print(f"  ⚠ {lod_level} LOD 파일이 없습니다: {mesh_path}")
        
        # 5단계: 다중 해상도 앙상블 기반 스켈레톤 파싱 및 자세 분석
        if len(lod_meshes) >= 3:  # 최소 3개 이상의 LOD 모델이 있어야 보팅 가능
            print(f"\n다중 해상도 앙상블 모드: {len(lod_meshes)}개 LOD 모델 사용")
            skeleton_points, angles, skeleton_pcd, skeleton_cylinders, all_predictions, _ = analyze_posture_from_lod_meshes(
                lod_meshes, views["front"]
            )
        else:
            print(f"\n단일 모드: LOD 모델이 부족하여 기본 포인트 클라우드 사용")
            skeleton_points, angles, skeleton_pcd, skeleton_cylinders = analyze_posture(
                merged_cloud, views["front"]
            )
            all_predictions = None
        
        # 스켈레톤 데이터를 모든 LOD 레벨에 대해 JSON으로 저장 (웹 뷰어용)
        print("\n=== 스켈레톤 데이터 저장 ===")
        lod_levels = ["ultra_low", "low", "medium", "default", "high", "ultra_high"]
        for lod_level in lod_levels:
            # 각 LOD별 개별 예측 결과 저장 (비교용)
            if all_predictions and lod_level in all_predictions:
                individual_skeleton = all_predictions[lod_level]['skeleton_points']
                individual_angles = all_predictions[lod_level]['angles']
                individual_pcd, individual_cylinders = create_skeleton_visualization(individual_skeleton)
                save_skeleton_data(individual_skeleton, individual_pcd, individual_cylinders, 
                                 output_dir, merged_cloud, lod_level=f"{lod_level}_individual", 
                                 angles=individual_angles)
            
            # 최종 보팅 결과 저장 (모든 LOD에 동일하게 적용)
            save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, 
                             output_dir, merged_cloud, lod_level=lod_level, angles=angles)
        
        # 보팅 결과를 별도 파일로 저장
        save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, 
                         output_dir, merged_cloud, lod_level="voted_ensemble", angles=angles)
        
        # 메시 내부 X-Ray 오버레이 이미지 생성 (디버그 경로 사용)
        xray_path = os.path.join(debug_dir, "xray_overlay.png")
        generate_xray_snapshot(mesh, skeleton_pcd, skeleton_cylinders, xray_path)

        # 6단계: 결과 시각화
        visualize_results(merged_cloud, mesh, skeleton_pcd, skeleton_cylinders)
        
        print("\n="*30)
        print("     3D 자세 분석이 완료되었습니다!")
        print("="*30)
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
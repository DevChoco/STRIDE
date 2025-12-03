"""
데이터 내보내기 모듈
스켈레톤 데이터를 JSON으로 저장하고 X-Ray 이미지를 생성합니다.
"""

import numpy as np
import os
import json
import cv2
import open3d as o3d


def save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, output_dir, merged_cloud=None, lod_level="default", angles=None):
    """
    스켈레톤 데이터를 JSON 파일로 저장합니다 (웹 뷰어용).
    
    Args:
        skeleton_points (dict): 스켈레톤 포인트 딕셔너리
        skeleton_pcd (o3d.geometry.PointCloud): 스켈레톤 포인트 클라우드
        skeleton_cylinders (list): 스켈레톤 연결선 실린더 리스트
        output_dir (str): 저장 디렉토리
        merged_cloud (o3d.geometry.PointCloud): 병합된 포인트 클라우드 (크기 계산용)
        lod_level (str): LOD 레벨 이름
        angles (dict): 척추 각도 정보
    """
    if skeleton_points is None:
        return
    
    try:
        skeleton_data = {
            "points": {},
            "connections": [],
            "mesh_info": {},
            "lod_level": lod_level,
            "quality_metrics": {
                "angles": angles if angles else {}
            }
        }
        
        # 메시/포인트클라우드 크기 정보 저장
        if merged_cloud is not None and len(merged_cloud.points) > 0:
            points = np.asarray(merged_cloud.points)
            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            
            skeleton_data["mesh_info"] = {
                "height": float(max_bound[1] - min_bound[1]),
                "width": float(max_bound[0] - min_bound[0]),
                "depth": float(max_bound[2] - min_bound[2]),
                "min_bound": {
                    "x": float(min_bound[0]),
                    "y": float(min_bound[1]),
                    "z": float(min_bound[2])
                },
                "max_bound": {
                    "x": float(max_bound[0]),
                    "y": float(max_bound[1]),
                    "z": float(max_bound[2])
                }
            }
            print(f"  메시 크기 - 높이: {skeleton_data['mesh_info']['height']:.2f}, "
                  f"너비: {skeleton_data['mesh_info']['width']:.2f}, "
                  f"깊이: {skeleton_data['mesh_info']['depth']:.2f}")
        
        # 포인트 데이터 저장
        for name, point in skeleton_points.items():
            if point is not None:
                skeleton_data["points"][name] = {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2])
                }
        
        # 연결선 데이터 저장 (스켈레톤 구조)
        connections = [
            ["HEAD", "NECK"],
            ["NECK", "SPINE_UPPER"],
            ["SPINE_UPPER", "SPINE_MID"],
            ["SPINE_MID", "SPINE_LOWER"],
            ["SPINE_LOWER", "PELVIS"],
            ["NECK", "SHOULDER_LEFT"],
            ["NECK", "SHOULDER_RIGHT"],
            ["SHOULDER_LEFT", "ELBOW_LEFT"],
            ["SHOULDER_RIGHT", "ELBOW_RIGHT"],
            ["ELBOW_LEFT", "WRIST_LEFT"],
            ["ELBOW_RIGHT", "WRIST_RIGHT"],
            ["PELVIS", "HIP_LEFT"],
            ["PELVIS", "HIP_RIGHT"],
            ["HIP_LEFT", "KNEE_LEFT"],
            ["HIP_RIGHT", "KNEE_RIGHT"],
            ["KNEE_LEFT", "ANKLE_LEFT"],
            ["KNEE_RIGHT", "ANKLE_RIGHT"]
        ]
        
        for connection in connections:
            if connection[0] in skeleton_data["points"] and connection[1] in skeleton_data["points"]:
                skeleton_data["connections"].append(connection)
        
        # JSON 파일로 저장 (LOD 레벨 포함)
        if lod_level == "default":
            json_path = os.path.join(output_dir, "skeleton_data_default.json")
        else:
            json_path = os.path.join(output_dir, f"skeleton_data_{lod_level}.json")
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n스켈레톤 데이터 저장 ({lod_level}): {json_path}")
        print(f"  포인트 수: {len(skeleton_data['points'])}")
        print(f"  연결선 수: {len(skeleton_data['connections'])}")
        if angles:
            print(f"  척추 각도 포함: {len(angles)}개")
        
    except Exception as e:
        print(f"스켈레톤 데이터 저장 중 오류: {e}")


def generate_xray_snapshot(mesh, skeleton_pcd, skeleton_cylinders, output_path="output/debug/xray_overlay.png"):
    """
    메시 위에 스켈레톤을 강제로 오버레이한 X-Ray 이미지를 생성합니다.

    Open3D의 실시간 뷰어는 깊이 테스트를 비활성화할 수 없어 완전한 투시가 어렵습니다.
    대신 오프스크린 렌더러로 메시와 스켈레톤을 각각 렌더링한 뒤 2D에서 합성합니다.
    """
    if mesh is None:
        return

    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 960)
    except Exception as exc:
        print(f"X-Ray 스냅샷 생성 실패: {exc}")
        return

    scene = renderer.scene
    scene.set_background([0, 0, 0, 1])

    bbox = mesh.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    extent = bbox.get_extent()
    radius = np.linalg.norm(extent)
    eye = center + np.array([0.0, 0.0, max(radius, 1.0)])
    up = np.array([0.0, 1.0, 0.0])

    mesh_material = o3d.visualization.rendering.MaterialRecord()
    mesh_material.shader = "defaultLit"
    mesh_material.base_color = (0.7, 0.75, 0.85, 1.0)
    scene.add_geometry("mesh", mesh, mesh_material)
    renderer.setup_camera(55.0, center, eye, up)
    mesh_image = np.asarray(renderer.render_to_image())

    scene.clear_geometry()

    skeleton_point_mat = o3d.visualization.rendering.MaterialRecord()
    skeleton_point_mat.shader = "defaultUnlit"
    skeleton_point_mat.base_color = (1.0, 0.2, 0.2, 1.0)
    skeleton_point_mat.point_size = 12.0
    scene.add_geometry("skeleton_points", skeleton_pcd, skeleton_point_mat)

    for idx, cylinder in enumerate(skeleton_cylinders):
        cyl_mat = o3d.visualization.rendering.MaterialRecord()
        cyl_mat.shader = "defaultUnlit"
        cyl_mat.base_color = (1.0, 0.5, 0.0, 1.0)
        scene.add_geometry(f"skeleton_bone_{idx}", cylinder, cyl_mat)

    renderer.setup_camera(55.0, center, eye, up)
    skeleton_image = np.asarray(renderer.render_to_image())

    if mesh_image.dtype != np.uint8:
        mesh_image = (mesh_image * 255).clip(0, 255).astype(np.uint8)
    if skeleton_image.dtype != np.uint8:
        skeleton_image = (skeleton_image * 255).clip(0, 255).astype(np.uint8)

    overlay = mesh_image.copy()
    mask = np.any(skeleton_image > 15, axis=2)
    overlay[mask] = skeleton_image[mask]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay_bgr)
    print(f"X-Ray 오버레이 이미지 저장: {output_path}")

import numpy as np
import os
import json
import cv2
import open3d as o3d


def save_skeleton_data(skeleton_points, skeleton_pcd, skeleton_cylinders, output_dir, merged_cloud=None, lod_level="default", angles=None):
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
            print(f"  Mesh size - Height: {skeleton_data['mesh_info']['height']:.2f}, "
                  f"Width: {skeleton_data['mesh_info']['width']:.2f}, "
                  f"Depth: {skeleton_data['mesh_info']['depth']:.2f}")
        for name, point in skeleton_points.items():
            if point is not None:
                skeleton_data["points"][name] = {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "z": float(point[2])
                }
        
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
        
        if lod_level == "default":
            json_path = os.path.join(output_dir, "skeleton_data_default.json")
        else:
            json_path = os.path.join(output_dir, f"skeleton_data_{lod_level}.json")
            
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(skeleton_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSkeleton data saved ({lod_level}): {json_path}")
        print(f"  Point count: {len(skeleton_data['points'])}")
        print(f"  Connection count: {len(skeleton_data['connections'])}")
        if angles:
            print(f"  Spine angles included: {len(angles)}")
        
    except Exception as e:
        print(f"Error saving skeleton data: {e}")


def generate_xray_snapshot(mesh, skeleton_pcd, skeleton_cylinders, output_path="output/debug/xray_overlay.png"):
    if mesh is None:
        return

    try:
        renderer = o3d.visualization.rendering.OffscreenRenderer(1280, 960)
    except Exception as exc:
        print(f"X-Ray snapshot generation failed: {exc}")
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
    print(f"X-Ray image saved: {output_path}")

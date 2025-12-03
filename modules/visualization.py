"""
시각화 모듈
포인트 클라우드, 메시, 스켈레톤 등의 3D 데이터를 시각화합니다.
"""

import numpy as np
import open3d as o3d


def visualize_results(merged_cloud, mesh, skeleton_pcd, skeleton_cylinders):
    """
    결과를 시각화합니다.
    
    Args:
        merged_cloud (o3d.geometry.PointCloud): 병합된 포인트 클라우드
        mesh (o3d.geometry.TriangleMesh): 생성된 메시
        skeleton_pcd (o3d.geometry.PointCloud): 스켈레톤 포인트 클라우드
        skeleton_cylinders (list): 스켈레톤 연결선 실린더 리스트
    """
    print("\n=== 3D 시각화 ===")
    
    # 시각화 창 생성
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose Analysis with FPFH Alignment", width=1024, height=768)
    
    # 포인트 클라우드 추가 (더 투명하게)
    merged_cloud_small = merged_cloud.voxel_down_sample(voxel_size=5.0)  # 더 많이 다운샘플링
    merged_cloud_small.paint_uniform_color([0.3, 0.3, 0.3])  # 더 어두운 회색으로 반투명 효과
    vis.add_geometry(merged_cloud_small)
    
    # 메시 추가 (있는 경우) - 와이어프레임으로만 표시해 스켈레톤을 가리지 않음
    if mesh is not None:
        mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_wireframe.paint_uniform_color([0.5, 0.5, 0.5])  # 연한 회색 와이어프레임
        vis.add_geometry(mesh_wireframe)
    
    # 스켈레톤 추가
    vis.add_geometry(skeleton_pcd)
    for cylinder in skeleton_cylinders:
        vis.add_geometry(cylinder)
    
    # 렌더링 옵션 설정
    opt = vis.get_render_option()
    opt.point_size = 3.0  # 스켈레톤 포인트 크기를 크게 설정
    opt.background_color = np.asarray([1, 1, 1])  # 검은색 배경
    opt.mesh_show_wireframe = True  # 와이어프레임 표시
    opt.mesh_show_back_face = True  # 메시 뒷면도 표시
    
    # 카메라 위치 설정
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    
    # 시각화 실행
    vis.run()
    vis.destroy_window()

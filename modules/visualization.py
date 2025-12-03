import numpy as np
import open3d as o3d


def visualize_results(merged_cloud, mesh, skeleton_pcd, skeleton_cylinders):
    print("\n=== 3D Visualization ===")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Pose Analysis with FPFH Alignment", width=1024, height=768)
    
    merged_cloud_small = merged_cloud.voxel_down_sample(voxel_size=5.0)
    merged_cloud_small.paint_uniform_color([0.3, 0.3, 0.3])
    vis.add_geometry(merged_cloud_small)
    
    if mesh is not None:
        mesh_wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
        mesh_wireframe.paint_uniform_color([0.5, 0.5, 0.5])
        vis.add_geometry(mesh_wireframe)
    vis.add_geometry(skeleton_pcd)
    for cylinder in skeleton_cylinders:
        vis.add_geometry(cylinder)
    
    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.asarray([1, 1, 1])
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = True
    
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.5, -0.5, -0.5])
    ctr.set_up([0, -1, 0])
    vis.run()
    vis.destroy_window()

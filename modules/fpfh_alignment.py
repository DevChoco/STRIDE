import numpy as np
import cv2
import open3d as o3d
import copy
import importlib
from scipy.spatial import cKDTree

# 상대 임포트 (패키지로 사용될 때)
try:
    from .pointcloud_generator import preprocess_for_icp
except ImportError:
    # 절대 임포트 (직접 실행될 때)
    try:
        from pointcloud_generator import preprocess_for_icp
    except ImportError:
        preprocess_for_icp = None
        print("Warning: Cannot import preprocess_for_icp.")


def compute_fpfh(pcd, voxel_size):
    """
    Compute FPFH features from point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): Point cloud
        voxel_size (float): Voxel size
        
    Returns:
        o3d.pipelines.registration.Feature: FPFH features
    """
    radius_normal = voxel_size * 2.0
    radius_feature = voxel_size * 5.0
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return fpfh


def global_registration_fpfh_ransac(source, target, voxel_size=5.0, ransac_iter=20000):
    """
    FPFH 특징 + RANSAC을 이용한 전역 초기 정합 (Open3D)
    
    Args:
        source (o3d.geometry.PointCloud): 소스 포인트 클라우드
        target (o3d.geometry.PointCloud): 타겟 포인트 클라우드
        voxel_size (float): 다운샘플링 복셀 크기
        ransac_iter (int): RANSAC 반복 횟수
        
    Returns:
        tuple: (transformation matrix, result object)
    """
    src_down = source.voxel_down_sample(voxel_size)
    tgt_down = target.voxel_down_sample(voxel_size)
    if len(src_down.points) == 0 or len(tgt_down.points) == 0:
        return np.eye(4), None
    
    src_fpfh = compute_fpfh(src_down, voxel_size)
    tgt_fpfh = compute_fpfh(tgt_down, voxel_size)

    distance_threshold = voxel_size * 2.5  # 1.5에서 2.5로 증가: 더 넓은 correspondence 허용
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_down, tgt_down, src_fpfh, tgt_fpfh, mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(max_iteration=ransac_iter, confidence=0.95)
    )
    return result.transformation, result


def multi_scale_icp(source, target, voxel_list=[20.0, 10.0, 5.0], use_point_to_plane=True, max_iter_per_scale=[150, 200, 250, 300]):
    """
    Multi-scale ICP based on voxel: progressing from coarse to fine alignment
    
    Args:
        source (o3d.geometry.PointCloud): Source point cloud
        target (o3d.geometry.PointCloud): Target point cloud
        voxel_list (list): Voxel size list (from large to small)
        use_point_to_plane (bool): Whether to use Point-to-Plane ICP
        max_iter_per_scale (list): Maximum iteration count per scale
        
    Returns:
        tuple: (final transformation, final result)
    """
    current_trans = np.eye(4)
    final_result = None
    
    for i, voxel in enumerate(voxel_list):
        src_down = source.voxel_down_sample(voxel)
        tgt_down = target.voxel_down_sample(voxel)
        if len(src_down.points) == 0 or len(tgt_down.points) == 0:
            continue
            
        if use_point_to_plane:
            src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
            tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))

        distance_threshold = voxel * 3.0
        estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if use_point_to_plane
                      else o3d.pipelines.registration.TransformationEstimationPointToPoint())

        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=distance_threshold,
            init=current_trans,
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter_per_scale[i if i < len(max_iter_per_scale) else -1]
            )
        )
        current_trans = result.transformation
        final_result = result

    if final_result is None:
        estimation = (o3d.pipelines.registration.TransformationEstimationPointToPlane() if use_point_to_plane
                      else o3d.pipelines.registration.TransformationEstimationPointToPoint())
        final_result = o3d.pipelines.registration.registration_icp(
            source, target,
            max_correspondence_distance=voxel_list[-1] * 2.0,
            init=np.eye(4),
            estimation_method=estimation,
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200)
        )
        current_trans = final_result.transformation

    return current_trans, final_result


def _get_centroid(pcd):
    """Calculate the centroid of point cloud."""
    pts = np.asarray(pcd.points)
    if pts.size == 0:
        return np.zeros(3, dtype=np.float64)
    return pts.mean(axis=0)


def _clamp_rotation(R, max_angle_rad):
    """Limit rotation angle to max_angle_rad using Rodrigues axis-angle representation."""
    rvec, _ = cv2.Rodrigues(R)
    angle = float(np.linalg.norm(rvec))
    if angle <= 1e-12 or angle <= max_angle_rad:
        return R
    rvec_unit = (rvec / angle) * max_angle_rad
    R_limited, _ = cv2.Rodrigues(rvec_unit)
    return R_limited


def small_rotation_icp(source, target, voxel_list=[20.0, 10.0, 5.0], max_iter_per_scale=[30, 30, 60],
                       max_angle_deg_per_scale=[2.0, 1.0, 0.5], init_trans=None):
    """
    ICP allowing only very small angles.
    
    Args:
        source (o3d.geometry.PointCloud): Source point cloud
        target (o3d.geometry.PointCloud): Target point cloud
        voxel_list (list): Voxel size list
        max_iter_per_scale (list): Maximum iteration count per scale
        max_angle_deg_per_scale (list): Maximum allowed angle per scale (degrees)
        init_trans (np.ndarray): Initial transformation matrix
        
    Returns:
        tuple: (transformation matrix, result stub)
    """
    if init_trans is None:
        current_trans = np.eye(4)
        t0 = _get_centroid(target) - _get_centroid(source)
        current_trans[:3, 3] = t0
    else:
        current_trans = init_trans.copy()

    last_fitness, last_rmse = 0.0, np.inf

    for i, voxel in enumerate(voxel_list):
        src_down = source.voxel_down_sample(voxel)
        tgt_down = target.voxel_down_sample(voxel)
        if len(src_down.points) == 0 or len(tgt_down.points) == 0:
            continue

        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.0, max_nn=30))

        distance_threshold = voxel * 1.5
        result = o3d.pipelines.registration.registration_icp(
            src_down, tgt_down,
            max_correspondence_distance=distance_threshold,
            init=current_trans,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=max_iter_per_scale[i if i < len(max_iter_per_scale) else -1]
            )
        )

        prev_trans = current_trans
        T_est = result.transformation
        delta = T_est @ np.linalg.inv(prev_trans)
        R_delta = delta[:3, :3]
        t_delta = delta[:3, 3]

        max_ang_rad = np.deg2rad(max_angle_deg_per_scale[i if i < len(max_angle_deg_per_scale) else -1])
        R_delta_limited = _clamp_rotation(R_delta, max_ang_rad)

        delta_limited = np.eye(4)
        delta_limited[:3, :3] = R_delta_limited
        delta_limited[:3, 3] = t_delta

        current_trans = delta_limited @ prev_trans

        src_eval = copy.deepcopy(source).transform(current_trans)
        dists = np.asarray(target.compute_point_cloud_distance(src_eval))
        if dists.size > 0:
            inliers = dists < distance_threshold
            last_fitness = float(np.sum(inliers)) / max(1, len(dists))
            last_rmse = float(np.sqrt(np.mean((dists[inliers] ** 2))) if np.any(inliers) else np.inf)

    result_stub = type("ICPResult", (), {})()
    result_stub.fitness = last_fitness
    result_stub.inlier_rmse = last_rmse
    return current_trans, result_stub


def translation_only_icp(source, target, voxel_list=[20.0, 10.0, 5.0], max_iter_per_scale=[20, 20, 30], max_corr_factor=1.5):
    """
    Translation-only ICP that does not allow rotation.
    
    Args:
        source (o3d.geometry.PointCloud): Source point cloud
        target (o3d.geometry.PointCloud): Target point cloud
        voxel_list (list): Voxel size list
        max_iter_per_scale (list): Maximum iteration count per scale
        max_corr_factor (float): Maximum correspondence distance factor
        
    Returns:
        tuple: (transformation matrix, result dict)
    """
    t = (_get_centroid(target) - _get_centroid(source)).astype(np.float64)
    last_fitness, last_rmse = 0.0, np.inf

    for i, voxel in enumerate(voxel_list):
        src_down = source.voxel_down_sample(voxel)
        tgt_down = target.voxel_down_sample(voxel)
        if len(src_down.points) == 0 or len(tgt_down.points) == 0:
            continue

        src_np = np.asarray(src_down.points)
        tgt_np = np.asarray(tgt_down.points)
        tree = cKDTree(tgt_np)
        max_corr_dist = voxel * max_corr_factor

        for it in range(max_iter_per_scale[i if i < len(max_iter_per_scale) else -1]):
            src_trans = src_np + t  # 회전 없이 번역만 적용
            dists, idx = tree.query(src_trans, k=1)
            inlier_mask = dists < max_corr_dist
            if not np.any(inlier_mask):
                break
            src_in = src_trans[inlier_mask]
            tgt_in = tgt_np[idx[inlier_mask]]
            # 평균 변위
            delta = (tgt_in - src_in).mean(axis=0)
            t += delta
            # 수렴 체크
            if np.linalg.norm(delta) < max(1e-3, voxel * 1e-3):
                break

        # 스케일별 평가
        if len(tgt_np) > 0:
            src_final = src_np + t
            dists, _ = tree.query(src_final, k=1)
            inliers = dists < max_corr_dist
            last_fitness = float(np.sum(inliers)) / max(1, len(dists))
            last_rmse = float(np.sqrt(np.mean((dists[inliers] ** 2))) if np.any(inliers) else np.inf)

    # 최종 4x4 변환 (R=I, t=t)
    T = np.eye(4)
    T[:3, 3] = t
    result_stub = type("ICPResult", (), {})()
    result_stub.fitness = last_fitness
    result_stub.inlier_rmse = last_rmse
    return T, result_stub


def cpd_refine_np(source_pcd, target_pcd, max_points=2000, cpd_beta=2.0, cpd_lambda=2.0, cpd_iter=40, w=0.0):
    """
    Non-rigid correction based on pycpd (DeformableRegistration)
    
    Args:
        source_pcd (o3d.geometry.PointCloud): Source point cloud
        target_pcd (o3d.geometry.PointCloud): Target point cloud
        max_points (int): Maximum number of points (computation control)
        cpd_beta (float): CPD beta parameter
        cpd_lambda (float): CPD lambda parameter
        cpd_iter (int): Maximum iteration count
        w (float): Noise ratio parameter
        
    Returns:
        o3d.geometry.PointCloud: Deformed point cloud
    """
    try:
        cpd_mod = importlib.import_module("pycpd")
        DeformableRegistration = getattr(cpd_mod, "DeformableRegistration")
    except Exception as e:
        raise RuntimeError("pycpd is not installed. Please add pycpd to requirements.txt and install it.") from e
    
    source_np = np.asarray(source_pcd.points)
    target_np = np.asarray(target_pcd.points)

    if source_np.shape[0] == 0 or target_np.shape[0] == 0:
        return copy.deepcopy(source_pcd)

    if source_np.shape[0] > max_points:
        idx = np.random.choice(source_np.shape[0], max_points, replace=False)
        source_sub = source_np[idx]
    else:
        source_sub = source_np.copy()
    if target_np.shape[0] > max_points:
        idx2 = np.random.choice(target_np.shape[0], max_points, replace=False)
        target_sub = target_np[idx2]
    else:
        target_sub = target_np.copy()

    reg = DeformableRegistration(X=target_sub, Y=source_sub, max_iterations=cpd_iter, w=w, beta=cpd_beta, lambd=cpd_lambda)
    TY, _ = reg.register()

    tree = cKDTree(source_sub)
    dists, idx_nn = tree.query(source_np, k=1)
    displacement = TY[idx_nn] - source_sub[idx_nn]
    warped = source_np + displacement

    warped_pcd = o3d.geometry.PointCloud()
    warped_pcd.points = o3d.utility.Vector3dVector(warped)
    if source_pcd.has_colors():
        warped_pcd.colors = copy.deepcopy(source_pcd.colors)
    return warped_pcd


def align_point_clouds_fpfh(source, target, params=None):
    """
    Improved FPFH-based alignment: (1) global FPFH+RANSAC -> (2) multi-scale ICP -> (3) optional CPD refine
    
    Args:
        source (o3d.geometry.PointCloud): Source point cloud
        target (o3d.geometry.PointCloud): Target point cloud
        params (dict): Alignment parameter dictionary
        
    Returns:
        o3d.geometry.PointCloud: Aligned point cloud
    """
    if source is None or target is None or len(source.points) == 0 or len(target.points) == 0:
        return source
    
    if params is None:
        params = {}
    
    voxel_coarse = params.get('voxel_coarse', 5.0)
    voxel_list = params.get('voxel_list', [20.0, 10.0, 5.0])
    ransac_iter = params.get('ransac_iter', 20000)
    use_cpd = params.get('use_cpd', True)
    cpd_params = params.get('cpd_params', {'max_points':1500, 'cpd_beta':2.0, 'cpd_lambda':2.0, 'cpd_iter':40})
    fitness_threshold_accept = params.get('fitness_threshold_accept', 0.02)
    allow_rotation = params.get('allow_rotation', True)
    allow_small_rotation = params.get('allow_small_rotation', False)
    use_point_to_plane_icp = params.get('use_point_to_plane_icp', True)

    src = copy.deepcopy(source)
    tgt = copy.deepcopy(target)

    if allow_rotation:
        try:
            init_trans, ransac_result = global_registration_fpfh_ransac(src, tgt, voxel_size=voxel_coarse, ransac_iter=ransac_iter)
            if ransac_result is not None:
                print(f"  Global RANSAC fitness={getattr(ransac_result,'fitness',None)}, inlier_rmse={getattr(ransac_result,'inlier_rmse',None)}")
            src.transform(init_trans)
        except Exception as e:
            print("  Global RANSAC 실패:", e)

        # 2) Multi-scale ICP (회전 허용, Point-to-Plane 또는 Point-to-Point)
        icp_type = "Point-to-Plane" if use_point_to_plane_icp else "Point-to-Point"
        print(f"  Multi-scale {icp_type} ICP 수행 중...")
        trans_icp, icp_result = multi_scale_icp(src, tgt, voxel_list=voxel_list, use_point_to_plane=use_point_to_plane_icp)
        print(f"  Multi-scale {icp_type} ICP fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")
    else:
        print("  회전 비허용 모드: translation-only ICP 수행")
        # 전역 RANSAC은 사용하지 않고 바로 번역 전용 ICP
        trans_icp, icp_result = translation_only_icp(src, tgt, voxel_list=voxel_list, max_iter_per_scale=[30,30,50])
        print(f"  Translation-Only ICP fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")
    
    if not allow_rotation and allow_small_rotation:
        print("  Small rotation allowed mode: fine-tuning with small-rotation ICP")
        trans_icp, icp_result = small_rotation_icp(src, tgt, voxel_list=voxel_list,
                                                   max_iter_per_scale=[20,20,40],
                                                   max_angle_deg_per_scale=[2.0, 1.0, 0.5],
                                                   init_trans=trans_icp)
        print(f"  Small-Rotation ICP fitness={icp_result.fitness:.4f}, rmse={icp_result.inlier_rmse:.4f}")

    src_transformed = src.transform(trans_icp)

    # 정합 품질 평가: ICP 결과 객체에서 직접 가져오기 (Point-to-Plane과 Point-to-Point 구분)
    if icp_result is not None:
        fitness = icp_result.fitness
        rmse = icp_result.inlier_rmse
        icp_type = "Point-to-Plane" if use_point_to_plane_icp else "Point-to-Point"
        print(f"  정렬 평가 ({icp_type} ICP): fitness={fitness:.4f}, RMSE={rmse:.4f}")
    else:
        # fallback: 수동 계산 (ICP 결과가 없는 경우)
        distances = np.asarray(tgt.compute_point_cloud_distance(src_transformed))
        if distances.size == 0:
            return src_transformed
        inliers = distances < (voxel_list[-1] * 2.0)
        fitness = float(np.sum(inliers)) / max(1, len(distances))
        print(f"  정렬 평가: estimated fitness (nearest dist<{voxel_list[-1]*2.0:.4f}) = {fitness:.4f}")

    # 3) 조건부 CPD 비강직 보정 (옵션)
    if use_cpd and allow_rotation:
        attempt_cpd = params.get('force_cpd', False) or (fitness < 0.25)
        if attempt_cpd:
            print("  CPD 비강직 보정 시도 (조건부)...")
            try:
                warped = cpd_refine_np(src_transformed, tgt, max_points=cpd_params.get('max_points',1500),
                                       cpd_beta=cpd_params.get('cpd_beta',2.0), cpd_lambda=cpd_params.get('cpd_lambda',2.0),
                                       cpd_iter=cpd_params.get('cpd_iter',40))
                distances2 = np.asarray(tgt.compute_point_cloud_distance(warped))
                fitness2 = float(np.sum(distances2 < (voxel_list[-1]*2.0))) / max(1, len(distances2))
                print(f"  CPD 후 fitness = {fitness2:.4f}")
                if fitness2 >= fitness or fitness < fitness_threshold_accept:
                    print("  CPD 결과 채택")
                    return warped
                else:
                    print("  CPD 결과 기각 (개선 없음)")
            except Exception as e:
                print("  CPD 오류:", e)

    return src_transformed
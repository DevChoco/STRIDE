"""Advanced Mesh Optimization and Vertex Reduction Module

This module provides the following features:
- Intelligent vertex reduction
- Adaptive mesh simplification
- Mesh quality analysis and optimization
- Multiple LOD (Level of Detail) generation
- Advanced mesh post-processing
"""

import numpy as np
import open3d as o3d
import os
import copy


def analyze_mesh_complexity(mesh):
    """
    Analyze mesh complexity.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Mesh to analyze
        
    Returns:
        dict: Complexity analysis results
    """
    if mesh is None:
        return {}
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    analysis = {
        'vertex_count': len(vertices),
        'triangle_count': len(triangles),
        'edge_count': len(mesh.get_non_manifold_edges()),
        'surface_area': mesh.get_surface_area(),
        'volume': mesh.get_volume() if mesh.is_watertight() else 0.0,
        'is_watertight': mesh.is_watertight(),
        'is_manifold': len(mesh.get_non_manifold_edges()) == 0
    }
    
    # 복잡성 점수 계산 (0.0 ~ 1.0)
    vertex_complexity = min(1.0, len(vertices) / 100000)  # 10만 버텍스를 기준으로
    triangle_complexity = min(1.0, len(triangles) / 200000)  # 20만 삼각형을 기준으로
    analysis['complexity_score'] = (vertex_complexity + triangle_complexity) / 2
    
    return analysis


def smart_vertex_reduction(mesh, target_ratio=0.5, quality_priority=True):
    """
    Perform intelligent vertex reduction.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Target mesh
        target_ratio (float): Target vertex ratio (0.0 ~ 1.0)
        quality_priority (bool): Whether to prioritize quality
        
    Returns:
        o3d.geometry.TriangleMesh: Optimized mesh
    """
    if mesh is None:
        return None
    
    print(f"\n=== Intelligent Vertex Reduction ===")
    
    original_vertices = len(mesh.vertices)
    original_triangles = len(mesh.triangles)
    target_triangles = max(100, int(original_triangles * target_ratio))
    
    print(f"Original: {original_vertices:,} vertices, {original_triangles:,} triangles")
    print(f"Target: {target_triangles:,} triangles ({(1-target_ratio)*100:.1f}% reduction)")
    print(f"Mode: {'Quality priority' if quality_priority else 'Speed priority'}")
    
    try:
        # 메시 전처리
        mesh_clean = copy.deepcopy(mesh)
        mesh_clean.remove_degenerate_triangles()
        mesh_clean.remove_duplicated_triangles()
        mesh_clean.remove_duplicated_vertices()
        mesh_clean.remove_non_manifold_edges()
        
        # 법선 벡터 계산 (Quadric 알고리즘에 필요)
        mesh_clean.compute_vertex_normals()
        mesh_clean.compute_triangle_normals()
        
        if quality_priority:
            # 품질 우선: Quadric Error Metrics 사용
            print("Quadric Error Decimation 적용 중...")
            simplified_mesh = mesh_clean.simplify_quadric_decimation(
                target_number_of_triangles=target_triangles,
                maximum_error=0.01,  # 낮은 오차 허용
                boundary_weight=1.0  # 경계 보존
            )
        else:
            # 속도 우선: 점진적 단순화
            print("점진적 단순화 적용 중...")
            simplified_mesh = progressive_simplification(mesh_clean, target_ratio)
        
        # 결과 분석
        final_vertices = len(simplified_mesh.vertices)
        final_triangles = len(simplified_mesh.triangles)
        
        vertex_reduction = (original_vertices - final_vertices) / original_vertices * 100
        triangle_reduction = (original_triangles - final_triangles) / original_triangles * 100
        
        print(f"Result: {final_vertices:,} vertices, {final_triangles:,} triangles")
        print(f"Actual reduction: vertices {vertex_reduction:.1f}%, triangles {triangle_reduction:.1f}%")
        
        return simplified_mesh
        
    except Exception as e:
        print(f"Error during vertex reduction: {e}")
        return mesh


def progressive_simplification(mesh, target_ratio, steps=5):
    """
    Perform progressive mesh simplification.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Target mesh
        target_ratio (float): Final target ratio
        steps (int): Number of simplification steps
        
    Returns:
        o3d.geometry.TriangleMesh: Simplified mesh
    """
    current_mesh = copy.deepcopy(mesh)
    step_ratio = pow(target_ratio, 1.0 / steps)
    
    for i in range(steps):
        current_triangles = len(current_mesh.triangles)
        step_target = max(100, int(current_triangles * step_ratio))
        
        print(f"  Step {i+1}/{steps}: {current_triangles:,} → {step_target:,} triangles")
        
        current_mesh = current_mesh.simplify_quadric_decimation(
            target_number_of_triangles=step_target
        )
        
        # 중간 정리
        current_mesh.remove_degenerate_triangles()
        current_mesh.remove_duplicated_vertices()
    
    return current_mesh


def adaptive_mesh_optimization(mesh, complexity_level="auto"):
    """
    Perform adaptive optimization based on mesh complexity.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Target mesh
        complexity_level (str): Complexity level ("low", "medium", "high", "auto")
        
    Returns:
        o3d.geometry.TriangleMesh: Optimized mesh
    """
    if mesh is None:
        return None
    
    analysis = analyze_mesh_complexity(mesh)
    
    if complexity_level == "auto":
        complexity_score = analysis['complexity_score']
        if complexity_score > 0.7:
            complexity_level = "high"
        elif complexity_score > 0.3:
            complexity_level = "medium"
        else:
            complexity_level = "low"
    
    print(f"\n=== Adaptive Mesh Optimization ===")
    print(f"Complexity level: {complexity_level}")
    print(f"Complexity score: {analysis['complexity_score']:.3f}")
    
    if complexity_level == "high":
        optimized_mesh = smart_vertex_reduction(mesh, target_ratio=0.3, quality_priority=True)
        optimized_mesh = optimized_mesh.filter_smooth_simple(number_of_iterations=3)
        
    elif complexity_level == "medium":
        optimized_mesh = smart_vertex_reduction(mesh, target_ratio=0.5, quality_priority=True)
        optimized_mesh = optimized_mesh.filter_smooth_simple(number_of_iterations=2)
        
    else:
        optimized_mesh = smart_vertex_reduction(mesh, target_ratio=0.8, quality_priority=False)
        optimized_mesh = optimized_mesh.filter_smooth_simple(number_of_iterations=1)
    
    optimized_mesh.remove_degenerate_triangles()
    optimized_mesh.remove_duplicated_triangles()
    optimized_mesh.remove_duplicated_vertices()
    optimized_mesh.compute_vertex_normals()
    optimized_mesh.compute_triangle_normals()
    
    return optimized_mesh


def create_lod_hierarchy(mesh, lod_levels=None):
    """
    Create hierarchical LOD (Level of Detail) meshes.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): Original mesh
        lod_levels (dict): LOD level definitions (auto-generated if None)
        
    Returns:
        dict: Dictionary of meshes by LOD level
    """
    if mesh is None:
        return {}
    
    analysis = analyze_mesh_complexity(mesh)
    
    if lod_levels is None:
        if analysis['triangle_count'] > 100000:
            lod_levels = {
                "ultra_high": 1.0,
                "high": 0.7,
                "medium": 0.4,
                "default": 0.25,
                "low": 0.12,
                "ultra_low": 0.05
            }
        elif analysis['triangle_count'] > 50000:
            lod_levels = {
                "ultra_high": 1.0,
                "high": 0.6,
                "medium": 0.35,
                "default": 0.2,
                "low": 0.1,
                "ultra_low": 0.05
            }
        else:
            lod_levels = {
                "ultra_high": 1.0,
                "high": 0.7,
                "medium": 0.5,
                "default": 0.3,
                "low": 0.15,
                "ultra_low": 0.08
            }
    
    print(f"\n=== LOD Hierarchy Generation ===")
    print(f"Original mesh: {analysis['triangle_count']:,} triangles")
    
    lod_meshes = {}
    
    for lod_name, ratio in lod_levels.items():
        print(f"\n{lod_name.upper()} LOD generating... (ratio: {ratio*100:.0f}%)")
        
        if ratio >= 0.95:
            lod_mesh = copy.deepcopy(mesh)
            lod_mesh.remove_degenerate_triangles()
            lod_mesh.remove_duplicated_vertices()
        else:
            lod_mesh = smart_vertex_reduction(mesh, target_ratio=ratio, 
                                            quality_priority=(ratio > 0.3))
        
        lod_mesh.compute_vertex_normals()
        lod_mesh.compute_triangle_normals()
        
        lod_meshes[lod_name] = lod_mesh
        
        print(f"  Complete: {len(lod_mesh.vertices):,} vertices, {len(lod_mesh.triangles):,} triangles")
    
    return lod_meshes


def measure_optimization_quality(original_mesh, optimized_mesh):
    """
    Measure optimization quality.
    
    Args:
        original_mesh (o3d.geometry.TriangleMesh): Original mesh
        optimized_mesh (o3d.geometry.TriangleMesh): Optimized mesh
        
    Returns:
        dict: Quality measurement results
    """
    if original_mesh is None or optimized_mesh is None:
        return {}
    
    orig_vertices = len(original_mesh.vertices)
    opt_vertices = len(optimized_mesh.vertices)
    orig_triangles = len(original_mesh.triangles)
    opt_triangles = len(optimized_mesh.triangles)
    
    vertex_reduction = (orig_vertices - opt_vertices) / orig_vertices * 100
    triangle_reduction = (orig_triangles - opt_triangles) / orig_triangles * 100
    
    orig_area = original_mesh.get_surface_area()
    opt_area = optimized_mesh.get_surface_area()
    area_preservation = min(opt_area / orig_area, orig_area / opt_area) * 100
    
    volume_preservation = 100.0
    if original_mesh.is_watertight() and optimized_mesh.is_watertight():
        orig_volume = original_mesh.get_volume()
        opt_volume = optimized_mesh.get_volume()
        if orig_volume > 0:
            volume_preservation = min(opt_volume / orig_volume, orig_volume / opt_volume) * 100
    
    quality_score = (area_preservation + volume_preservation) / 2
    
    return {
        'vertex_reduction_percent': vertex_reduction,
        'triangle_reduction_percent': triangle_reduction,
        'area_preservation_percent': area_preservation,
        'volume_preservation_percent': volume_preservation,
        'overall_quality_score': quality_score,
        'compression_ratio': orig_triangles / max(1, opt_triangles)
    }


def save_optimized_mesh(mesh, output_dir, filename, quality_info=None):
    """
    최적화된 메시를 저장합니다.
    
    Args:
        mesh (o3d.geometry.TriangleMesh): 저장할 메시
        output_dir (str): 출력 디렉토리
        filename (str): 파일명 (확장자 제외)
        quality_info (dict): 품질 정보 (선택사항)
        
    Returns:
        list: 저장된 파일 경로 리스트
    """
    if mesh is None:
        return []
    
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []
    
    # 다양한 형식으로 저장
    formats = [
        ('.obj', '범용 3D 모델 형식'),
        ('.ply', '포인트 클라우드 형식'),
        ('.stl', '3D 프린팅 형식')
    ]
    
    for ext, description in formats:
        filepath = os.path.join(output_dir, f"{filename}{ext}")
        try:
            success = o3d.io.write_triangle_mesh(filepath, mesh)
            if success:
                saved_files.append(filepath)
                print(f"Saved ({description}): {filepath}")
            else:
                print(f"Save failed: {filepath}")
        except Exception as e:
            print(f"Error saving ({ext}): {e}")
    
    if quality_info and saved_files:
        info_path = os.path.join(output_dir, f"{filename}_optimization_info.txt")
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                f.write("=== Mesh Optimization Info ===\n\n")
                f.write(f"Vertex reduction: {quality_info.get('vertex_reduction_percent', 0):.1f}%\n")
                f.write(f"Triangle reduction: {quality_info.get('triangle_reduction_percent', 0):.1f}%\n")
                f.write(f"Surface area preservation: {quality_info.get('area_preservation_percent', 0):.1f}%\n")
                f.write(f"Volume preservation: {quality_info.get('volume_preservation_percent', 0):.1f}%\n")
                f.write(f"Overall quality score: {quality_info.get('overall_quality_score', 0):.1f}/100\n")
                f.write(f"Compression ratio: {quality_info.get('compression_ratio', 1):.2f}:1\n")
            print(f"Optimization info saved: {info_path}")
        except Exception as e:
            print(f"Error saving info file: {e}")
    
    return saved_files
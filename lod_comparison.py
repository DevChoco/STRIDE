#!/usr/bin/env python3
"""
LOD 메시 파일 비교 도구
"""

import os
import sys

def analyze_obj_file(filepath):
    """OBJ 파일의 버텍스, 면 개수를 분석합니다."""
    if not os.path.exists(filepath):
        return None
    
    vertices = 0
    faces = 0
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('v '):  # 버텍스
                    vertices += 1
                elif line.startswith('f '):  # 면
                    faces += 1
    except:
        return None
    
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
    
    return {
        'vertices': vertices,
        'faces': faces,
        'file_size_mb': file_size
    }

def main():
    base_dir = r"d:\Lab2\3D_Body_Posture_Analysis_FPFH_nonSMPLX\output\3d_models"
    
    files_to_analyze = [
        ("body_mesh_fpfh_ultra_high.obj", "Ultra High (최고 품질)"),
        ("body_mesh_fpfh_high.obj", "High (고품질)"),
        ("body_mesh_fpfh_medium.obj", "Medium (중품질)"), 
        ("body_mesh_fpfh_low.obj", "Low (저품질)"),
        ("body_mesh_fpfh_ultra_low.obj", "Ultra Low (최저 품질)"),
        ("body_mesh_fpfh.obj", "Standard (기본)")
    ]
    
    print("="*80)
    print("     LOD(Level of Detail) 메시 파일 비교")
    print("="*80)
    print()
    print(f"{'파일명':<30} {'품질 레벨':<20} {'버텍스':<10} {'면':<10} {'크기(MB)':<10}")
    print("-" * 80)
    
    for filename, description in files_to_analyze:
        filepath = os.path.join(base_dir, filename)
        analysis = analyze_obj_file(filepath)
        
        if analysis:
            print(f"{filename:<30} {description:<20} {analysis['vertices']:<10,} {analysis['faces']:<10,} {analysis['file_size_mb']:<10.2f}")
        else:
            print(f"{filename:<30} {description:<20} {'파일 없음':<10} {'파일 없음':<10} {'파일 없음':<10}")
    
    print()
    print("="*80)
    print("LOD 설명:")
    print("- Ultra High: 원본 품질 (100%) - 최대 디테일, 가장 큰 용량")
    print("- High: 고품질 (약 50% 감소) - 높은 디테일, 적당한 용량")
    print("- Medium: 중품질 (약 70% 감소) - 보통 디테일, 보통 용량")
    print("- Low: 저품질 (약 80% 감소) - 낮은 디테일, 작은 용량")
    print("- Ultra Low: 최저품질 (약 92% 감소) - 최소 디테일, 최소 용량")
    print()
    print("용도:")
    print("- Ultra High: 고품질 렌더링, 정밀 분석")
    print("- High: 일반 렌더링, 상세 확인")
    print("- Medium: 실시간 시각화")
    print("- Low: 빠른 프리뷰, 모바일")
    print("- Ultra Low: 실시간 처리, 웹 전송")
    print("="*80)

if __name__ == "__main__":
    main()
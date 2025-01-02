"""
Initialization package for Visual Odometry.

Handles the bootstrap phase of VO using SIFT features.
"""

from .bootstrap import (
    InitializationResult,
    initialize_vo,
    detect_sift_keypoints,
    match_keypoints,
    estimate_pose,
    triangulate_points
)

__all__ = [
    'InitializationResult',
    'initialize_vo',
    'detect_sift_keypoints',
    'match_keypoints',
    'estimate_pose',
    'triangulate_points'
]
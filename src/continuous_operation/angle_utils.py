"""
Angle calculation utilities for Visual Odometry.
"""

import numpy as np
from typing import Tuple


def calculate_angles(current_points: np.ndarray,
                    first_observed_points: np.ndarray,
                    T_current: np.ndarray,
                    T_first: np.ndarray,
                    K: np.ndarray) -> np.ndarray:
    """
    Calculate angles between first observed points and current points.
    
    Args:
        current_points: Nx2 array of current point coordinates
        first_observed_points: Nx2 array of points when first observed
        T_current: 3x4 or 4x4 current camera pose
        T_first: Nx12 array of camera poses when points were first observed
        K: 3x3 camera intrinsic matrix
    
    Returns:
        Array of N angles (in radians)
    """
    # Ensure inputs are numpy arrays with correct shape
    current_points = np.asarray(current_points)
    first_observed_points = np.asarray(first_observed_points)
    K = np.asarray(K)
    
    num_points = current_points.shape[0]
    
    # Convert points to normalized camera coordinates
    current_homogeneous = np.hstack([current_points, np.ones((num_points, 1))])
    first_homogeneous = np.hstack([first_observed_points, np.ones((num_points, 1))])
    
    # Apply K inverse to normalize coordinates
    current_normalized = (np.linalg.inv(K) @ current_homogeneous.T).T
    first_normalized = (np.linalg.inv(K) @ first_homogeneous.T).T
    
    # Extract rotation matrices
    R_current = T_current[:3, :3].T  # Transpose to get camera to world
    
    # Calculate angles for each point
    angles = np.zeros(num_points)
    
    for i in range(num_points):
        # Get rotation matrix from first observation
        T_first_i = T_first[i].reshape(3, 4)
        R_first_i = T_first_i[:3, :3].T
        
        # Calculate relative rotation
        R_relative = R_first_i.T @ R_current
        
        # Get normalized vectors
        v1 = current_normalized[i]
        v2 = (R_relative @ first_normalized[i])
        
        # Calculate angle using dot product
        cos_angle = np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
            -1.0, 1.0
        )
        angles[i] = np.arccos(cos_angle)
    
    return angles


def filter_points_by_angle(current_points: np.ndarray,
                         first_observed_points: np.ndarray,
                         T_current: np.ndarray,
                         T_first: np.ndarray,
                         K: np.ndarray,
                         min_angle_rad: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter points based on the angle between first and current observation.
    
    Args:
        current_points: Nx2 array of current point coordinates
        first_observed_points: Nx2 array of points when first observed
        T_current: 3x4 or 4x4 current camera pose
        T_first: Nx12 array of camera poses when points were first observed
        K: 3x3 camera intrinsic matrix
        min_angle_rad: Minimum angle in radians for valid points
        
    Returns:
        Tuple of (mask, angles) where mask is a boolean array indicating valid points
        and angles contains the calculated angles for all points
    """
    angles = calculate_angles(current_points, first_observed_points, 
                            T_current, T_first, K)
    mask = np.abs(angles) > min_angle_rad
    
    return mask, angles
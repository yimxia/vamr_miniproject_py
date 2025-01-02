"""
Initialization module for Visual Odometry using SIFT features.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt

@dataclass
class InitializationResult:
    """Result from initialization phase"""
    R_C2_W: np.ndarray      # 3x3 rotation matrix from camera 2 to world frame
    t_C2_W: np.ndarray      # 3x1 translation vector from camera 2 to world frame
    keypoints_init: np.ndarray  # 2xN array of initial keypoints
    P3D_init: np.ndarray    # 3xN array of initial 3D points
    descriptors_init: np.ndarray  # NxD array of SIFT descriptors


def detect_sift_keypoints(img1: np.ndarray, 
                         img2: np.ndarray,
                         config: dict) -> Tuple[List[cv2.KeyPoint], List[cv2.KeyPoint], np.ndarray, np.ndarray]:
    """
    Detect SIFT keypoints and compute descriptors for two images.
    """
    # Create SIFT detector
    sift = cv2.SIFT_create(
        nfeatures=config.get('n_features', 2000),
        nOctaveLayers=config.get('n_octave_layers', 3),
        contrastThreshold=config.get('contrast_threshold', 0.04),
        edgeThreshold=config.get('edge_threshold', 10),
        sigma=config.get('sigma', 1.6)
    )
    
    # Detect and compute descriptors
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    
    print(f"Detected {len(kp1)} keypoints in first image")
    print(f"Detected {len(kp2)} keypoints in second image")
    
    if config.get('plot_initialization', False):
        # Draw keypoints
        img1_kp = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        img2_kp = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        plt.imshow(img1_kp)
        plt.title('SIFT Keypoints - Image 1')
        plt.subplot(122)
        plt.imshow(img2_kp)
        plt.title('SIFT Keypoints - Image 2')
        plt.show()
    
    return kp1, kp2, desc1, desc2


def match_keypoints(kp1: List[cv2.KeyPoint],
                   kp2: List[cv2.KeyPoint],
                   desc1: np.ndarray,
                   desc2: np.ndarray,
                   img1: np.ndarray,
                   img2: np.ndarray,
                   config: dict) -> Tuple[np.ndarray, np.ndarray, List]:
    """
    Match keypoints between images using Lowe's ratio test.
    
    Returns:
        pts1: Matched points in first image
        pts2: Matched points in second image
        matches: List of DMatch objects
    """
    # Create BF matcher
    bf = cv2.BFMatcher()
    
    # Match descriptors
    matches = bf.knnMatch(desc1, desc2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    pts1 = []
    pts2 = []
    
    ratio_threshold = config.get('match_ratio', 0.8)
    
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            good_matches.append(m)
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    
    print(f"Found {len(good_matches)} good matches")
    
    if config.get('plot_initialization', False):
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(15, 5))
        plt.imshow(match_img)
        plt.title('Matched Features')
        plt.show()
    
    return pts1, pts2, good_matches


def estimate_pose(pts1: np.ndarray,
                 pts2: np.ndarray,
                 K: np.ndarray,
                 config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate relative pose using essential matrix.
    """
    # Find essential matrix
    E, mask = cv2.findEssentialMat(
        pts2, pts1, K,
        method=cv2.RANSAC,
        prob=config.get('ransac_confidence', 0.999)/100.0,
        threshold=config.get('error_threshold', 1.0),
        maxIters=config.get('ransac_numtrials', 32000)
    )
    
    mask = mask.ravel()
    
    # Select inlier points
    pts1_inlier = pts1[mask == 1]
    pts2_inlier = pts2[mask == 1]
    
    print(f"Found {np.sum(mask)} inliers from {len(mask)} matches")
    
    # Recover pose from essential matrix
    _, R, t, pose_mask = cv2.recoverPose(E, pts2_inlier, pts1_inlier, K)
    
    pose_mask = pose_mask.ravel()
    
    # Create final mask
    final_mask = np.zeros_like(mask, dtype=bool)
    final_mask[mask == 1] = pose_mask > 0
    
    return R, t, final_mask


def triangulate_points(pts1: np.ndarray,
                      pts2: np.ndarray,
                      K: np.ndarray,
                      R: np.ndarray,
                      t: np.ndarray,
                      config: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from matched keypoints.
    """
    # Create projection matrices
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    
    # Triangulate points
    pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts_3d = (pts_4d[:3] / pts_4d[3]).T
    
    # Verify points
    valid_pts = np.ones(len(pts_3d), dtype=bool)
    
    # Check depth
    depths = pts_3d[:, 2]
    valid_pts &= (depths > 0) & (depths < config.get('furthest_triangulate', 100.0))
    
    # Check reprojection error
    proj1 = project_points(pts_3d, np.eye(3), np.zeros(3), K)
    proj2 = project_points(pts_3d, R, t.ravel(), K)
    
    error1 = np.linalg.norm(proj1 - pts1, axis=1)
    error2 = np.linalg.norm(proj2 - pts2, axis=1)
    
    max_error = config.get('max_reprojection_error', 2.0)
    valid_pts &= (error1 < max_error) & (error2 < max_error)
    
    print(f"Triangulated {np.sum(valid_pts)} valid points from {len(pts_3d)} matches")
    
    return pts_3d, valid_pts


def project_points(points_3d: np.ndarray,
                  R: np.ndarray,
                  t: np.ndarray,
                  K: np.ndarray) -> np.ndarray:
    """Project 3D points to 2D using camera parameters."""
    # Transform points to camera frame
    points_cam = (R @ points_3d.T + t.reshape(3, 1))
    
    # Project to image plane
    points_img = K @ points_cam
    
    # Convert to homogeneous coordinates
    points_img = points_img[:2] / points_img[2]
    
    return points_img.T


def initialize_vo(img0: np.ndarray,
                 img1: np.ndarray,
                 img2: np.ndarray,
                 config: dict) -> InitializationResult:
    """
    Initialize the Visual Odometry pipeline using SIFT features.
    """
    # Validate inputs
    if img0 is None or img1 is None or img2 is None:
        raise ValueError("Input images cannot be None")
    
    # Get camera matrix
    K = config.get_camera_matrix()
    print("Starting initialization...")
    
    try:
        # 1. Detect SIFT keypoints and descriptors
        kp1, kp2, desc1, desc2 = detect_sift_keypoints(img1, img2, config)
        
        # 2. Match keypoints
        pts1, pts2, matches = match_keypoints(kp1, kp2, desc1, desc2, img1, img2, config)
        
        if len(pts1) < 8:
            raise RuntimeError(f"Not enough matches: {len(pts1)}")
        
        # Get indices for descriptor mapping
        desc2_indices = np.array([m.trainIdx for m in matches])
        
        # 3. Estimate relative pose
        R, t, valid_points = estimate_pose(pts1, pts2, K, config)
        
        # 4. Triangulate 3D points
        valid_pts1 = pts1[valid_points]
        valid_pts2 = pts2[valid_points]
        valid_desc2 = desc2[desc2_indices[valid_points]]
        
        P3D, valid_3d = triangulate_points(
            valid_pts1,
            valid_pts2,
            K, R, t, config
        )
        
        # 5. Keep only valid points
        keypoints = valid_pts2[valid_3d]
        descriptors = valid_desc2[valid_3d]
        landmarks = P3D[valid_3d]
        
        print(f"Final initialization: {len(keypoints)} points")
        
        # 6. Return initialization result with correct shapes
        return InitializationResult(
            R_C2_W=R,
            t_C2_W=t,
            keypoints_init=keypoints.T,  # 2xN
            P3D_init=landmarks.T,        # 3xN
            descriptors_init=descriptors  # NxD
        )
        
    except Exception as e:
        print(f"Initialization failed: {str(e)}")
        raise

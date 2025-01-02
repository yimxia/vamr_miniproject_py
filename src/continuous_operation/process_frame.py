"""
Frame-to-frame tracking and pose estimation for Visual Odometry using Optical Flow and feature detection with validation.
"""
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import matplotlib.pyplot as plt

@dataclass
class State:
    """Visual Odometry state"""
    keypoints: np.ndarray          # 2xN array of keypoint coordinates
    landmarks: np.ndarray          # 3xN array of 3D points
    descriptors: np.ndarray        # NxD array of SIFT descriptors (optional if using optical flow)
    candidates: np.ndarray         # 2xM array of candidate points
    candidate_counts: np.ndarray   # M array of observation counts
    first_observations: np.ndarray # 2xM array of first observations
    first_poses: np.ndarray        # Mx12 array of camera poses at first observation

def track_keypoints(prev_img, curr_img, prev_pts):
    """
    Track keypoints using optical flow.

    Args:
        prev_img: Previous grayscale image
        curr_img: Current grayscale image
        prev_pts: Previous keypoints (Nx2 array)

    Returns:
        curr_pts: Tracked keypoints in the current image
        valid_prev_pts: Corresponding keypoints in the previous image
    """
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
    )
    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_img, curr_img, prev_pts, None, **lk_params)

    # Filter points with valid tracking
    valid_mask = status.ravel() == 1
    return curr_pts[valid_mask], prev_pts[valid_mask]

def process_frame(curr_img: np.ndarray,
                  prev_img: np.ndarray,
                  prev_state: State,
                  K: np.ndarray,
                  config: dict,
                  prev_R: np.ndarray,
                  prev_t: np.ndarray) -> Tuple[State, np.ndarray]:
    """
    改良的帧处理函数，优化关键点分布和跨帧轨迹估计。
    """
    try:
        # 初始化特征提取器
        sift = cv2.SIFT_create(
            nfeatures=config.get('n_features', 5000),
            contrastThreshold=config.get('contrast_threshold', 0.04),
            edgeThreshold=config.get('edge_threshold', 10)
        )
        orb = cv2.ORB_create(nfeatures=config.get('n_features_orb', 2000))
        fast = cv2.FastFeatureDetector_create(threshold=config.get('fast_threshold', 20), nonmaxSuppression=True)

        # 1. 跟踪关键点
        prev_pts = prev_state.keypoints.T.astype(np.float32)
        curr_pts, valid_prev_pts = track_keypoints(prev_img, curr_img, prev_pts)
        print(f"Initial tracked points: {curr_pts.shape}, Valid previous points: {valid_prev_pts.shape}")

        if len(curr_pts) < config.get('keypoints_min', 100):
            print("Tracked points are insufficient. Re-detecting keypoints...")

            # 多特征检测器结合
            new_kp_sift, _ = sift.detectAndCompute(curr_img, None)
            new_pts_sift = np.array([kp.pt for kp in new_kp_sift], dtype=np.float32)

            new_kp_orb = orb.detect(curr_img, None)
            new_pts_orb = np.array([kp.pt for kp in new_kp_orb], dtype=np.float32)

            new_kp_fast = fast.detect(curr_img, None)
            new_pts_fast = np.array([kp.pt for kp in new_kp_fast], dtype=np.float32)

            # 合并所有关键点
            new_pts = np.vstack([new_pts_sift, new_pts_orb, new_pts_fast])

            # 动态随机采样与均匀化
            random_sample_size = config.get('random_sample_size', 300)
            if len(new_pts) > random_sample_size:
                np.random.shuffle(new_pts)
                new_pts = new_pts[:random_sample_size]

            grid_size = config.get('grid_size', (20, 20))
            min_per_grid = config.get('min_per_grid', 5)
            max_per_grid = config.get('max_per_grid', 20)
            new_pts = enforce_keypoint_uniformity(new_pts, curr_img.shape, grid_size, min_per_grid, max_per_grid)

            # 验证新关键点
            validated_new_pts, _ = track_keypoints(curr_img, prev_img, new_pts)

            max_new_points = config.get('max_new_points', 1000)
            if len(validated_new_pts) > max_new_points:
                validated_new_pts = validated_new_pts[:max_new_points]

            curr_pts = np.vstack([curr_pts, validated_new_pts])

        print(f"Tracked points after detection: {curr_pts.shape}")

        # 2. 获取对应的3D点
        valid_landmarks = prev_state.landmarks.T[:len(valid_prev_pts)]
        print(f"Valid landmarks before adjustment: {valid_landmarks.shape}")

        if len(valid_landmarks) > len(curr_pts):
            valid_landmarks = valid_landmarks[:len(curr_pts)]
        elif len(curr_pts) > len(valid_landmarks):
            curr_pts = curr_pts[:len(valid_landmarks)]

        assert len(curr_pts) == len(valid_landmarks), f"Keypoint and landmark size mismatch: {curr_pts.shape} vs {valid_landmarks.shape}"

        print(f"Valid landmarks after adjustment: {valid_landmarks.shape}, Tracked points: {curr_pts.shape}")

        # 3. 使用PnP + RANSAC估计姿态
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            valid_landmarks,
            curr_pts,
            K,
            None,
            confidence=config.get('ransac_confidence', 0.99),
            iterationsCount=config.get('ransac_numtrials', 1000),
            reprojectionError=config.get('error_threshold', 1.0),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None:
            print("Failed to estimate pose. Using previous pose.")
            R_C_W, t_C_W = prev_R, prev_t
        else:
            print(f"Found {len(inliers)} inliers")
            inliers = inliers.ravel()
            assert inliers.max() < len(curr_pts), f"Inliers index out of range: max={inliers.max()}, points={len(curr_pts)}"
            R_C_W, _ = cv2.Rodrigues(rvec)
            t_C_W = tvec

        print(f"Rotation matrix shape: {R_C_W.shape}, Translation vector shape: {t_C_W.shape}")

        if prev_t.ndim == 1:
            prev_t = prev_t.reshape(-1, 1)
        if t_C_W.ndim == 1:
            t_C_W = t_C_W.reshape(-1, 1)

        T_i_wc = smooth_pose_estimation(prev_R, prev_t, R_C_W, t_C_W, alpha=config.get('smooth_alpha', 0.8))

        inlier_mask = np.zeros(len(curr_pts), dtype=bool)
        inlier_mask[inliers] = True
        curr_pts = curr_pts[inlier_mask]
        valid_landmarks = valid_landmarks[inlier_mask]

        assert curr_pts.shape[0] == valid_landmarks.shape[0], f"Mismatch after inlier filtering: {curr_pts.shape} vs {valid_landmarks.shape}"

        print(f"Final inliers: {curr_pts.shape}, Landmarks: {valid_landmarks.shape}")

        if len(curr_pts) < config.get('min_inliers', 50):
            print("Insufficient inliers. Augmenting keypoints with new detections.")
            new_kp, _ = sift.detectAndCompute(curr_img, None)
            new_pts = np.array([kp.pt for kp in new_kp], dtype=np.float32)
            if len(new_pts) > config.get('max_new_points', 500):
                new_pts = new_pts[:config.get('max_new_points', 500)]
            curr_pts = np.vstack([curr_pts, new_pts])

        new_state = State(
            keypoints=curr_pts.T,
            landmarks=valid_landmarks.T,
            descriptors=None,
            candidates=prev_state.candidates,
            candidate_counts=prev_state.candidate_counts,
            first_observations=prev_state.first_observations,
            first_poses=prev_state.first_poses
        )

        return new_state, T_i_wc

    except Exception as e:
        print(f"Error in process_frame: {str(e)}")
        raise

def enforce_keypoint_uniformity(keypoints: np.ndarray, img_shape: Tuple[int, int], grid_size: Tuple[int, int], min_per_grid: int, max_per_grid: int) -> np.ndarray:
    """
    确保关键点在图像中均匀分布，并限制每个网格内的最大和最小关键点数量
    """
    h, w = img_shape[:2]
    grid_h, grid_w = grid_size
    cell_h, cell_w = h // grid_h, w // grid_w

    uniform_keypoints = []
    for i in range(grid_h):
        for j in range(grid_w):
            y_min, y_max = i * cell_h, (i + 1) * cell_h
            x_min, x_max = j * cell_w, (j + 1) * cell_w

            grid_points = keypoints[(keypoints[:, 0] >= x_min) & (keypoints[:, 0] < x_max) &
                                    (keypoints[:, 1] >= y_min) & (keypoints[:, 1] < y_max)]

            if len(grid_points) < min_per_grid:
                uniform_keypoints.extend(grid_points)
            elif len(grid_points) > max_per_grid:
                uniform_keypoints.extend(grid_points[:max_per_grid])
            else:
                uniform_keypoints.extend(grid_points)

    return np.array(uniform_keypoints, dtype=np.float32)

def smooth_pose_estimation(prev_R, prev_t, curr_R, curr_t, alpha=0.8):
    """
    对相机姿态进行平滑处理
    """
    smoothed_R = alpha * prev_R + (1 - alpha) * curr_R
    smoothed_t = alpha * prev_t + (1 - alpha) * curr_t
    return np.hstack((smoothed_R, smoothed_t))

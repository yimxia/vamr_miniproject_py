"""
Configuration parameters for Visual Odometry pipeline.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import cv2

@dataclass
class VOConfig:
    """Configuration parameters for Visual Odometry"""
    
    # Dataset selection
    dataset_type: int = 0 # 0:KITTI 1:Malaga 2:Parking

    # Bootstrap parameters
    bootstrap_frames: List[int] = field(default_factory=lambda: [1, 2, 3])
    keypoints_min: int = 100
    max_candidates: int = 500
    
    # SIFT parameters
    n_features = 9999  # 增加最大特征点数量
    contrast_threshold = 0.02  # 降低对比度阈值，检测更多特征
    edge_threshold = 20  # 放宽边缘过滤阈值
    match_ratio = 0.85  # 更严格的Lowe比率测试
    error_threshold = 5.0  # 放宽重投影误差

    
    # Feature matching parameters
    match_ratio: float = 0.75  # Lowe's ratio test threshold
    min_matches: int = 50
    
    # RANSAC parameters
    ransac_confidence: float = 0.999
    ransac_numtrials: int = 1000
    error_threshold: float = 3.0  # 增加误差阈值
    min_inlier_ratio: float = 0.3
    furthest_triangulate = 500.0  # 放宽深度限制
    max_reprojection_error = 10.0 
    
    # Triangulation parameters
    furthest_triangulate: float = 200.0  # 增加最大深度
    max_reprojection_error: float = 5.0  # 增加重投影误差阈值
    min_angle_deg: float = 2.0  # 降低最小角度要求
    min_consecutive_frames: int = 3  # 减少连续帧要求
    
    # Visualization flags
    plot_initialization: bool = True
    plot_matches: bool = True
    plot_current: bool = True
    plot_local: bool = True
    plot_keypoints_candidates: bool = True
    plot_cameras: bool = True
    plot_correspondences: bool = True
    plot_new: bool = True
    plot_final: bool = True
    
    def __post_init__(self):
        """Convert parameters to dictionary for easier access"""
        self.params = {
            'keypoints_min': self.keypoints_min,
            'max_candidates': self.max_candidates,
            'n_features': self.n_features,
            # 'n_octave_layers': self.n_octave_layers,
            'contrast_threshold': self.contrast_threshold,
            'edge_threshold': self.edge_threshold,
            'match_ratio': self.match_ratio,
            'min_matches': self.min_matches,
            'ransac_confidence': self.ransac_confidence,
            'ransac_numtrials': self.ransac_numtrials,
            'error_threshold': self.error_threshold,
            'min_inlier_ratio': self.min_inlier_ratio,
            'furthest_triangulate': self.furthest_triangulate,
            'max_reprojection_error': self.max_reprojection_error,
            'min_angle_deg': self.min_angle_deg,
            'min_consecutive_frames': self.min_consecutive_frames
        }
        
    def __getitem__(self, key):
        """Allow dictionary-like access to parameters"""
        return self.params.get(key, None)
    
    def get(self, key, default=None):
        """Get parameter value with default"""
        return self.params.get(key, default)

    def get_camera_matrix(self) -> np.ndarray:
        """Returns the camera intrinsic matrix based on dataset type"""
        if self.dataset_type == 0:  # KITTI
            return np.array([
                [718.856, 0, 607.1928],
                [0, 718.856, 185.2157],
                [0, 0, 1]
            ])
        elif self.dataset_type == 1:  # Malaga
            return np.array([
                [621.18428, 0, 404.0076],
                [0, 621.18428, 309.05989],
                [0, 0, 1]
            ])
        elif self.dataset_type == 2:  # Parking
            return np.array([
                [845.52896472, 0, 468.10562767],
                [0, 835.35434918, 298.01291054],
                [0, 0, 1]
            ])
        else:  # countrylife and vineyards
            return np.array([
                [845.52896472, 0, 468.10562767],
                [0, 835.35434918, 298.01291054],
                [0, 0, 1]
            ])

    def get_dataset_path(self) -> str:
        """Returns the dataset path based on dataset type"""
        base_path = "datasets"
        if self.dataset_type == 0:
            return f"{base_path}/kitti"
        elif self.dataset_type == 1:
            return f"{base_path}/malaga-urban-dataset-extract-07"
        elif self.dataset_type == 2:
            return f"{base_path}/parking"
        elif self.dataset_type == 3:
            return f"{base_path}/countrylife"
        else:
            return f"{base_path}/vineyards"

    def get_last_frame(self) -> int:
        """Returns the last frame number based on dataset type"""
        if self.dataset_type == 0:
            return 2300
        elif self.dataset_type == 1:
            return float('inf')  # Should be determined from actual dataset
        elif self.dataset_type == 2:
            return 598
        elif self.dataset_type == 3:
            return 302
        else:
            return 525
            
    def create_sift(self) -> cv2.SIFT_create:
        """Create SIFT detector with configured parameters"""
        return cv2.SIFT_create(
            nfeatures=self.n_features,
            nOctaveLayers=self.n_octave_layers,
            contrastThreshold=self.contrast_threshold,
            edgeThreshold=self.edge_threshold,
            sigma=self.sigma
        )
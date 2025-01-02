"""
Main program for running Visual Odometry pipeline with SIFT features.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
from matplotlib.gridspec import GridSpec

from src.config import VOConfig
from src.initialization import initialize_vo
from src.continuous_operation import State, process_frame


class VisualOdometry:
    def __init__(self, config: VOConfig):
        self.config = config
        self.trajectory_history = []
        self.landmarks_history = []
        self.num_landmarks_history = []
        self.setup_visualization()
        
    def setup_visualization(self):
        """Setup visualization windows"""
        plt.ion()  # Enable interactive plotting
        self.fig = plt.figure(figsize=(15, 10))
        self.gs = GridSpec(2, 2, figure=self.fig)
        
        # Current image with features
        self.ax_image = self.fig.add_subplot(self.gs[0, 0])
        self.ax_image.set_title('Current Frame Features')
        
        # Top view of trajectory and landmarks
        self.ax_trajectory = self.fig.add_subplot(self.gs[0, 1])
        self.ax_trajectory.set_title('Trajectory and Landmarks (Top View)')
        self.ax_trajectory.grid(True)
        
        # Number of tracked features
        self.ax_features = self.fig.add_subplot(self.gs[1, 0])
        self.ax_features.set_title('Number of Tracked Features')
        self.ax_features.grid(True)
        
        # Side view of trajectory
        self.ax_side = self.fig.add_subplot(self.gs[1, 1])
        self.ax_side.set_title('Trajectory (Side View)')
        self.ax_side.grid(True)
        
        plt.tight_layout()

    def update_visualization(self, 
                           image: np.ndarray, 
                           state: State, 
                           pose: np.ndarray):
        """Update all visualization plots"""
        # Clear all plots
        self.ax_image.clear()
        self.ax_trajectory.clear()
        self.ax_features.clear()
        self.ax_side.clear()
        
        # Update current image view
        if image is not None:
            self.ax_image.imshow(image, cmap='gray')
            if state.keypoints.size > 0:
                self.ax_image.plot(state.keypoints[0], state.keypoints[1], 'g.', 
                                 label='Keypoints')
            if state.candidates.size > 0:
                self.ax_image.plot(state.candidates[0], state.candidates[1], 'r.', 
                                 label='Candidates')
            self.ax_image.legend()
            self.ax_image.set_title(f'Features (Frame {len(self.trajectory_history)})')
        
        # Update trajectory and landmarks (top view)
        if len(self.trajectory_history) > 0:
            trajectory = np.array(self.trajectory_history)
            self.ax_trajectory.plot(trajectory[:, 0], trajectory[:, 2], 'b-')
            self.ax_trajectory.plot(trajectory[-1, 0], trajectory[-1, 2], 'rx')
            
            if len(self.landmarks_history) > 0:
                recent_landmarks = self.landmarks_history[-1]
                if recent_landmarks.size > 0:
                    self.ax_trajectory.scatter(recent_landmarks[0], 
                                            recent_landmarks[2],
                                            c='k', s=1, alpha=0.5)
            
            self.ax_trajectory.grid(True)
            self.ax_trajectory.set_title('Trajectory and Landmarks (Top View)')
            
        # Update feature count plot
        if len(self.num_landmarks_history) > 20:
            x = range(len(self.num_landmarks_history))
            self.ax_features.plot(x, self.num_landmarks_history, 'g-')
            self.ax_features.set_title('Number of Tracked Features')
            self.ax_features.grid(True)
            
        # Update side view
        if len(self.trajectory_history) > 0:
            trajectory = np.array(self.trajectory_history)
            self.ax_side.plot(trajectory[:, 2], trajectory[:, 1], 'b-')
            self.ax_side.plot(trajectory[-1, 2], trajectory[-1, 1], 'rx')
            self.ax_side.grid(True)
            self.ax_side.set_title('Trajectory (Side View)')
            
        plt.tight_layout()
        plt.pause(0.01)
        
    def run(self):
        """Run the Visual Odometry pipeline"""
        # Setup paths
        dataset_path = Path(self.config.get_dataset_path())
        assert dataset_path.exists(), f"Dataset path not found: {dataset_path}"
        
        K = self.config.get_camera_matrix()
        print(f"Using camera matrix:\n{K}")
        
        # Load bootstrap images
        bootstrap_frames = self.config.bootstrap_frames
        try:
            images = [self.load_image(dataset_path, i) for i in bootstrap_frames]
            print("Successfully loaded bootstrap images")
        except Exception as e:
            print(f"Failed to load bootstrap images: {e}")
            return
            
        # Initialize VO
        try:
            print("\nInitializing Visual Odometry...")
            result = initialize_vo(images[0], images[1], images[2], self.config)
            print(f"Initialization successful with {result.keypoints_init.shape[1]} keypoints")
        except Exception as e:
            print(f"Initialization failed: {e}")
            return
            
        # Initialize state
        current_state = State(
            keypoints=result.keypoints_init,
            landmarks=result.P3D_init,
            descriptors=result.descriptors_init,
            candidates=np.empty((2, 0)),
            candidate_counts=np.array([]),
            first_observations=np.empty((2, 0)),
            first_poses=np.empty((0, 12))
        )
        
        # Initialize trajectory
        T_current = np.eye(4)
        T_current[:3, :3] = result.R_C2_W
        T_current[:3, 3] = result.t_C2_W.ravel()
        self.trajectory_history.append(T_current[:3, 3])
        self.landmarks_history.append(current_state.landmarks)
        self.num_landmarks_history.append(current_state.keypoints.shape[1])
        
        # Process frames
        last_frame = self.config.get_last_frame()
        frame_range = range(bootstrap_frames[-1] + 1, last_frame + 1)
        
        for frame_idx in frame_range:
            print(f"\nProcessing frame {frame_idx}")
            
            try:
                # Load images
                curr_img = self.load_image(dataset_path, frame_idx)
                prev_img = self.load_image(dataset_path, frame_idx-1)
                
                # Process frame
                new_state, T_new = process_frame(
                    curr_img,
                    prev_img,
                    current_state,
                    K,
                    self.config.params,
                    T_current[:3, :3],
                    T_current[:3, 3]
                )
                
                # Update state and history
                current_state = new_state
                T_current = T_new
                
                self.trajectory_history.append(T_current[:3, 3])
                self.landmarks_history.append(current_state.landmarks)
                self.num_landmarks_history.append(current_state.keypoints.shape[1])
                
                # Update visualization
                self.update_visualization(curr_img, current_state, T_current)
                
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                continue
                
        plt.ioff()
        plt.show()
        
    def load_image(self, dataset_path: Path, frame_num: int) -> np.ndarray:
        """Load image from dataset"""
        try:
            if self.config.dataset_type == 0:  # KITTI
                img_path = dataset_path / "05" / "image_0" / f"{frame_num:06d}.png"
            elif self.config.dataset_type == 1:  # Malaga
                img_path = dataset_path / "malaga-urban-dataset-extract-07_rectified_800x600_Images" / f"{frame_num}.jpg"
            elif self.config.dataset_type == 2:  # Parking
                img_path = dataset_path / "images" / f"img_{frame_num:05d}.png"
            else:
                raise ValueError(f"Unknown dataset type: {self.config.dataset_type}")
                
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to load image: {img_path}")
            return img
            
        except Exception as e:
            raise RuntimeError(f"Error loading image {frame_num}: {e}")

def visualize_trajectory(landmarks, trajectory):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(landmarks[0], landmarks[1], landmarks[2], c='r', s=1, label='Landmarks')
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], c='b', label='Trajectory')
    ax.set_title('3D Landmarks and Camera Trajectory')
    plt.legend()
    plt.show()

def main():
    # Create configuration
    config = VOConfig()
    
    # Create and run VO system
    vo = VisualOdometry(config)
    vo.run()

if __name__ == "__main__":
    main()
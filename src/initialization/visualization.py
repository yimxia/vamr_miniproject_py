import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import Tuple, Optional, List

def create_coordinate_frame_vectors(rotation: np.ndarray, origin: np.ndarray, 
                                 length: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create vectors for coordinate frame visualization.
    
    Args:
        rotation: 3x3 rotation matrix
        origin: 3D position of the coordinate frame origin
        length: Length of the coordinate frame axes
        
    Returns:
        Tuple of start points and end points for coordinate frame axes
    """
    # Ensure inputs are numpy arrays
    rotation = np.asarray(rotation)
    origin = np.asarray(origin).reshape(3)
    
    # Create start points (origin repeated 3 times)
    starts = np.tile(origin, (3, 1))
    
    # Create end points by applying rotation and scaling
    ends = starts + length * rotation.T
    
    return starts, ends

def plot_coordinate_frame(ax: plt.Axes, rotation: np.ndarray, origin: np.ndarray, 
                        length: float = 1.0, colors: Optional[List[str]] = None,
                        arrow_size: float = 0.1) -> None:
    """
    Plot a 3D coordinate frame with arrows.
    
    Args:
        ax: Matplotlib 3D axis
        rotation: 3x3 rotation matrix
        origin: 3D position of the coordinate frame origin
        length: Length of the coordinate frame axes
        colors: List of colors for x, y, z axes
        arrow_size: Size of the arrow heads relative to axis length
    """
    if colors is None:
        colors = ['red', 'green', 'blue']
    
    starts, ends = create_coordinate_frame_vectors(rotation, origin, length)
    
    # Plot each axis
    for i, (start, end, color) in enumerate(zip(starts, ends, colors)):
        # Draw the main axis line
        ax.quiver(start[0], start[1], start[2],
                 end[0] - start[0], end[1] - start[1], end[2] - start[2],
                 color=color, arrow_length_ratio=arrow_size)

def create_3d_arrow(start: np.ndarray, end: np.ndarray, 
                   head_angle: float = 30.0, 
                   head_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create vertices for a 3D arrow.
    
    Args:
        start: Starting point of arrow
        end: End point of arrow
        head_angle: Angle of arrow head in degrees
        head_ratio: Ratio of head length to total arrow length
        
    Returns:
        Tuple of vertices for arrow body and head
    """
    # Convert inputs to numpy arrays
    start = np.asarray(start)
    end = np.asarray(end)
    
    # Calculate arrow direction and length
    direction = end - start
    length = np.linalg.norm(direction)
    unit_direction = direction / length
    
    # Calculate head length and radius
    head_length = length * head_ratio
    head_radius = head_length * np.tan(np.radians(head_angle))
    
    # Create orthogonal vectors for arrow head base
    if np.allclose(unit_direction, [0, 0, 1]) or np.allclose(unit_direction, [0, 0, -1]):
        ortho1 = np.array([1, 0, 0])
    else:
        ortho1 = np.cross(unit_direction, [0, 0, 1])
        ortho1 /= np.linalg.norm(ortho1)
    ortho2 = np.cross(unit_direction, ortho1)
    
    # Create vertices for arrow head
    head_start = end - head_length * unit_direction
    n_points = 32
    angles = np.linspace(0, 2*np.pi, n_points)
    circle_points = (head_radius * (ortho1.reshape(-1, 1) * np.cos(angles) + 
                                  ortho2.reshape(-1, 1) * np.sin(angles))).T
    head_base = head_start + circle_points
    
    # Create arrow body vertices
    body_radius = head_radius * 0.3
    body_points = (body_radius * (ortho1.reshape(-1, 1) * np.cos(angles) + 
                                ortho2.reshape(-1, 1) * np.sin(angles))).T
    body_start = start + body_points
    body_end = head_start + body_points
    
    return (body_start, body_end), (head_base, end)

def plot_3d_arrow(ax: plt.Axes, start: np.ndarray, end: np.ndarray, 
                 color: str = 'blue', alpha: float = 0.6) -> None:
    """
    Plot a 3D arrow using Poly3DCollection.
    
    Args:
        ax: Matplotlib 3D axis
        start: Starting point of arrow
        end: End point of arrow
        color: Color of the arrow
        alpha: Transparency of the arrow
    """
    (body_start, body_end), (head_base, head_tip) = create_3d_arrow(start, end)
    
    # Create polygon vertices for arrow body
    n_points = body_start.shape[0]
    body_verts = []
    for i in range(n_points-1):
        quad = [body_start[i], body_start[i+1], 
                body_end[i+1], body_end[i]]
        body_verts.append(quad)
    
    # Create polygon vertices for arrow head
    head_verts = []
    for i in range(n_points-1):
        triangle = [head_base[i], head_base[i+1], head_tip]
        head_verts.append(triangle)
    
    # Create Poly3DCollection objects
    body_poly = Poly3DCollection(body_verts, alpha=alpha, color=color)
    head_poly = Poly3DCollection(head_verts, alpha=alpha, color=color)
    
    # Add to axis
    ax.add_collection3d(body_poly)
    ax.add_collection3d(head_poly)

def plot_camera_pose(ax: plt.Axes, 
                    rotation: np.ndarray, 
                    translation: np.ndarray,
                    scale: float = 1.0,
                    label: Optional[str] = None,
                    camera_color: str = 'gray') -> None:
    """
    Plot camera pose with coordinate frame and a simple camera model.
    
    Args:
        ax: Matplotlib 3D axis
        rotation: 3x3 rotation matrix (camera to world)
        translation: 3D translation vector
        scale: Scale factor for visualization
        label: Text label for the camera
        camera_color: Color of the camera wireframe
    """
    # Plot coordinate frame
    plot_coordinate_frame(ax, rotation, translation, length=scale)
    
    # Create camera wireframe
    cam_points = np.array([
        [-1, -1, 1.5],  # Camera frustum corners
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [0, 0, 0]      # Camera center
    ]) * (scale * 0.5)
    
    # Transform camera points
    cam_points = (rotation @ cam_points.T + translation.reshape(3, 1)).T
    
    # Plot camera wireframe
    for i in range(4):
        j = (i + 1) % 4
        ax.plot([cam_points[i,0], cam_points[j,0]],
                [cam_points[i,1], cam_points[j,1]],
                [cam_points[i,2], cam_points[j,2]], color=camera_color)
        ax.plot([cam_points[i,0], cam_points[4,0]],
                [cam_points[i,1], cam_points[4,1]],
                [cam_points[i,2], cam_points[4,2]], color=camera_color)
    
    # Add label if provided
    if label is not None:
        ax.text(translation[0], translation[1], translation[2], 
                label, fontsize=10, color='black')

def set_3d_plot_properties(ax: plt.Axes, 
                          title: str = 'Camera Pose',
                          equal_axes: bool = True) -> None:
    """
    Set common properties for 3D plots.
    
    Args:
        ax: Matplotlib 3D axis
        title: Plot title
        equal_axes: Whether to set equal aspect ratio for all axes
    """
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    if equal_axes:
        # Set equal aspect ratio for all axes
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d()
        ])
        center = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        ax.set_xlim3d([center[0] - radius, center[0] + radius])
        ax.set_ylim3d([center[1] - radius, center[1] + radius])
        ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    # Enable grid
    ax.grid(True)
    ax.view_init(elev=20, azim=45)
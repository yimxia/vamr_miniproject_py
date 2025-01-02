"""
Visual Odometry Pipeline using SIFT features.
"""

from .config import VOConfig
from .initialization import initialize_vo, InitializationResult
from .continuous_operation import State, process_frame

__all__ = [
    # Configuration
    'VOConfig',
    
    # Initialization
    'initialize_vo',
    'InitializationResult',
    
    # Continuous Operation
    'State',
    'process_frame'
]

__version__ = '1.0.0'
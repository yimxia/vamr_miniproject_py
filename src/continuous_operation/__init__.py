"""
Continuous operation module for Visual Odometry.
"""

from .process_frame import process_frame, State

__all__ = [
    'process_frame',
    'State',
]

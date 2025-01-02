from setuptools import setup, find_packages

setup(
    name="visual_odometry",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'opencv-python>=4.5.0',
        'matplotlib>=3.4.0',
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Visual Odometry Pipeline",
    keywords="computer vision, visual odometry, slam",
    python_requires='>=3.7',
)
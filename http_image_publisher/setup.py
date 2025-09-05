from setuptools import setup
import os
from glob import glob

package_name = 'http_image_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files - corrected glob pattern
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Alternative: include all Python files in launch directory
        # (os.path.join('share', package_name, 'launch'),
        #     glob('launch/*.py')),
        # Install config files (optional)
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'opencv-python',
        'requests',
    ],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='HTTP stream to ROS2 image publisher',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'http_image_publisher = http_image_publisher.http_image_publisher:main',
        ],
    },
)

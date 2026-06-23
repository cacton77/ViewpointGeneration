import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'viewpoint_generation'


def files(pattern):
    """glob() but only regular files (skips __pycache__ and other dirs)."""
    return [f for f in glob(pattern) if os.path.isfile(f)]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), files('launch/*')),
        (os.path.join('share', package_name, 'config'), files('config/*')),
        (os.path.join('share', package_name, 'nodes'), files('nodes/*')),
    ],
    package_data={
        'viewpoint_generation': ['assets/*.stl'],
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='col',
    maintainer_email='actonc@uw.edu',
    description='TODO: Package description',
    license='Apache-2.0',

    entry_points={
        'console_scripts': [
            'viewpoint_generation_node = nodes.viewpoint_generation_node:main',
            'gui_node = nodes.gui:main',
            'viewpoint_traversal_node = nodes.viewpoint_traversal_node:main',
            'task_planning_node = nodes.task_planning_node:main'
        ],
    },
)

from setuptools import setup
import os
from glob import glob

package_name = 'aitstar_path_planner'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Adaptively Informed Trees (AIT*) path planning algorithm for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aitstar_planner_node = aitstar_path_planner.aitstar_planner_node:main',
            'aitstar_visualizer = aitstar_path_planner.aitstar_visualizer:main',
        ],
    },
)

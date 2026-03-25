from setuptools import setup
import os
from glob import glob

package_name = 'bitstar_path_planner'

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
    description='Batch Informed Trees (BIT*) path planning algorithm for ROS2',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bitstar_planner_node = bitstar_path_planner.bitstar_planner_node:main',
            'bitstar_visualizer = bitstar_path_planner.bitstar_visualizer:main',
        ],
    },
)

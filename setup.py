from glob import glob
from setuptools import find_packages, setup

package_name = 'lidar_detector_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(
        include=[
            "lidar_detector_pkg",
            "lidar_detector_pkg.*",
            "lidar_detector_scripts"],
        exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', ['launch/lidar_detect.launch.py'])
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'open3d',
        'mmdet3d',
        'markerarraystamped'
        ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='ernestoroque777@gmail.com',
    description='This package aims to create a solid lidar-based 3d detector for humans from Kitti dataset using a PointPillars DL model.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lidar_detector_node = lidar_detector_pkg.lidar_detector_node:main',
            'proyection_node = lidar_detector_pkg.proyection_node:main'
        ],
    },
)

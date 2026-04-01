from setuptools import find_packages, setup

package_name = 'module2_visual_odometry'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ajay',
    maintainer_email='ajay@drone.dev',
    description='Phase 2: Visual Odometry',
    license='MIT',
    entry_points={
        'console_scripts': [
            'epipolar_geometry  = module2_visual_odometry.epipolar_geometry:main',
            'circle_flight      = module2_visual_odometry.circle_flight:main',
            'spawn_landmarks    = module2_visual_odometry.spawn_landmarks:main',
            'monocular_vo       = module2_visual_odometry.monocular_vo:main',
        ],
    },
)

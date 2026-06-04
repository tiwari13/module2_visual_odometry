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
    description='Mission planners (circle_flight) and benchmark world setup (spawn_landmarks) for the drone test stack.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'circle_flight   = module2_visual_odometry.circle_flight:main',
            'spawn_landmarks = module2_visual_odometry.spawn_landmarks:main',
        ],
    },
)

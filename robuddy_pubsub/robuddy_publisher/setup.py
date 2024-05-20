from setuptools import find_packages, setup

package_name = 'robuddy_publisher'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='P9',
    maintainer_email='mmathi19@student.aau.dk',
    description='A script for moving Robuddy',
    license='None',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'send_message = robuddy_publisher.send_message:main',
            'send_message_continuous = robuddy_publisher.send_message_continuous:main'
        ],
    },
)

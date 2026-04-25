from setuptools import setup

package_name = 'my_robot_pkg'   # ← change if your package name differs

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your@email.com',
    description='Motor command publisher node',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [    
            'command_sender = scripts.command_sender:main',
        ],
    },
)
from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # generic
    "psutil",
    "numpy>=1.26.0",
    "torch>=2.7.0",  # compatible with Isaac Sim 5.1 and Isaac Lab 2.3
    "torchvision>=0.22.0",  # compatible with torch 2.7.0
    # basic logger
    "tensorboard",
    # "pybullet",
    # "rosbags",
    # "pinocchio=3.1.0",
    "pytorch-kinematics"
]

# Installation operation
setup(
    name="agent_tasks",
    packages=find_packages(include=["agent_tasks", "agent_tasks.*"]),
    author="zzr",
    maintainer="zzr",
    url="",
    version="0.1.0",
    description="SCUT Robot Lab, Agent Tasks",
    keywords="Wheelbipe, Reinforcement Learning, Agent Tasks",
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    include_package_data=True,
    python_requires=">=3.10,<3.12",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)

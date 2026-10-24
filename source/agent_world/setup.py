"""Installation script for the 'agent_world' python package."""
from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
    "pyyaml>=6.0",
]

# Installation operation
setup(
    name="agent_world",
    packages=find_packages(include=["agent_world", "agent_world.*"]),
    package_data={"agent_world": ["assets/usd_files/**/*.usd"]},
    author="zzr",
    maintainer="zzr",
    url="",
    version="0.1.0",
    description="SCUT Robot Lab, Agent World",
    keywords="Wheelbipe, Reinforcement Learning, Agent World",
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

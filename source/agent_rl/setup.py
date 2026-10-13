"""Installation script for the 'agent_rl' python package."""
from setuptools import find_packages, setup

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
    "pybullet",
    # Isaac Lab selects the compatible rsl-rl release.  The local compatibility
    # module supplies store_code_state across the 2.x-5.x API range.
    "rsl-rl-lib>=2.0,<6.0",
]

# Installation operation
setup(
    name="agent_rl",
    packages=find_packages(include=["agent_rl", "agent_rl.*"]),
    author="zzr",
    maintainer="zzr",
    url="",
    version="0.1.0",
    description="SCUT Robot Lab, Agent RL",
    keywords="Wheelbipe, Reinforcement Learning, Agent RL",
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

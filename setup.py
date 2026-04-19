from setuptools import setup, find_packages

setup(
    name="assetto_rl_bot_claude",
    version="1.0.0",
    description="Autonomous driving RL agent for Assetto Corsa — Claude build",
    author="Claude (Anthropic)",
    python_requires=">=3.11",
    packages=find_packages(exclude=["scripts", "logs", "models"]),
    install_requires=[
        "stable-baselines3[extra]>=2.3.0",
        "gymnasium>=0.29.1",
        "numpy>=1.26.0",
        "torch>=2.2.0",
        "pyyaml>=6.0.1",
        "pyautogui>=0.9.54",
        "pydirectinput>=1.0.4",
        "vgamepad>=0.1.0",
        "tensorboard>=2.16.0",
        "pandas>=2.2.0",
        "tqdm>=4.66.0",
        "colorlog>=6.8.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

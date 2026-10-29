from setuptools import setup, find_packages

setup(
    name="ci-active-inference",
    version="0.1.0",
    description="Active Inference prediction system for CI Server database",
    author="Companion Intelligence",
    license="PolyForm-Noncommercial-1.0.0",
    classifiers=[
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
    ],
    packages=find_packages(),
    install_requires=[
        "psycopg2-binary>=2.9.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "python-dotenv>=0.19.0",
        "pydantic>=1.8.0",
        "sqlalchemy>=1.4.0",
        "asyncio-mqtt>=0.10.0",
        "fastapi>=0.70.0",
        "uvicorn>=0.15.0",
    ],
    python_requires=">=3.8",
)
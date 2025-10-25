"""
Aura Platform - Shared Library Setup
Setup configuration for the shared library package.
"""

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aura-shared-lib",
    version="0.1.0",
    author="Aura Platform Team",
    author_email="dev@aura-platform.com",
    description="Shared library for Aura Platform services",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aura-platform/shared-lib",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=[
        # Core dependencies
        "pydantic>=2.0.0",
        "sqlalchemy>=2.0.0",
        "asyncpg>=0.28.0",
        "alembic>=1.12.0",
        
        # Security
        "passlib[bcrypt]>=1.7.4",
        "python-jose[cryptography]>=3.3.0",
        "cryptography>=41.0.0",
        
        # Utilities
        "python-multipart>=0.0.6",
        "python-dotenv>=1.0.0",
        
        # Logging
        "structlog>=23.1.0",
        
        # Testing (optional)
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "test": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "pytest-mock>=3.11.0",
            "httpx>=0.24.0",
        ],
    },
    include_package_data=True,
    package_data={
        "aura_shared_lib": [
            "py.typed",
        ],
    },
    zip_safe=False,
)

import os

from setuptools import setup

import versioneer

packages = [
    "py2vtk",
    "py2vtk.api",
    "py2vtk.core",
    # 'py2vtk.parallel',
    # 'py2vtk.dask',
]

tests = ["tests"]

install_requires = [
    "numpy",
]
extras_require = {
    "tests": ["pytest", "vtk", "mpi4py"],
    "mpi": ["mpi4py"],
}

setup(
    name="py2vtk",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Low dependency module to export to VTK from python",
    url="https://github.com/anlavandier/py2vtk",
    author="Antoine Lavandier",
    author_email="antoine.lavandier.24.11@gmail.com",
    license="MIT",
    keywords="Visualization VTK",
    packages=packages + tests,
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    zip_safe=False,  # https://mypy.readthedocs.io/en/latest/installed_packages.html
)

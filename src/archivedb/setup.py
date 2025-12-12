from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()
version = {}
with open("archivedb/version.py", "r") as f:
    exec(f.read(), version)

setup(
    name='archivedb',
    version=version["__version__"],
    author="Sergey A. Bozhenkov",
    author_email='boz@ipp.mpg.de',
    description='Python client for W7-X REST ArchiveDB.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="git.ipp-hgw.mpg.de/boz/archivedb",
    packages=["archivedb",],
    license="LGPLv3",
    install_requires=[ 'numpy', "pillow", "numexpr", "tables",
                      "future", "cbor"],
    zip_safe=False,
    classifiers=[ "Programming Language :: Python",
                 ("License :: OSI Approved :: GNU Lesser General "
                  "Public License v3 or later (LGPLv3+)"),
                 "Operating System :: OS Independent",
                 "Intended Audience :: End Users/Desktop"
                 ],
    )

archivedb is a python library to work with the REST API of the 
W7-X experimental data storage. It provides simple functions to:
find program information, find available time intervals,
read/write signals and signal boxes, read/write parameter logs,
read/write images, resolve aliases. In addition, a caching layer
is provided that significantly accelerates repeated requests and
is useful for typical data analysis tasks.

# Installation
To install the library do the usual in the command line:

    pip install -r requirements.txt
    python setup.py install

The first step will install all necessary extra libraries.

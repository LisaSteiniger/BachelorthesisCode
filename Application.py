'''This file is responsible for the final calculation of sputtering yields and thickness of the erosion layer. 
It uses the functions defined in SputteringYieldFunctions after reading the data from the archieveDB'''

import w7xarchive
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants
import scipy.integrate as integrate

import BachelorthesisCode.src.SputteringYieldFunctions as calc
import BachelorthesisCode.src.ReadArchieveDB as read

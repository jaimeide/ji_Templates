"""
================================================================
     Copyright (c) 2018, Investment Technology Group (ITG)
                     All Rights Reserved
================================================================

 NAME : ji_p5_importAll.py
 @DATE     Created: 12/13/18 (10:40 AM)
	       Modifications:
	       - 12/13/18:

 @AUTHOR          : Jaime Shinsuke Ide
                    jaime.ide@itg.com or jaime.ide@yale.edu
===============================================================
"""

## General
import argparse as ag
import numpy as np
import pandas as pd
import seaborn as sns
import os
import random
import datetime
import matplotlib.pyplot as plt
import matplotlib
import time
import subprocess
import glob

from scipy.stats import pearsonr
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

## Kris
from kf_data import tickdata
from kf_data import securitymaster
from kf_analytics import gets_dob
from kf_analytics import intraday_volume
#import kf_analytics as kf
#import kf_data as kf
from pathlib import Path


import sys
sys.path.insert(0, '/home/jide/projects')
import ji_utils_fin2019 as jifin
import ji_utils_plots2019 as ji
import ji_utils_ml2019 as jiml
import ji_itg2019 as itg
import CostCurve

# Magic command to update external library in every cell in Jupyter!
#%matplotlib inline
#%load_ext autoreload
#%autoreload 2

class matlablike():
    pass

## PATH
# newfe-s18:/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data
dpath = '/local/data/itg/fe/prod/data1/jide-local/Projects/P5/data'
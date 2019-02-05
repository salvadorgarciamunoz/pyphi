#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 22:36:20 2019

@author: famgarciatorres
"""

import pandas as pd
import numpy as np
Cars_Features=pd.read_excel('Automobiles PLS.xls', 'Features', index_col=None, na_values=np.nan)
Cars_Features_data=np.array(Cars_Features.values[:,2:]).astype(float)

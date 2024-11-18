# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 13:32:27 2024

@author: salva
"""
import pyphi as phi
import pandas as pd

ex_data=pd.read_excel('data.xlsx')
factors=[
    'Variable A',
    'Var B',
    'VarC',
    'Variable A*VarC',
    'Var B^2',
    'Variable A^2*VarC'
    ]

betas,facts,X,Y=phi.build_polynomial(ex_data,factors,'Response 1')

 
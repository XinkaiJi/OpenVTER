#!/usr/bin/env python 
# -*- coding: utf-8 -*-
'''
@Time : 2022-10-07 16:04
@Author : Xinkai Ji
@contact: jhjxk@hotmail.com
@File : setup.py 
@Software: PyCharm
@desc: 
'''
from distutils.core import setup
from Cython.Build import cythonize
setup(
    ext_modules = cythonize("sort_r.py")
)
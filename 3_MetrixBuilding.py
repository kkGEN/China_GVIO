# -*- coding: utf-8 -*-
import os
import functools
import time
import numpy as np
import arcpy
import pandas as pd
import re
import fnmatch


def Time_Decorator(func):
    # 输出函数运行时间的修饰器
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'{func.__name__} Start_time: {start_time_str}.')
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} Excution_time: {end_time - start_time}.')
        return result
    return wrapper


if __name__ == "__main__":
    rootpath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/'
    excel_files_path = os.path.join(rootpath, '11-工厂缓冲区土地利用对应灯光强度excel/')
    excel_files = os.listdir(excel_files_path)

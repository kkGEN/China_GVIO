# -*- coding: utf-8 -*-
import os
import functools
import time
import numpy as np
import arcpy
import pandas as pd
import re
import fnmatch
import shutil


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


def Creat_New_Folder(root_fold):
    excel_files = os.listdir(root_fold)
    for fold in excel_files:
        print(fold)
        fold_path = os.path.join(root_fold, fold)
        for year in range(2014, 2023):
            for month in range(1, 13):
                if month < 10:
                    new_folder = os.path.join(fold_path, f'{year}0{month}')
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
                else:
                    new_folder = os.path.join(fold_path, f'{year}{month}')
                    if not os.path.exists(new_folder):
                        os.makedirs(new_folder)
        print('New Folder Created Successfully!')


def List_Element_Minus(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    set3 = set1 - set2
    folder_file = list(set3)
    return folder_file


def Move_Files_to_Folder(root_path):
    # 按照年月新建文件夹
    Creat_New_Folder(root_path)
    for folder in os.listdir(root_path):
        # 读取存储不同图层（灯光、地温、植被指数）文件的文件夹
        layer_excel_path = os.path.join(root_path, folder)
        # 将文件夹中的文件和文件夹分别存储，并将对应名称的文件放入文件夹
        file_folder = os.listdir(layer_excel_path)
        xlsx_file = fnmatch.filter(file_folder, '*xlsx')
        # 获取只包含文件夹路径的列表
        folder_file = List_Element_Minus(file_folder, xlsx_file)
        for fold_name in folder_file:
            for file_name in xlsx_file:
                # 如果文件名中有文件夹名称，则移动到对应文件夹中
                if fold_name in file_name:
                    source_path = os.path.join(layer_excel_path, file_name)
                    terget_path = os.path.join(layer_excel_path, fold_name)
                    shutil.move(source_path, terget_path)
        print(f'{folder} Done!')


def Reshape_Excel(root_path, path, out_path):
    for yearmonth in path:
        yearmonth_path = os.path.join(root_path, yearmonth)
        single_file_path = os.listdir(yearmonth_path)

        out_month_path = os.path.join(out_path, yearmonth)
        if not os.path.exists(out_month_path):
            os.makedirs(out_month_path)

        for table in single_file_path:
            table_path = os.path.join(yearmonth_path, table)
            out_table_path = os.path.join(out_month_path, table)

            df_table = pd.read_excel(table_path, usecols=fields)
            df_table_groupby = df_table.groupby('ORIG_FID')
            Columns = ['ID', '0', '101', '201', '202', '301', '401', '402', '403', '501', '502', '503', '504', '505']

            df_new = pd.DataFrame(columns=Columns)
            for key, value in df_table_groupby:
                df_new.loc[key, 'ID'] = key

                for index, row in value.iterrows():
                    level2_code = str(int(row['Level2']))
                    if level2_code in Columns:
                        df_new.loc[key, level2_code] = row['grid_code']
            df_new.to_excel(out_table_path)
            print(df_new)



def Make_Out_Dirs(folder_name, root_path):
    out_folder_name = folder_name[-10:-6]
    out_folder_path = os.path.join(root_path, out_folder_name)
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    return out_folder_path


if __name__ == "__main__":
    rootpath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/'
    fields = ['ORIG_FID', 'Level2', 'grid_code', 'Shape_Area']
    excel_files_path = os.path.join(rootpath, '11-工厂缓冲区土地利用对应植被指数excel/')
    out_root_path = os.path.join(rootpath, r'12-工厂缓冲区内各类特征提取/')

    # 1.按照年月重新分类存储excel
    # Move_Files_to_Folder(excel_files_path)

    # 2.根据不同土地利用类型
    file_path = os.listdir(excel_files_path)
    out_folder = Make_Out_Dirs(excel_files_path, out_root_path)
    Reshape_Excel(excel_files_path, file_path, out_folder)
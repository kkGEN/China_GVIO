# -*- coding: utf-8 -*-
import os
import functools
import time
import pandas as pd
import fnmatch
import shutil
import numpy as np
import arcpy
import re


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


def Create_New_Dir(out_path, out_name):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f'{out_name} created successfully!')
    else:
        print(f'{out_name} is already exists!')


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


@Time_Decorator
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


@Time_Decorator
def Reshape_Excel(root_path, out_root_path, path, read_fields):
    # 按三类数据名称创建输出文件路径
    out_path = Make_Out_Dirs(root_path, out_root_path)
    # 逐月遍历原始数据文件夹（以月份编号）
    for yearmonth in path:
        yearmonth_path = os.path.join(root_path, yearmonth)
        single_file_path = os.listdir(yearmonth_path)
        # 创建月份文件夹，存储输出数据
        out_month_path = os.path.join(out_path, yearmonth)
        if not os.path.exists(out_month_path):
            os.makedirs(out_month_path)

        # 遍历每个原始excel文件
        for table in single_file_path:
            table_path = os.path.join(yearmonth_path, table)
            out_table_path = os.path.join(out_month_path, table)
            # 按照原始ID进行分组
            df_table = pd.read_excel(table_path, usecols=read_fields)
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


@Time_Decorator
def Execute_Reshape_Excel(root_path):
    excel_folders = os.listdir(root_path)
    for folder in excel_folders:
        excel_yearmonth = os.path.join(root_path, folder)
        # 地表温度,夜间灯光,植被指数
        file_path = os.listdir(excel_yearmonth)
        Reshape_Excel(excel_yearmonth, out_root_path, file_path, fields)


def Make_Out_Dirs(folder_name, root_path):
    out_folder_name = folder_name[-10:-6]
    out_folder_path = os.path.join(root_path, out_folder_name)
    if not os.path.exists(out_folder_path):
        os.makedirs(out_folder_path)
    return out_folder_path

@Time_Decorator
def RS_Data_to_Metrix(data_path, out_root_path):
    year_list = os.listdir(data_path)
    # 创建输出年份文件夹
    for year in year_list:
        # 创建输出年文件夹
        out_year_path = os.path.join(out_root_path, year)
        Create_New_Dir(out_year_path, year)
        # 读取月度文件夹
        year_path = os.path.join(rs_data_path, year)
        month_list = os.listdir(year_path)
        for month in month_list:
            # 创建输出月文件夹
            out_month_path = os.path.join(out_year_path, month)
            Create_New_Dir(out_month_path, month)
            # 读取各省GDB数据库
            month_path = os.path.join(year_path, month)
            gdb_list = os.listdir(month_path)
            for gdb in gdb_list:
                prov_path = os.path.join(out_month_path, gdb[:-4])
                Create_New_Dir(prov_path, gdb[:-4])
                # 将省GDB设置为工作空间
                gdb_path = os.path.join(month_path, gdb)
                arcpy.env.workspace = gdb_path
                # 获取各省地级市列表
                dijishi_list = arcpy.ListRasters()
                for dijishi in dijishi_list:
                    dijishi_arr_name = os.path.join(prov_path, f'{dijishi}.npy')
                    np_arr = arcpy.RasterToNumPyArray(os.path.join(gdb_path, dijishi))
                    np.save(dijishi_arr_name, np_arr)
                print(gdb)
            print(month)
        print(year)


if __name__ == "__main__":
    rootpath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/'
    fields = ['ORIG_FID', 'Level2', 'grid_code', 'Shape_Area']
    excel_files_path = os.path.join(rootpath, '11-工厂缓冲区内灯光植被地温excel/')
    out_root_path = os.path.join(rootpath, r'12-工厂缓冲区内各类特征提取/')

    # 1.按照年月重新分类存储excel
    # Move_Files_to_Folder(excel_files_path)

    # 2.根据不同土地利用类型将各类数据excel结果进行变形
    # Execute_Reshape_Excel(excel_files_path)

    # 3.（第二种数据预处理路径）将POI和EULUC矢量转换为与遥感影像数据相同分辨率的栅格

    # 4. 提取POI和EULUC图层对应的各种影像的地级市层次的数据矩阵

    # 1.将经过掩膜的每一类遥感数据输出为矩阵
    Root_Path = r'E:/ChinaMonthlyIndustrial/11-各类遥感数据地级市分割/'
    Out_RS_Metrix_Path = r'E:/ChinaMonthlyIndustrial/12-遥感数据输出为矩阵/'
    RS_Data_Path_List = ['地表温度'] #'地表温度', '夜间灯光', '增强植被指数'
    buff_dist = 500
    for rs_data_name in RS_Data_Path_List:
        rs_data_path = os.path.join(Root_Path, f'11.2-{rs_data_name}_Extract_{buff_dist}m/')
        out_rs_metrix_path = os.path.join(Out_RS_Metrix_Path, f'12.1-{rs_data_name}-{buff_dist}m/')
        RS_Data_to_Metrix(rs_data_path, out_rs_metrix_path)

    # nn
    #fff




















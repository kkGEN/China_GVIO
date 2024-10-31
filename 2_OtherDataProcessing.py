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


def Write_Landuse_Path(path, flexible_buffer, outname):
    Landuse_folder_Path = os.path.join(path, flexible_buffer)
    gdb_path = os.listdir(Landuse_folder_Path)
    file_list_df = pd.DataFrame(columns=['FilePath', 'OutTable'])
    for gdb in gdb_path:
        prov_gdb_path = os.path.join(Landuse_folder_Path, gdb)
        # 设置每个省的gdb为工作空间，逐个地级市进行处理
        arcpy.env.workspace = prov_gdb_path
        landuse_path = arcpy.ListFeatureClasses()
        for landuse in landuse_path:
            dijishi_landuse = os.path.join(prov_gdb_path, landuse)
            tablename = dijishi_landuse.split('.gdb\\')[1].split('_FactoryLU')[0]
            df_row = pd.DataFrame({'FilePath': [dijishi_landuse], 'OutTable': [tablename]})
            file_list_df = pd.concat([file_list_df, df_row], axis=0)
    file_list_df.to_csv(outname, encoding='utf-8-sig', index=False)
    print('Files path saved done!')


def Raster_to_Points(raster_path, workplace):
    # 读取每个月的全国夜间灯光合成数据
    NTL_path = os.listdir(raster_path)
    for month in NTL_path:
        NTL_month_folder_path = os.path.join(raster_path, month)
        ntl_month_path = os.listdir(NTL_month_folder_path)
        ntl_month_path = fnmatch.filter(ntl_month_path, '*.tif')
        for ntl in ntl_month_path:
            # 设置输出栅格转点的路径
            arcpy.env.workspace = workplace
            arcpy.env.overwriteOutput = True
            ras_to_feature = f'{ntl[0:16]}'
            # 执行栅格转点
            ntl_month = os.path.join(NTL_month_folder_path, ntl)
            arcpy.RasterToPoint_conversion(ntl_month, ras_to_feature, "VALUE")
            print(ras_to_feature)
    print('All Rasters have turned into Feature Points!')

@Time_Decorator
def Spatial_Join_to_Excel(fc_path, workspace, spatialjoin_path, excel_path):
    arcpy.env.workspace = workspace
    ras2point_path = arcpy.ListFeatureClasses()
    for ras2point in ras2point_path:
        print(ras2point)

        for index, row in fc_path.iterrows():
            arcpy.env.workspace = spatialjoin_path
            arcpy.env.overwriteOutput = True
            # 每个地级市工厂缓冲区土地利用路径
            featureclass_path = row[0]
            # 地级市名称
            out_name = row[1]

            # spatialjoin后的结果feature
            out_join_feature = os.path.join(spatialjoin_path, f'{ras2point}_{out_name}')
            # 输出表格路径
            out_excel = os.path.join(excel_path, f'{ras2point}_{out_name}.xlsx')

            arcpy.SpatialJoin_analysis(featureclass_path, os.path.join(workspace, ras2point), out_join_feature)
            arcpy.TableToExcel_conversion(out_join_feature, out_excel)
            print(f'{out_name} done!')




if __name__ == "__main__":
    rootpath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/'

    # 将每月的夜间灯光数据转化为点，方便后续计算，提高效率
    # NTL_rootpath = r'E:/夜间灯光月合成-中国2014-2022/'
    # EVI_rootpath = r'E:/增强植被指数月合成-中国2014-2022/'
    # LST_rootpath = r'E:/地表温度月合成-中国2014-2022/'

    # out_ras2fc_root = r'G:/中国夜间灯光月合成栅格转点.gdb/'
    # out_ras2fc_root_evi = r'E:/中国植被指数月合成栅格转点.gdb/'
    # out_ras2fc_root_lst = r'E:/中国地表温度月合成栅格转点.gdb/'

    # Raster_to_Points(NTL_rootpath, out_ras2fc_root)

    # 将所有工厂缓冲区土地利用矢量路径存储为csv
    flexible_buffer_fold = r'10-中国EULUC数据_1000m/10.2-全国各地级市EULUC_工厂缓冲区内土地利用/'
    out_filelist = os.path.join(rootpath, '地级市工厂缓冲土地利用路径列表.csv')
    # Write_Landuse_Path(rootpath, flexible_buffer_fold, out_filelist)

    # 读取整饰完成后的featureclass路径，并执行空间连接输出为excel
    ras2fc_gdb = r'E:/中国植被指数月合成栅格转点/evi_ras2point_2014-2016.gdb/'
    out_excel_root = os.path.join(rootpath, f'11-工厂缓冲区土地利用对应植被指数excel/')
    out_join_feature_root = os.path.join(rootpath, f'11-工厂缓冲区土地利用对应植被指数join/工厂缓冲区土地利用与植被指数矢量点连接2014-2016.gdb/')
    file_list_df = pd.read_csv(out_filelist)
    # Spatial_Join_to_Excel(file_list_df, ras2fc_gdb, out_join_feature_root, out_excel_root)

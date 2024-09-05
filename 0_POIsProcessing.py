# -*- coding: utf-8 -*-
import arcpy
import os
import functools
import time
import pandas as pd


def Time_Decorator(func):
    # 输出函数运行时间的修饰器
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_time_str = time.strftime('%Y-%m-%d %H:%M:%S')
        print(f'{func.__name__} Start_time: {start_time_str}.')
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f'{func.__name__} Excution_time: {end_time-start_time}.')
        return result
    return wrapper


def Creat_New_GDB(rootgdb, gdbname, gdbfullname):
    if not arcpy.Exists(gdbfullname):
        arcpy.CreateFileGDB_management(rootgdb, gdbname)
        print(f'{gdbname} created successfully!')
    else:
        print(f'{gdbname} is already exists!')
    return


@Time_Decorator
def Split_POI_by_Dijishi(rawpoi):
    poi_prov_pathlist = os.listdir(rawpoi)
    for poi_prov_path in poi_prov_pathlist:
        # 根据文件夹名提取每个省的名称
        fold_prov_name = poi_prov_path.replace('POI数据', '')

        # 读取每个省份文件夹中的csv文件
        poi_prov_fold = os.path.join(rawpoi, poi_prov_path)
        poi_list = os.listdir(poi_prov_fold)
        for poi in poi_list:
            poi_file = os.path.join(poi_prov_fold, poi)
            print(poi_file)
            poi_df = pd.read_csv(poi_file)
            # 按照地级市进行分组,并输出为csv格式
            poi_df_grouped = poi_df.groupby('城市')
            for key, value in poi_df_grouped:
                poi_dijishi_name = os.path.join(poi_prov_fold, f'{fold_prov_name}_{key}POI数据_2022年3月.csv')
                value.to_csv(poi_dijishi_name, index=False)
                print(f'{fold_prov_name}_{key} done!')
    return


@Time_Decorator
def CSV_to_Shapfile(rawpoi, gdbpath):
    poi_prov_pathlist = os.listdir(rawpoi)
    for poi_prov_path in poi_prov_pathlist:
        # 根据文件夹名提取每个省的名称,并命名新的GDB
        fold_prov_name = poi_prov_path.replace('POI数据', '')
        gdb_prov_name = f'{fold_prov_name}.gdb'
        poi_prov_gdb = os.path.join(gdbpath, gdb_prov_name)
        Creat_New_GDB(gdbpath, gdb_prov_name, poi_prov_gdb)
        # 将当前工作空间设置为新建或已存在的gdb数据库
        arcpy.env.workspace = poi_prov_gdb
        arcpy.env.overwriteOutput = True

        # 读取每个省份文件夹中的csv文件
        poi_prov_fold = os.path.join(rawpoi, poi_prov_path)
        poi_list = os.listdir(poi_prov_fold)
        for poi in poi_list:
            poi_file = os.path.join(poi_prov_fold, poi)
            out_poi_feature = poi.replace('POI数据_2022年3月.csv', '')
            arcpy.management.XYTableToPoint(poi_file, out_poi_feature, '经度', '纬度')
            print(f'{out_poi_feature} done!')
    return


@Time_Decorator
def POI_Export_by_Category(gdb_fold, catogry, type, outgdb_fold):
    where_clause = f"{catogry} = '{type}'"
    gdb_list = os.listdir(gdb_fold)
    for gdb in gdb_list:
        gdb_path = os.path.join(gdb_fold, gdb)
        outgdb_path = os.path.join(outgdb_fold, gdb)
        Creat_New_GDB(outgdb_fold, gdb, outgdb_path)

        arcpy.env.workspace = gdb_path
        arcpy.env.overwriteOutput = True
        all_poi = arcpy.ListFeatureClasses()

        for poi in all_poi:
            outpoi_path = os.path.join(outgdb_path, f'{poi}POI_{type}')
            arcpy.ExportFeatures_conversion(poi, outpoi_path, where_clause)
            print(f'{poi} done!')
    return


if __name__ == "__main__":
    RootPath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/9-中国POI数据/'
    RawPOIPath = os.path.join(RootPath, '9.0-全国地级市POI原始数据/')
    GDBPath = os.path.join(RootPath, '9.1-全国各省地级市POI/')

    # 1.将每个省一个csv文件的poi按照地级市分割
    # Split_POI_by_Dijishi(RawPOIPath)

    # 2.新建每个省份/直辖市的GDB,并存放转换为shp的poi
    CSV_to_Shapfile(RawPOIPath, GDBPath)

    # 3.按照大类导出
    poi_category_I = '大类'
    poi_category_type_I = '公司企业'
    out_fold_path_I = os.path.join(RootPath, '9.2-全国地级市POI_大类_公司企业')
    # POI_Export_by_Category(GDBPath, poi_category_I, poi_category_type_I, out_fold_path_I)

    # 4.按照中类导出
    poi_category_II = '中类'
    poi_category_type_II = '工厂'
    out_fold_path_II = os.path.join(RootPath, '9.3-全国地级市POI_中类_工厂')
    # POI_Export_by_Category(GDBPath, poi_category_II, poi_category_type_II, out_fold_path_II)

    # 5.为工厂建立缓冲区
    out = os.path.join(RootPath, '9.4-全国地级市POI_中类_工厂_缓冲区')












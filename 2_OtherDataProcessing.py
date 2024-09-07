# -*- coding: UTF-8 -*-
import torch
import pandas as pd
import geopandas as gpd
import random
import fnmatch
import functools
import time
from osgeo import gdal, ogr, osr
import os
ogr.UseExceptions()
gdal.UseExceptions()
gdal.SetConfigOption('SHAPE_ENCODING', 'UTF-8')

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


# 通过文件路径读取GDB中的layer,并返回数据集和图层数量
def Open_GDB_by_GDAL(gdbPath):
    # driver = gdal.GetDriverByName('OpenFileGDB')
    gdb_dataset = gdal.OpenEx(gdbPath, gdal.OF_VECTOR)
    if gdb_dataset is None:
        print('无法打开GDB数据源！')
        exit(1)
    # 获取图层数量
    layer_count = gdb_dataset.GetLayerCount()
    return gdb_dataset, layer_count


def Open_Shp_by_GDAL(shp_path):
    dataset = ogr.Open(shp_path, 0)
    if dataset is None:
        print('无法打开ShapeFile数据源！')
        exit(1)
    layer_count = dataset.GetLayerCount()
    return dataset, layer_count


def Clip_Layer(in_ds, method_ds, out_path):
    in_lyr = in_ds.GetLayer()

    # method_ds = ogr.Open(os.path.join(out_buff_path, '北京_POI_中类工厂.shp'))
    method_lyr = method_ds.GetLayer()

    fname = os.path.join(out_path, 'Clipped.shp')
    # 创建被裁剪以后的输出文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(fname):
        driver.DeleteDataSource(fname)

    # 新建DataSource，Layer
    out_ds = driver.CreateDataSource(fname)
    out_lyr = out_ds.CreateLayer(fname, in_lyr.GetSpatialRef(), in_lyr.GetGeomType())

    # 开始进行裁剪
    in_lyr.Clip(method_lyr, out_lyr)
    out_ds.FlushCache()
    del in_ds, method_ds, out_ds
    print(fname)

    return

if __name__ == "__main__":
    PreRootPath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/9-中国POI数据/'
    SinglePOI_Fold = os.path.join(PreRootPath, '9.5-全国地级市POI_中类_工厂_缓冲区_逐个输出/')
    RootPath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/10-中国EULUC数据/'
    Landuse_Fold = os.path.join(RootPath, 'EULUC-2018/')

    # 读取土地利用数据
    landuse_path = os.path.join(Landuse_Fold, 'euluc-latlonnw.shp')
    landuse_dataset, landuse_layer_count = Open_Shp_by_GDAL(landuse_path)
    landuse_layer = landuse_dataset.GetLayer()

    singlepoi_path = os.listdir(SinglePOI_Fold)
    for singlepoi in singlepoi_path:
        singlepoi_dijishi_path = os.path.join(SinglePOI_Fold, singlepoi)
        dijishi_gdb_list = os.listdir(singlepoi_dijishi_path)

        for gdb in dijishi_gdb_list:
            dijishi_gdb = os.path.join(singlepoi_dijishi_path, gdb)
            dijishi_dataset, dijishi_layer_count = Open_GDB_by_GDAL(dijishi_gdb)







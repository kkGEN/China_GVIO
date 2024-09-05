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
    # layer = gdb_dataset.GetLayer(0)
    # print(layer_count)

    # feature_count = layer.GetFeatureCount()
    # feature = layer.GetNextFeature()
    #
    # while feature:
    #     feature_attris = feature.GetField(0)
    #     print(feature_attris)
    #
    #     feature = layer.GetNextFeature()
    #
    # gdb_dataset = None
    return gdb_dataset, layer_count


# 通过文件路径读取shapefile,
def Open_Shp_by_GDAL(shp_path):
    dataset = ogr.Open(shp_path, 0)
    if dataset is None:
        print('无法打开ShapeFile数据源！')
        exit(1)
    layer_count = dataset.GetLayerCount()
    return dataset, layer_count


def Creat_Buff_of_Points(Layer, distance, Outname):
    # 创建用于存储结果的新数据源
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_dataset = output_driver.CreateDataSource(Outname)
    output_layer = output_dataset.CreateLayer('buffers', srs=Layer.GetSpatialRef())

    # 设置新图层的属性字段与原始图层一致
    output_layer_defn = Layer.GetLayerDefn()
    for i in range(output_layer_defn.GetFieldCount()):
        fieldDefn = output_layer_defn.GetFieldDefn(i)
        output_layer.CreateField(fieldDefn)

    # 逐个feature建立缓冲区
    feature = Layer.GetNextFeature()
    while feature:
        # 获取原始POI图层的几何参考，使用Buffer函数构建缓冲区
        geometry = feature.GetGeometryRef()
        buffer_geometry = geometry.Buffer(distance)

        # 创建新的要素，并拷贝原始属性和设置其缓冲区几何属性
        new_feature = ogr.Feature(Layer.GetLayerDefn())
        new_feature.SetGeometry(buffer_geometry)

        new_featureDefn = Layer.GetLayerDefn()
        # 添加属性及要素到新图层
        for i in range(new_featureDefn.GetFieldCount()):
            new_feature.SetField(i, feature.GetField(i))  # 输出字段乱码问题有待解决
        output_layer.CreateFeature(new_feature)

        # 销毁特征对象，以释放资源
        new_feature.Destroy()
        feature = Layer.GetNextFeature()
    return


@Time_Decorator
def Poi_Buff_of_China(poi_path, out_path, buff_distance):
    # 读取gdb中的图层，返回图层集合和数量
    POIs_Dataset, POIs_LayerCount = Open_GDB_by_GDAL(poi_path)
    # 按照图层数量，逐个读取图层
    for layercount in range(POIs_LayerCount):
        # 遍历读取每一个poi图层，并获取图层的名称
        poi_layer = POIs_Dataset.GetLayer(layercount)
        print(poi_layer.GetName())
        out_buff_name = os.path.join(out_path, f'{poi_layer.GetName()}.shp')
        # 调用缓冲区函数，对每个图层的poi建立缓冲区
        Creat_Buff_of_Points(poi_layer, buff_distance, out_buff_name)
    return


def Feature_to_Layer(feature, old_layer, out_path, new_layer_name):
    # feature：原始图层的特征
    # old_layer：原始图层
    # out_path：输出路径
    # new_layer_name:新图层的名字

    # 创建一个新的图层
    output_driver = ogr.GetDriverByName('ESRI Shapefile')
    output_dataset = output_driver.CreateDataSource(out_path)
    check_file_exist = random.randint(1, 99)
    out_new_path = os.path.join(out_path, f'{new_layer_name}.shp')
    if os.path.exists(out_new_path):
        print(f'{new_layer_name} is already exists!')
        new_layer_name = f'{new_layer_name}_{check_file_exist}'

    # 注意：一些工厂会出现不明确指代,如多个面粉厂,在上一步添加随机数后仍可能重复,这里选择跳过
    try:
        new_layer = output_dataset.CreateLayer(new_layer_name, srs=old_layer.GetSpatialRef())
        # 设置新图层的属性字段与原始图层一致
        new_layer_defn = old_layer.GetLayerDefn()
        for i in range(new_layer_defn.GetFieldCount()):
            fieldDefn = new_layer_defn.GetFieldDefn(i)
            new_layer.CreateField(fieldDefn)

        # 从原始图层中获取字段名称、几何信息等，传入新的feature
        new_featureDefn = old_layer.GetLayerDefn()
        new_feature = ogr.Feature(new_featureDefn)
        new_feature.SetGeometry(feature.GetGeometryRef())

        # 逐个字段进行赋值
        for i in range(new_featureDefn.GetFieldCount()):
            new_feature.SetField(i, feature.GetField(i))  # 输出字段乱码问题有待解决
        new_layer.CreateFeature(new_feature)
        new_feature.Destroy()
    except:
        print(f'{new_layer_name} is not handled well!')
    finally:
        pass


def Clip_Vector(input_layer, output_layer, clip_layer):
    result = gdal.Warp(output_layer, input_layer, format='ESRI Shapefile',
                       cutlineDSName=clip_layer, cropToCutline=True)

    # in_lyr.Clip(method_lyr, out_lyr)
    result.FlushCache()


@Time_Decorator
def POI_Buff_Clip_Landuse(landuse_path):
    # landuse_path:原始的全国土地利用矢量图层

    # 读取土地利用图层数据
    landuse_list = fnmatch.filter(os.listdir(landuse_path), '*shp')
    for landuse in landuse_list:
        landuse_dataset, landuse_layercount = Open_Shp_by_GDAL(os.path.join(landuse_path, landuse))
        landuse_layer = landuse_dataset.GetLayer()
        print(landuse_layer.GetName())
    # 用逐个缓冲区对土地利用数据进行clip

    return


@Time_Decorator
def POI_Buff_Feature_to_Layer(poi_path, out_path):
    # poi_path:带有缓冲区的POI图层
    # out_path:存储每个缓冲区的图层文件夹路径

    # 读取POI图层数据
    poi_buff_list = fnmatch.filter(os.listdir(poi_path), '*shp')
    for poi_buff in poi_buff_list:
        poi_buff_dataset, poi_buff_layercount = Open_Shp_by_GDAL(os.path.join(poi_path, poi_buff))
        poi_buff_layer = poi_buff_dataset.GetLayer()

        # 获取每个地级市POI矢量文件的名称，并根据此名称建立输出文件夹路径
        poi_buff_layer_name = poi_buff_layer.GetName()
        print(poi_buff_layer_name)
        # 按照图层名称(每个地级市的POI名称)新建文件夹
        layer_fold_path = os.path.join(out_path, poi_buff_layer_name)
        # 如果输出的文件夹路径不存在，则新建
        if not os.path.exists(layer_fold_path):
            os.makedirs(layer_fold_path)

        # 将每一个图层中的feature转换为layer矢量
        poi_buff_feature = poi_buff_layer.GetNextFeature()
        while poi_buff_feature:
            # 获取每一个feature的第一个字段,即名称
            poi_buff_feature_attris = poi_buff_feature.GetField(0)
            # 根据每个feature命名其对应的新图层
            Feature_to_Layer(poi_buff_feature, poi_buff_layer, layer_fold_path, poi_buff_feature_attris)
            # print(f'{poi_buff_feature_attris} is Done!')
            poi_buff_feature = poi_buff_layer.GetNextFeature()
    return


if __name__ == "__main__":
    DataRootPath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/'
    POIs_Fold = os.path.join(DataRootPath, '9-中国POI数据/')
    Landuse_Fold = os.path.join(DataRootPath, '10-中国EULUC数据/')

    # 对POI数据进行缓冲区分析
    buffer_distance = 0.01
    out_buff_path = os.path.join(POIs_Fold, '全国地级市POI_中类工厂_缓冲区')
    pois_path = os.path.join(POIs_Fold, '全国地级市POI_中类工厂.gdb')
    # Poi_Buff_of_China(pois_path, out_buff_path, buffer_distance)


    # 根据每个POI点对应的缓冲区对土地利用数据进行分割
    out_single_buff_path = os.path.join(POIs_Fold, '全国地级市POI_中类工厂_缓冲区_逐个输出/')
    # 首先,将POI缓冲区逐个生成一个圆形图层
    POI_Buff_Feature_to_Layer(out_buff_path, out_single_buff_path)
    # 然后,逐个圆形图层clip土地利用图层,得到每个工厂周围的土地利用类型
    landuse_path = os.path.join(Landuse_Fold, 'EULUC-2018/')
    POI_Buff_Clip_Landuse(landuse_path)




    # # out_landuse_path = os.path.join(landuse_path, '全国地级市EULUC_由POI裁剪得到/')
    # in_ds = ogr.Open(os.path.join(landuse_path, 'euluc-latlonnw.shp'))
    # in_lyr = in_ds.GetLayer()
    #
    # method_ds = ogr.Open(os.path.join(out_buff_path, '北京_POI_中类工厂.shp'))
    # method_lyr = method_ds.GetLayer()
    #
    # fname = os.path.join(out_landuse_path, 'Clipped.shp')
    # # 创建被裁剪以后的输出文件
    # driver = ogr.GetDriverByName('ESRI Shapefile')
    # if os.path.exists(fname):
    #     driver.DeleteDataSource(fname)
    #
    # # 新建DataSource，Layer
    # out_ds = driver.CreateDataSource(fname)
    # out_lyr = out_ds.CreateLayer(fname, in_lyr.GetSpatialRef(), in_lyr.GetGeomType())
    #
    # # 开始进行裁剪
    # in_lyr.Clip(method_lyr, out_lyr)
    # out_ds.FlushCache()
    # del in_ds, method_ds, out_ds
    # print(fname)

































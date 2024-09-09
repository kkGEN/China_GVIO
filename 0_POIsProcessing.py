# -*- coding: utf-8 -*-
import arcpy
import arcpy.da
import os
import functools
import time
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


def Creat_New_GDB(rootpath, gdbname, gdbfullname):
    if not arcpy.Exists(gdbfullname):
        arcpy.CreateFileGDB_management(rootpath, gdbname)
        print(f'{gdbname} created successfully!')
    else:
        print(f'{gdbname} is already exists!')


def Create_New_Dir(out_path, out_name):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print(f'{out_name} created successfully!')
    else:
        print(f'{out_name} is already exists!')


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


@Time_Decorator
def POI_Export_by_Category(gdb_fold, catogry, type, outgdb_fold):
    where_clause = f"{catogry} = '{type}'"
    gdb_list = os.listdir(gdb_fold)
    for gdb in gdb_list:
        # 设置当前读取gdb为工作空间
        gdb_path = os.path.join(gdb_fold, gdb)
        arcpy.env.workspace = gdb_path
        arcpy.env.overwriteOutput = True
        all_poi = arcpy.ListFeatureClasses()

        # 设置输出路径
        outgdb_path = os.path.join(outgdb_fold, gdb)
        Creat_New_GDB(outgdb_fold, gdb, outgdb_path)

        # 逐个读取图层,判断后输出
        for poi in all_poi:
            out_poi_path = os.path.join(outgdb_path, f'{poi}POI_{type}')
            arcpy.ExportFeatures_conversion(poi, out_poi_path, where_clause)
            print(f'{poi} done!')


@Time_Decorator
def Create_Buffer_of_POI(buff_distance, in_path, out_path):
    gdb_list = os.listdir(in_path)
    for gdb in gdb_list:
        # 设置当前读取gdb为工作空间
        gdb_path = os.path.join(in_path, gdb)
        arcpy.env.workspace = gdb_path
        arcpy.env.overwriteOutput = True
        all_poi = arcpy.ListFeatureClasses()

        # 设置输出路径
        outgdb_path = os.path.join(out_path, gdb)
        Creat_New_GDB(out_path, gdb, outgdb_path)

        for poi in all_poi:
            out_poi_buff_path = os.path.join(outgdb_path, f'{poi}_buff')
            arcpy.Buffer_analysis(poi, out_poi_buff_path, f'{buff_distance} Meters')
            print(f'{poi} done!')

@Time_Decorator
def FeatureClass_to_SingleFeatureClass(input_feature_class, output_folder):
    # 遍历每个要素并输出 ORIG_FID
    arcpy.SplitByAttributes_analysis(input_feature_class, output_folder, 'ORIG_FID')
    print(f'{input_feature_class} done!')
    return


@Time_Decorator
def POI_Single_Buff_to_Shp(in_path, out_path):
    gdb_list = os.listdir(in_path)
    for gdb in gdb_list:
        gdb_path = os.path.join(in_path, gdb)
        arcpy.env.workspace = gdb_path
        arcpy.env.overwriteOutput = True
        all_poi = arcpy.ListFeatureClasses()

        # 设置输出路径
        out_single_fold_name = gdb.replace('.gdb', '')
        outgdb_dir_path = os.path.join(out_path, out_single_fold_name)
        Create_New_Dir(outgdb_dir_path, out_single_fold_name)

        # 以每个地级市为文件夹,存放每个地级市的工厂缓冲区
        for poi in all_poi:
            dijish_gdb_name = poi.replace('POI_工厂_buff', '.gdb')
            dijishi_gdb_path = os.path.join(outgdb_dir_path, dijish_gdb_name)
            Creat_New_GDB(outgdb_dir_path, dijish_gdb_name, dijishi_gdb_path)
            print(dijishi_gdb_path)

            # 遍历每一个地级市缓冲区图层,并输出每一座工厂的缓冲区
            in_featureclass = os.path.join(gdb_path, poi)
            FeatureClass_to_SingleFeatureClass(in_featureclass, dijishi_gdb_path)
    return


@Time_Decorator
def Landuse_Clip_by_Province(in_shp_path, clip_shp_path, out_shp_path):
    # 全国的土地利用,切分为以省为单位
    prov_gdb_list = os.listdir(clip_shp_path)
    for gdb in prov_gdb_list:
        prov_gdb = os.path.join(clip_shp_path, gdb)
        # print(prov_gdb)
        arcpy.env.workspace = prov_gdb
        prov_shp_list = arcpy.ListFeatureClasses()
        for prov_shp in prov_shp_list:
            # print(prov_shp)
            out_name = re.sub(r"_.*", "", str(prov_shp))
            out_name = f'{out_name}_Euluc.shp'
            out_path = os.path.join(out_shp_path, out_name)
            arcpy.Clip_analysis(in_shp_path, prov_shp, out_path)
            print(f'{out_name} done!')


# [废弃]
def Landuse_Indentify_Single_Buff(landuse_fold, landuse_iden_fold, poi_path):
    prov_landuse_fold = os.listdir(landuse_fold)
    prov_landuse_fold = fnmatch.filter(prov_landuse_fold, '*.shp')
    for prov_landuse in prov_landuse_fold:
        prov_landuse_path = os.path.join(landuse_fold, prov_landuse)
        prov_name = re.sub(r"_.*", ".gdb", prov_landuse)
        print(prov_landuse_path)

        # 以地级市为单位创建新的gdb数据库,存储输出结果
        out_single_buff_iden_gdb = os.path.join(landuse_iden_fold, prov_name)
        Creat_New_GDB(landuse_iden_fold, prov_name, out_single_buff_iden_gdb)

        # 遍历存储各省POI缓冲区的gdb,并与土地利用识别分析
        single_buff_gdb_path = os.path.join(poi_path, prov_name)
        arcpy.env.workspace = single_buff_gdb_path
        dijishi_single_buff = arcpy.ListFeatureClasses()
        for single in dijishi_single_buff:
            out_single_buff_iden_name = os.path.join(out_single_buff_iden_gdb, single)
            in_sijishi_single_buff = os.path.join(single_buff_gdb_path, single)
            arcpy.analysis.Identity(in_sijishi_single_buff, prov_landuse_path, f'{out_single_buff_iden_name}_iden')
            print(f'{out_single_buff_iden_name} done!')


# [废弃]
def Landuse_Dissolve(lu_iden_fold, lu_diss_fold):
    landuse_iden_gdb = os.listdir(lu_iden_fold)
    for iden_gdb in landuse_iden_gdb:
        iden_gdb_path = os.path.join(lu_iden_fold, iden_gdb)

        # 设置输出路径
        out_iden_gdb_path = os.path.join(lu_diss_fold, iden_gdb)
        Creat_New_GDB(lu_diss_fold, iden_gdb, out_iden_gdb_path)

        # 对相同id的各类土地利用进行聚合
        arcpy.env.workspace = iden_gdb_path
        iden_path_list = arcpy.ListFeatureClasses()
        for iden in iden_path_list:
            out_iden = os.path.join(out_iden_gdb_path, re.sub(r'POI_工厂_buff_iden', '_FactoryLU', iden))
            arcpy.management.Dissolve(iden, out_iden, ["ORIG_FID", "Level2"])  #, "SINGLE_PART", "DISSOLVE_LINES"
            print(f'{iden} done!')


@Time_Decorator
def Landuse_in_Buff(landuse_fold, landuse_iden_fold, poi_path):
    prov_landuse_fold = os.listdir(landuse_fold)
    prov_landuse_fold = fnmatch.filter(prov_landuse_fold, '*.shp')
    for prov_landuse in prov_landuse_fold:
        prov_landuse_path = os.path.join(landuse_fold, prov_landuse)
        prov_name = re.sub(r"_.*", ".gdb", prov_landuse)
        print(prov_landuse_path)

        # 以地级市为单位创建新的gdb数据库,存储输出结果
        out_single_buff_iden_gdb = os.path.join(landuse_iden_fold, prov_name)
        Creat_New_GDB(landuse_iden_fold, prov_name, out_single_buff_iden_gdb)

        # 遍历存储各省POI缓冲区的gdb,并与土地利用识别分析
        single_buff_gdb_path = os.path.join(poi_path, prov_name)
        arcpy.env.workspace = single_buff_gdb_path
        dijishi_single_buff = arcpy.ListFeatureClasses()
        for single in dijishi_single_buff:
            out_single_buff_iden_name = os.path.join(out_single_buff_iden_gdb, single)
            out_single_buff_iden_name = f'{out_single_buff_iden_name}_iden'
            in_sijishi_single_buff = os.path.join(single_buff_gdb_path, single)

            arcpy.env.workspace = out_single_buff_iden_gdb
            arcpy.env.overwriteOutput = True

            arcpy.analysis.Identity(in_sijishi_single_buff, prov_landuse_path, out_single_buff_iden_name)
            print(f'{out_single_buff_iden_name} identity done!')
            out_dissolve_name = re.sub(r'POI_工厂_buff_iden', '_FactoryLU', out_single_buff_iden_name)
            arcpy.management.Dissolve(out_single_buff_iden_name, out_dissolve_name, ["ORIG_FID", "Level2"])
            print(f'{out_dissolve_name} dissolve done!')
            arcpy.Delete_management(out_single_buff_iden_name)
            print(f'{out_single_buff_iden_name} identity deleted!')

    return


if __name__ == "__main__":
    RootPath = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/9-中国POI数据/'
    RawPOIPath = os.path.join(RootPath, '9.0-全国地级市POI原始数据/')
    GDBPath = os.path.join(RootPath, '9.1-全国各省地级市POI/')

    # 1.将每个省一个csv文件的poi按照地级市分割
    # Split_POI_by_Dijishi(RawPOIPath)

    # 2.新建每个省份/直辖市的GDB,并存放转换为shp的poi
    # CSV_to_Shapfile(RawPOIPath, GDBPath)

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
    out_poi_buff_fold = os.path.join(RootPath, '9.4-全国地级市POI_中类_工厂_缓冲区_500m')
    buff_distance = 500
    # Create_Buffer_of_POI(buff_distance, out_fold_path_II, out_poi_buff_fold)

    # 6.输出每一个缓冲区为shp(后面的代码优化了这一步骤,不再单独输出每一个缓冲区,采取先裁剪后合并的策略)
    # out_single_buff_fold = os.path.join(RootPath, '9.5-全国地级市POI_中类_工厂_缓冲区_逐个输出')
    # POI_Single_Buff_to_Shp(out_poi_buff_fold, out_single_buff_fold)

    # 7.使用每个缓冲区识别土地利用数据
    # 土地利用路径
    RootPath_II = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/10-中国EULUC数据_500m/'
    Single_Prov_Landuse_Fold = os.path.join(RootPath_II, '10.1-EULUC-2018_单个省市/')
    Out_Landuse_fold = os.path.join(RootPath_II, '10.2-全国各地级市EULUC_工厂缓冲区内土地利用/')
    # Landuse_in_Buff(Single_Prov_Landuse_Fold, Out_Landuse_fold, out_poi_buff_fold)







































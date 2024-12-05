# -*- coding: utf-8 -*-
import os
import functools
import time
import numpy as np
import arcpy
from arcpy.sa import *
import pandas as pd
import re
import fnmatch

from sqlalchemy import extract


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


def Creat_New_GDB(rootpath, gdbname, gdbfullname):
    if not arcpy.Exists(gdbfullname):
        arcpy.CreateFileGDB_management(rootpath, gdbname)
        print(f'{gdbname} created successfully!')
    else:
        print(f'{gdbname} is already exists!')


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


def Creat_Monthly_Fold(root_path, year_n):
    for month in range(1, 13):
        if month < 10:
            month_str = '0' + str(month)
            fold_name = f'{year_n}{month_str}'
            fold_path = os.path.join(root_path, fold_name)
            Create_New_Dir(fold_path, fold_name)
        else:
            month_str = str(month)
            fold_name = f'{year_n}{month_str}'
            fold_path = os.path.join(root_path, fold_name)
            Create_New_Dir(fold_path, fold_name)


def Check_String_in_List(in_list, check_string):
    out_string = ''
    for item in in_list:
        if check_string in item:
            out_string = item
    return out_string


def Extract_Ras_of_Single_Month(monthly_data_path, shp_gdb_path, out_month_path):
    # 使用地级市边界逐月切割影像
    shp_gdb_list = os.listdir(shp_gdb_path)
    for prov_gdb in shp_gdb_list:
        prov_gdb_path = os.path.join(shp_gdb_path, prov_gdb)
        # 列举每个省地级市的边界
        arcpy.env.workspace = prov_gdb_path
        dijishi_list = arcpy.ListFeatureClasses()
        # 创建输出GDB
        out_gdb_path = os.path.join(out_month_path, prov_gdb)
        Creat_New_GDB(out_month_path, prov_gdb, out_gdb_path)

        for dijishi in dijishi_list:
            # 构建输出和输出路径
            clip_feature = os.path.join(prov_gdb_path, dijishi)
            out_ras = os.path.join(out_gdb_path, dijishi)
            # 以输出GDB为工作空间，并改为可覆写
            arcpy.env.workspace = out_gdb_path
            arcpy.env.overwriteOutput = True
            # 执行裁剪操作
            extract_result = ExtractByMask(monthly_data_path, clip_feature, "INSIDE")
            extract_result.save(out_ras)


def Extract_Ras_of_Single_Type_Image(images_path, out_path):
    # 读取每一类数据的年度文件夹
    year_list = os.listdir(images_path)
    for year in year_list:
        data_path = os.path.join(images_path, year)
        # 构建输出的年度文件夹
        out_year_path = os.path.join(out_path, year)
        Create_New_Dir(out_year_path, year)
        # 判断每年的每月文件夹是否存在，不存在则新建
        Creat_Monthly_Fold(out_year_path, year)

        # 按照每年逐月的文件夹遍历
        monthly_fold_list = os.listdir(out_year_path)
        for monthly_folder in monthly_fold_list:
            monthly_folder_path = os.path.join(out_year_path, monthly_folder)
            # 获取对应月份的遥感影像路径
            data_list = os.listdir(data_path)
            data_list = fnmatch.filter(data_list, '*.tif')
            monthly_ras_data = Check_String_in_List(data_list, monthly_folder)
            # 打印输入的月度影像，方便查看是否正确读取
            print(monthly_ras_data)
            # 使用地级市边界切割逐月的遥感影像
            monthly_ras_data_path = os.path.join(data_path, monthly_ras_data)
            Extract_Ras_of_Single_Month(monthly_ras_data_path, SHP_GDB_Root_Path, monthly_folder_path)


def Excute_Extract_Ras_of_Single_Type_Image(path_ls):
    for path in path_ls:
        path_full = f'{path}月合成-中国2014-2022/'
        origi_data_path = os.path.join(Origi_RS_Data_Root_Path, path_full)
        out_dijishi_path = os.path.join(Dijishi_RS_Data_Path, f'11.1-{path[:-16]}')
        Extract_Ras_of_Single_Type_Image(origi_data_path, out_dijishi_path)


def Mask_RS_Data_by_POI_EULUC(rs_path, out_rs_path, poi_path, euluc_path):
    # 列出每种遥感影像的年度列表
    year_list = os.listdir(rs_path)
    for year in year_list:
        # 创建输出的年度文件夹
        out_year_path = os.path.join(out_rs_path, year)
        Create_New_Dir(out_year_path, year)
        # 列出每种遥感影像每年的月度列表
        year_path = os.path.join(rs_path, year)
        month_list = os.listdir(year_path)
        for month in month_list:
            # 创建输出的月度文件夹
            out_month_path = os.path.join(out_year_path, month)
            Create_New_Dir(out_month_path, month)
            # 列出每月的各省GDB
            month_path = os.path.join(year_path, month)
            gdb_list = os.listdir(month_path)
            for gdb in gdb_list:
                # 创建输出的各省GDB
                out_gdb_path = os.path.join(out_month_path, gdb)
                Creat_New_GDB(out_month_path, gdb, out_gdb_path)

                poi_buff_gdb_path = os.path.join(poi_path, gdb)
                euluc_gdb_path = os.path.join(euluc_path, gdb)

                # 列举每个省的地级市遥感影像，然后使用POI和EULUC栅格对其进行掩膜
                gdb_path = os.path.join(month_path, gdb)
                arcpy.env.workspace = gdb_path
                dijishi_ras_list = arcpy.ListRasters()

                for dijishi_ras in dijishi_ras_list:
                    dijishi_ras_path = os.path.join(gdb_path, dijishi_ras)
                    poi_extract_path = os.path.join(poi_buff_gdb_path, f'{dijishi_ras}_POI_工厂_buff')
                    euluc_extract_path = os.path.join(euluc_gdb_path, f'{dijishi_ras}_EULUC_Indus')
                    # euluc是临时图层，会mosaic到poi图层上，所有POI图层保持地级市原始名称
                    out_dijishi_poi_extr_path = os.path.join(out_gdb_path, f'{dijishi_ras}')
                    out_dijishi_euluc_extr_path = os.path.join(out_gdb_path, f'{dijishi_ras}_euluc_extr')

                    arcpy.env.workspace = out_gdb_path
                    arcpy.env.overwriteOutput = True
                    # 分别用poi和euluc对遥感影像进行extract，尝试过将poi和euluc合并为一个图层，但是执行栅格计算器后会出现只保留交集的问题
                    try:
                        poi_extract_result = arcpy.sa.ExtractByMask(dijishi_ras_path, poi_extract_path, "INSIDE")
                        poi_extract_result.save(out_dijishi_poi_extr_path)
                        euluc_extract_result = arcpy.sa.ExtractByMask(dijishi_ras_path, euluc_extract_path, "INSIDE")
                        euluc_extract_result.save(out_dijishi_euluc_extr_path)

                        arcpy.Mosaic_management(out_dijishi_euluc_extr_path, out_dijishi_poi_extr_path)
                        arcpy.Delete_management(out_dijishi_euluc_extr_path)
                    except Exception as e:
                        print(e)
                    print(f'{dijishi_ras} done!')


def Mask_RS_Data_by_POI_Buff_Dist(path_list, buff_dist_list, rs_root_path, poi_path, euluc_path):
    for buff_dist in buff_dist_list:
        poi_buff_last = f'{buff_dist}m_Ras/'
        poi_buff_path = f'{poi_path}{poi_buff_last}'
        # buff_dist = 500 #1000, 1500

        for path in path_list:
            RS_Data_Path = os.path.join(rs_root_path, f'11.1-{path}')
            Out_RS_Data_Path = os.path.join(rs_root_path, f'11.2-{path}_Extract_{buff_dist}m')
            Mask_RS_Data_by_POI_EULUC(RS_Data_Path, Out_RS_Data_Path, poi_buff_path, euluc_path)



if __name__ == "__main__":
    All_Root_Path = r'E:/ChinaMonthlyIndustrial/'

    # # 将每月的夜间灯光数据转化为点，方便后续计算，提高效率
    # NTL_rootpath = r'D:/中国工业总产值估算原始数据处理/夜间灯光月合成-中国2014-2022/'
    # EVI_rootpath = r'D:/中国工业总产值估算原始数据处理/增强植被指数月合成-中国2014-2022/'
    # LST_rootpath = r'D:/中国工业总产值估算原始数据处理/地表温度月合成-中国2014-2022/'
    #
    # out_ras2fc_root = r'D:/中国工业总产值估算中间数据/中国夜间灯光处理中间数据/中国夜间灯光月合成栅格转点/中国夜间灯光月合成栅格转点2014-2016.gdb/'
    # out_ras2fc_root_evi = r'D:/中国工业总产值估算中间数据/中国植被指数处理中间数据/中国植被指数月合成栅格转点/evi_ras2point_2014-2016.gdb/'
    # out_ras2fc_root_lst = r'D:/中国工业总产值估算中间数据/中国地表温度处理中间数据/中国地表温度月合成栅格转点/lst_ras2point_2014-2016.gdb/'
    # # Raster_to_Points(NTL_rootpath, out_ras2fc_root)

    # # 将所有工厂缓冲区土地利用矢量路径存储为csv
    # flexible_buffer_fold = r'10-中国EULUC数据_1000m/10.2-全国各地级市EULUC_工厂缓冲区内土地利用/'
    # out_filelist = os.path.join(All_Root_Path, '地级市工厂缓冲土地利用路径列表.csv')
    # # Write_Landuse_Path(rootpath, flexible_buffer_fold, out_filelist)

    # # 读取整饰完成后的featureclass路径，并执行空间连接输出为excel
    # ras2fc_gdb = r'E:/中国植被指数月合成栅格转点/evi_ras2point_2014-2016.gdb/'
    # out_excel_root = os.path.join(All_Root_Path, f'11-工厂缓冲区土地利用对应植被指数excel/')
    # out_join_feature_root = os.path.join(All_Root_Path,
    #                                      f'11-工厂缓冲区土地利用对应植被指数join/工厂缓冲区土地利用与植被指数矢量点连接2014-2016.gdb/')
    # file_list_df = pd.read_csv(out_filelist)
    # # Spatial_Join_to_Excel(file_list_df, ras2fc_gdb, out_join_feature_root, out_excel_root)

    # 1.地级市矢量切割每月的各类遥感影像
    SHP_GDB_Root_Path = os.path.join(All_Root_Path, r'0-地级市矢量-中文/')
    Origi_RS_Data_Root_Path = r'D:/中国工业总产值估算原始数据处理/'
    Dijishi_RS_Data_Path = os.path.join(All_Root_Path, r'11-各类遥感数据地级市分割/')
    # 按照影像类型逐类切割
    Path_List = ['地表温度', '夜间灯光', '增强植被指数'] #, '夜间灯光', '增强植被指数'
    # Excute_Extract_Ras_of_Single_Type_Image(Path_List)

    # 2.按照poi的缓冲区距离对各类栅格影像掩膜处理
    Buff_Dist_List = [500, 1000, 1500]
    EULUC_Root_Path = os.path.join(All_Root_Path, r'10-中国ELUC数据/10.2-中国EULUC_工业用地_地级市_Ras/')
    RS_Data_Root_Path = os.path.join(All_Root_Path, r'11-各类遥感数据地级市分割/')
    POI_Root_Path = os.path.join(All_Root_Path, f'9-中国POI数据/9.6-全国地级市POI_Buff_')
    # Mask_RS_Data_by_POI_Buff_Dist(Path_List, Buff_Dist_List, RS_Data_Root_Path, POI_Root_Path, EULUC_Root_Path )






























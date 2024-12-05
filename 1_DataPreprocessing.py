# -*- coding: utf-8 -*-
import arcpy
import arcpy.da
import os
import functools
import time
import codecs
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


def ReadFile(filePath, encoding="utf-8"):
    with codecs.open(filePath, "r", encoding) as f:
        return f.read()


def WriteFile(filePath, u, encoding="utf-8-sig"):
    with codecs.open(filePath, "wb") as f:
        f.write(u.encode(encoding, errors="ignore"))


@Time_Decorator
def UTF8_2_GBK(src, dst):
    # 转换UTF8编码的csv，解决乱码问题
    content = ReadFile(src, encoding="utf-8")
    WriteFile(dst, content, encoding="utf-8-sig")
    UTF8_2_GBK(src, dst)


@Time_Decorator
def Change_File_into_Uniform_Style(EVIPath, P1, P2):
    name_ls = os.listdir(EVIPath)
    # 分别替换文件名中的P1和P2字符串
    for name in name_ls:
        if P1 in name:
            new_name = re.sub(P1, '_P1.tif', name)
            new_name_path = os.path.join(EVIPath, new_name)
            old_name_path = os.path.join(EVIPath, name)
            try:
                os.rename(old_name_path, new_name_path)
                print(f"文件已从 {name} 重命名为 {new_name}")
            except Exception as e:
                print(f"错误：重命名文件 {name} 时发生错误 - {e}")
        elif P2 in name:
            new_name = re.sub(P2, '_P2.tif', name)
            new_name_path = os.path.join(EVIPath, new_name)
            old_name_path = os.path.join(EVIPath, name)
            try:
                os.rename(old_name_path, new_name_path)
                print(f"文件已从 {name} 重命名为 {new_name}")
            except Exception as e:
                print(f"错误：重命名文件 {name} 时发生错误 - {e}")


def Get_Unique_in_List(in_list, position):
    new_ls = []
    for item in in_list:
        new_ls.append(item[:position])
    new_ls = sorted(set(new_ls))
    return new_ls


@Time_Decorator
def Mosic2New_and_Clip(EVIPath, out_EVIPath, clip_shp):
    name_ls = os.listdir(EVIPath)
    name_ls = fnmatch.filter(name_ls, '*.tif')
    position = 10  #MOD_EVI:18 #MCD_EVI: 14
    new_name_ls = Get_Unique_in_List(name_ls, position)

    # 将原始数据路径作为工作空间，对同名的EVI进行合并操作
    arcpy.env.workspace = EVIPath
    for name in new_name_ls:
        arcpy.env.workspace = EVIPath
        target_raster = f'{name}_P1.tif'
        in_raster = f'{name}_P2.tif'
        # out_name = f'{name}.tif'
        arcpy.Mosaic_management(in_raster, target_raster)

        # 将合并完成后的EVI图层，逐个按照国界裁剪
        arcpy.env.workspace = out_EVIPath
        arcpy.env.overwriteOutput = True
        extract_area = 'INSIDE'
        out_ras = f'{name}.tif'
        in_ras = os.path.join(EVIPath, target_raster)
        extract_result = arcpy.sa.ExtractByMask(in_ras, clip_shp, extract_area)
        extract_result.save(out_ras)
        print(name)


def Excute_ChangeName_Clip_Merge(image_root, p1, p2, shp_path, image_types):
    for image in image_types:
        image_path = os.path.join(image_root, image)
        Change_File_into_Uniform_Style(image_path, p1, p2)

        processed_image_path = os.path.join(image_root, f'{image}_Processed/')
        Mosic2New_and_Clip(image_path, processed_image_path, shp_path)


def Create_FIRMS_Buff(in_path, out_path, distance):
    year_list = os.listdir(in_path)
    for year in year_list:
        year_path = os.path.join(in_path, year)
        out_year_path = os.path.join(out_path, year)
        Create_New_Dir(out_year_path, year)

        firms_list = os.listdir(year_path)
        firms_list = fnmatch.filter(firms_list, '*.shp')

        arcpy.env.workspace = out_year_path
        arcpy.env.overwriteOutput = True

        for firm in firms_list:
            firms_path = os.path.join(year_path, firm)
            out_firms_path = os.path.join(out_year_path, firm)
            arcpy.Buffer_analysis(firms_path, out_firms_path, distance)

    print(f'{distance}m Buffer created successfully!')


def Buff_Shp_to_Ras(in_path, out_path, cellsize, feild):
    year_list = os.listdir(in_path)
    for year in year_list:
        firms_year_path = os.path.join(in_path, year)
        out_firms_year_path = os.path.join(out_path, year)
        Create_New_Dir(out_firms_year_path, year)

        firms_list = os.listdir(firms_year_path)
        firms_list = fnmatch.filter(firms_list, '*.shp')
        for firms in firms_list:
            firms_path = os.path.join(firms_year_path, firms)
            out_firms_path = os.path.join(out_firms_year_path, f'{firms[:-4]}.tif')
            arcpy.FeatureToRaster_conversion(firms_path, feild, out_firms_path, cellsize)
        print(f'{year} done!')


def Dijishi_Extract_Firms_Ras(in_firms_ras, shp_path, month_folder):
    dijishi_gdb_list = os.listdir(shp_path)
    for dijishi_gdb in dijishi_gdb_list:
        dijishi_gdb_path = os.path.join(shp_path, dijishi_gdb)
        # 创建每月的输出文件夹
        out_firms_dijishi_gdb_path = os.path.join(month_folder, dijishi_gdb)
        Creat_New_GDB(month_folder, dijishi_gdb, out_firms_dijishi_gdb_path)

        arcpy.env.workspace = dijishi_gdb_path
        dijishi_list = arcpy.ListFeatureClasses()

        for dijishi in dijishi_list:
            dijishi_path = os.path.join(dijishi_gdb_path, dijishi)
            # 创建输出路径，并设置输出路径的GDB为工作空间，开启覆写
            out_dijishi_path = os.path.join(out_firms_dijishi_gdb_path, dijishi)
            arcpy.env.workspace = out_firms_dijishi_gdb_path
            arcpy.env.overwriteOutput = True
            try:
                extract_result = arcpy.sa.ExtractByMask(in_firms_ras, dijishi_path, "INSIDE")
                extract_result.save(out_dijishi_path)
            except Exception as e:
                print(e)
                print(dijishi)
        print(f'{dijishi_gdb} done!')


def Dijishi_Extract_Firms(firms_ras_path, dijishi_firms_ras_path, dijishi_shp_path):
    year_list = os.listdir(firms_ras_path)
    for year in year_list:
        firms_year_path = os.path.join(firms_ras_path, year)
        firms_buff_list = os.listdir(firms_year_path)
        firms_buff_list = fnmatch.filter(firms_buff_list, '*.tif')
        # 构建输出年份文件夹
        out_firms_year_path = os.path.join(dijishi_firms_ras_path, year)
        Create_New_Dir(out_firms_year_path, year)

        for firms in firms_buff_list:
            # 输出月份路径
            month_folder_name = f'{year}{firms[-6:-4]}'
            month_folder_path = os.path.join(out_firms_year_path, month_folder_name)
            Create_New_Dir(month_folder_path, month_folder_name)

            # 输入栅格，全中国每月的火点缓冲区栅格
            firms_path = os.path.join(firms_year_path, firms)

            Dijishi_Extract_Firms_Ras(firms_path, dijishi_shp_path, month_folder_path)


if __name__ == "__main__":
    RootPath = r'D:/中国工业总产值估算原始数据/'

    # 1.统一修改evi文件名,然后裁剪、合并每月的两幅evi
    pattern1 = '-0000000000-0000046592.tif'
    pattern2 = '-0000000000-0000069888.tif'
    China_SHP_Path = r'F:/ChinaShapefile/ChinaProvince_HKMacau/ChinaProvince_ALL_Merge.shp'
    Image_Types = ['MCD_EVI', 'MOD_EVI', 'MOD_LST']
    # Excute_ChangeName_Clip_Merge(RootPath, pattern1, pattern2, China_SHP_Path, Image_Types)

    # 2. 每月火点数据生成缓冲区
    Root_Path = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/'
    FIRMS_Root_Path = os.path.join(Root_Path, r'8-中国火点数据2014-2023/')

    Original_FIRMS_Path = os.path.join(FIRMS_Root_Path, r'8.1-中国火点数据原始/')
    buffer_distance_number = 500
    buffer_distance = f'{buffer_distance_number} METERS'
    FIRMS_Buffer_Path = os.path.join(FIRMS_Root_Path, f'8.2-中国火点数据缓冲区_{buffer_distance_number}m/')
    # Create_FIRMS_Buff(Original_FIRMS_Path, FIRMS_Buffer_Path, buffer_distance)

    # 3.缓冲区矢量转栅格
    FIRMS_Buff_Ras_Path = os.path.join(FIRMS_Root_Path, f'8.3-中国火点数据缓冲区_{buffer_distance_number}m_Ras/')
    F2R_feild = 'BUFF_DIST'
    cell_size = 0.0041666667
    # Buff_Shp_to_Ras(FIRMS_Buffer_Path, FIRMS_Buff_Ras_Path, cell_size, F2R_feild)

    # 4.缓冲区矢量地级市切割
    FIRMS_Buff_Dijishi_Path = os.path.join(FIRMS_Root_Path, f'8.4-中国地级市火点数据缓冲区栅格_{buffer_distance_number}m/')
    Dijishi_Shp_Path = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/0-地级市矢量-中文/'
    Dijishi_Extract_Firms(FIRMS_Buff_Ras_Path, FIRMS_Buff_Dijishi_Path, Dijishi_Shp_Path)

















































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


def ReadFile(filePath,encoding="utf-8"):
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
    position = 10 #MOD_EVI:18 #MCD_EVI: 14
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


if __name__ == "__main__":
    RootPath = r'F:/'
    MCDEVIPath = os.path.join(RootPath, 'MCD_EVI/')
    MODEVIPath = os.path.join(RootPath, 'MOD_EVI/')
    MODLSTPath = os.path.join(RootPath, 'MOD_LST/')

    # 1.统一修改evi文件名
    pattern1 = '-0000000000-0000046592.tif'
    pattern2 = '-0000000000-0000069888.tif'
    # Change_File_into_Uniform_Style(MODLSTPath, pattern1, pattern2)

    # 2.裁剪、合并每月的两幅evi
    out_evi_path = os.path.join(RootPath, 'MCD_EVI_Processed/')
    out_modevi_path = os.path.join(RootPath, 'MOD_EVI_Processed')
    out_modlst_path = os.path.join(RootPath, 'MOD_LST_Processed')
    cliped_shp_path = r'F:/ChinaShapefile/ChinaProvince_HKMacau/ChinaProvince_ALL_Merge.shp'
    # Mosic2New_and_Clip(MODLSTPath, out_modlst_path, cliped_shp_path)



































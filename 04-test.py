# -*- coding: utf-8 -*-
import os
import functools
import time
import pandas as pd
import fnmatch
import torch
from torchvision import transforms
import shutil
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

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


def Derived_Match_Pattern(data_path):
    y_data_files = os.listdir(data_path)
    pattern_list = []
    for y_file in y_data_files:
        print(y_file)
        y_file_path = os.path.join(data_path, y_file)
        y_df = pd.read_excel(y_file_path, index_col=False)
        y_df = y_df.dropna(how='any')
        for index, row in y_df.iterrows():
            Y_M = int(row['月份'])
            match_pattern = f'{str(Y_M)}_{y_file}'
            # print(match_pattern)
            pattern_list.append(match_pattern)
    # print(new_pattern_list)
    return pattern_list

@ Time_Decorator
def List_Data_Path_Needed_to_Transform(y_data, x_data):
    y_pattern_list = Derived_Match_Pattern(y_data)
    x_data_folders = os.listdir(x_data)
    patterns = []
    for x_folder in x_data_folders:
        x_data_fold = os.path.join(x_data, x_folder)
        x_data_months = os.listdir(x_data_fold)
        for x_month in x_data_months:
            x_month_path = os.path.join(x_data_fold, x_month)
            x_files = os.listdir(x_month_path)
            for pettern in y_pattern_list:
                match_element = fnmatch.filter(x_files, f'*{pettern}')
                if match_element == []:
                    pass
                else:
                    match_element = f'{x_month_path}/{match_element[0]}'
                    patterns.append(match_element)
    return patterns


def Copy_Partial_X_Data(patterns, root_path):

    folders = os.listdir(root_path)
    for fold in folders:
        fold_path = os.path.join(root_path, fold)
        for pattern in patterns:
            if fold in pattern:
                destination_file = os.path.join(fold_path, os.path.basename(pattern))
                shutil.copy(pattern, destination_file)




@Time_Decorator
def Read_Xtensors(XFolder, outXTensor_path):
    # 将每月的经过处理的灯光矩阵，转换成tensor形式
    # inYFile: excel形式存储的工业产值
    # outYTensor：适合网络学习用的tensor形式

    # 定义一个空tensor储存所有月的数据，并将形状预定义为side_shape
    files = os.listdir(XFolder)
    file_path = os.path.join(XFolder, files[0])
    df_all = pd.read_excel(file_path, header=None)
    side_shape = df_all.shape[0]
    xtensor = torch.empty([0, 1, side_shape, side_shape])

    for file in files:
        # 1.读取每月excel中的灯光统计信息为dataframe
        df_path = os.path.join(XFolder, file)
        df_file = pd.read_excel(df_path, header=None)
        df = df_file.iloc[:, 0:side_shape]  # 读取维度(216,216)
        # 2.将dataframe转换为nparray
        df_arr = df.values
        # 3.使用transforms.ToTensor可以将tensor增加一维，本来主要是用来处理图片
        trans = transforms.ToTensor()  # 会增加一个维度([1, 216,216])
        tensor = trans(df_arr)
        # 4.继续给tensor增加维度，为了使其符合卷积的输入要求
        tensor = tensor.unsqueeze(0)  # 继续增加一维([1, 1, 216,216])
        # 将每月的数据作为一张灰度图像，即只有一个通道的二维图像，然后拼接为一个
        xtensor = torch.vstack((xtensor, tensor))
    # 将创建的全零向量删除
    xtensor = xtensor.squeeze(0)
    # 存储tensor
    torch.save(xtensor, outXTensor_path)
    print(f'{outXTensor_path}_XTensor存储成功！')
    return


def Read_YTensor(inYFile, outYTensor):
    # 将工业总产值（Y）转换成tensor形式
    # inYFile: excel形式存储的工业产值
    # outYTensor：适合网络学习用的tensor形式

    y_df = pd.read_excel(inYFile)
    y_df = y_df.iloc[:, 1:2]
    y = y_df.dropna(how='any')
    ytensor = torch.from_numpy(y.values).squeeze()
    torch.save(ytensor, outYTensor)
    print('YTensor存储成功！')
    return


@Time_Decorator
def Shift_Data_to_CNN_Shape(folderpath, outfilepath):
    # 把原始的长条数据reshape变为方便CNN处理的方形数据
    # folderpath：经过地理处理后的各土地利用类型下的灯光强度矩阵
    # outfilepath：满足CNN输入数据的输出路径

    files = os.listdir(folderpath)
    files = fnmatch.filter(files, 'SVDNB*')
    for filename in files:
        df_all = pd.read_excel(os.path.join(folderpath, filename), header=0)
        df_all = df_all.iloc[:, 2:15]
        df_all = df_all.fillna(0)
        # 计算原始df的列数和大小，存储列名
        df_cols = df_all.shape[1]
        df_size = df_all.size
        df_colnames = df_all.columns
        # 计算新的方形矩阵的边长（向上取整）
        side_length = int(np.ceil(np.sqrt(df_size)))
        if side_length % 2 != 0:
            side_length += 1
        # 将原数组拉平
        rect_array = df_all.values
        flat_array = rect_array.flatten()
        # 计算需要填充的零的数量
        zero_count = side_length*side_length-df_size
        zero_arr = np.zeros((1, zero_count), dtype=rect_array.dtype)
        flat_array = np.append(flat_array, zero_arr)
        # 重塑数组为方形
        square_array = flat_array.reshape((side_length, side_length))

        # #填补空缺和零值为0.001
        reshaped_df = pd.DataFrame(square_array)
        reshaped_df_out = reshaped_df.replace(0, 0.0001)
        reshaped_df_out.to_excel(os.path.join(outfilepath, filename), header=None, index=False)
        print(f"{filename} has been processed successfully!")


def Read_tensors(Xtensorpath, Ytensorpath):
    # 读取xytensor，并完成归一化等预处理，然后使用dataloader操作数据
    # Xtensorpath: xtensor的路径
    # Ytensorpath: ytensor的路径

    #读取tensor
    xtensor = torch.load(Xtensorpath)
    ytensor = torch.load(Ytensorpath)
    print('Tensor读取成功！')

    # 转换tensor字符类型为float，匹配卷积操作的字符类型
    xtensor = xtensor.float()
    ytensor = ytensor.float()
    # 归一化Xtensor和Ytensor
    x_mean, x_std = torch.mean(xtensor, dim=0), torch.std(xtensor, dim=0)
    y_mean, y_std = torch.mean(ytensor, dim=0), torch.std(ytensor, dim=0)
    # x_norm = 1E-06 + ((xtensor - x_mean) / x_std)
    x_norm = xtensor
    y_norm = (ytensor - y_mean) / y_std
    return x_norm, y_norm, y_mean, y_std


def Get_File_Path(path, string):
    # path: 要匹配的文件夹
    # string：待匹配的文件名称含有的字符串

    filepath = [file for file in os.listdir(path) if string in file]
    outpath = os.path.join(path, filepath[0])
    return outpath


def Check_FolderPath_Exist(outFolderPath):
    if not os.path.exists(outFolderPath):
        os.makedirs(outFolderPath)
        print(f'{outFolderPath} is created successflly!')
    return


def Train_Test_Data_Loader(x_norm, y_norm, Batchsize):
    # 数据读进dataloader，方便后续训练
    torch_dataset = TensorDataset(x_norm, y_norm)  # 组成torch专门的数据库
    # 划分训练集测试集与验证集
    torch.manual_seed(seed=2021)  # 设置随机种子分关键，不然每次划分的数据集都不一样，不利于结果复现
    # 先将数据集拆分为训练集+验证集（共108组），测试集（10组）
    split_shape = torch_dataset.tensors[0].shape[0]
    train_validaion, test = random_split(torch_dataset, [80, split_shape-80])
    # 再将训练集划分批次，每batch_size个数据一批（测试集与验证集不划分批次）
    train_data_dl = DataLoader(train_validaion, batch_size=Batchsize, shuffle=True)
    test_dl = DataLoader(test, batch_size=Batchsize, shuffle=False)
    print('Dataloader完成！')
    return train_data_dl, test_dl

def Allset_Data_Loader(x_norm, y_norm):  # y_mean, y_std, outexcel, device
    # 将整个数据集读进dataloader
    torch_dataset = TensorDataset(x_norm, y_norm)
    data_dl = DataLoader(torch_dataset, batch_size=108, shuffle=False)
    return data_dl


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  #1.第一层卷积输入的灯光统计数据1通道，相当于灰度图，大小为256行X180列，即(1, 288, 180)
            nn.Conv2d(
                in_channels=1,  #数据输入的通道数，对于彩色图片是3，对于灰度图是1
                out_channels=16,  #卷积核的个数，每个卷积核都会生成一层新的卷积特征
                kernel_size=3,  #卷积核的大小
                stride=1,  #卷积核每次移动的距离
                padding=1,  #如果想要卷积出来的图片长宽没有变化, padding=(kernel_size-1)/2
            ),  #2.输出时的形状为(16, 288, 180),形状保持不变
            nn.ReLU(),  #激活函数
            nn.MaxPool2d(kernel_size=5, stride=2)  #3.在2x2的空间里使用最大值向下采样,输出的形状为(16,144,90)
        )
        self.conv2 = nn.Sequential(  #4.第二层卷积输入的为上一层的输出，即(16,144,90)
            nn.Conv2d(16, 32, 3, 1, 1),  #5.输出的形状为(32,144,90)
            nn.ReLU(),
            nn.MaxPool2d(5, 2),  #6.再次下采样，输出形状为(32,72,45)
        )
        self.conv3 = nn.Sequential(  #4.第三层卷积输入的为上一层的输出，即(16,144,90)
            nn.Conv2d(32, 64, 3, 1, 1),  #5.输出的形状为(64,144,90)
            nn.ReLU(),
            nn.MaxPool2d(5, 2),  #6.再次下采样，输出形状为(64,33,19)
        )
#         self.conv4 = nn.Sequential(  #4.第三层卷积输入的为上一层的输出，即(16,144,90)
#             nn.Conv2d(64, 128, 3, 1, 'same'),  #5.输出的形状为(64,144,90)
#             nn.ReLU(),
#             nn.MaxPool2d(5, 2),  #6.再次下采样，输出形状为(64,33,19)
#         )
        #通过改变全连接层的输出，决定时分类还是回归，此外，还要注意损失函数的定义
        # self.fc1 = nn.Linear(64*13*13, 128)
        # self.out = nn.Linear(10816, 40)  # 全连接层输出：36864(3层卷积)
        self.CalculateFlattenSize()



    def CalculateFlattenSize(self):
        # 创建一个假的输入数据，用于模拟网络的前向传播
        test_input = torch.rand(1, 1, 194, 194)#北京130，佛山361, 广州304，深圳市236，成都市194
        test_input = self.conv1(test_input)
        test_input = self.conv2(test_input)
        test_input = self.conv3(test_input)
        # 计算展平后的特征数量
        self.flatten_size = test_input.view(1, -1).size(1)
        self.out = nn.Linear(self.flatten_size, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
#         x = self.conv4(x)
#         print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        output = self.out(x)
        return output





def Train_CNN(xy_tensor_path, city_name):
    def Get_Input_Shape(train_data_dl):
        input_shape = ''
        for batch_idx, (inputs, targets) in enumerate(train_data_dl):
            input_shape = inputs.shape
            if input_shape is not None:
                break
        return input_shape


    def Trian_Modle(train_data_dl, epoch, device):
        # train_data_dl：用于训练的dataloader

        total_train_step = 0
        for ep in range(epoch):
            if ep % 1000 == 0:
                print(f'第{ep}次epoch')
            cnn.train()
            for batch_idx, (inputs, targets) in enumerate(train_data_dl):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = cnn(inputs)
                loss = loss_func(outputs, targets.unsqueeze(1))
                # 以下是固定写法
                optimizer.zero_grad()  # 梯度归零
                loss.backward()  # 误差传播
                optimizer.step()  # 应用梯度

                # 打印训练过程
                total_train_step += 1
                if total_train_step % 1000 == 0:
                    print(f'第{total_train_step}次训练，loss = {loss.item()}')
        return

    def Test_Model(test_dl, y_mean, y_std, outexcel, device):
        # test_dl: 用于测试的dataloader
        # y_mean，y_std：ytensor的均值和标准差

        df = pd.DataFrame(columns=['True', 'Pred'])
        for batch_idx, (testinputs, targets) in enumerate(test_dl):
            testinputs = testinputs.to(device)
            targets = targets.to(device)
            testoutput = cnn(testinputs)
            pred = testoutput.data.cpu().numpy().squeeze() * y_std.numpy() + y_mean.numpy()
            real = targets.cpu().numpy() * y_std.numpy() + y_mean.numpy()
            # 将每一个batch的数据装进df中
            df_temp = pd.DataFrame({'True': real, 'Pred': pred})
            df_temp_cleaned = df_temp.dropna(axis=1, how='all')
            df = pd.concat([df, df_temp_cleaned], axis=0, ignore_index=True)
        print('Predicted number:', df['Pred'])
        print('Real number:', df['True'])
        df.to_excel(outexcel)
        # 展示测试数据结果
        x_true, y_pred = df['True'], df['Pred']
        sns.regplot(x=x_true, y=y_pred)
        plt.show()
        return

    def All_Model(all_dl, y_mean, y_std, outexcel, device):
        # all_dl: 全体数据的dataloader
        # y_mean，y_std：ytensor的均值和标准差

        df = pd.DataFrame(columns=['True', 'Pred'])
        for batch_idx, (inputs, targets) in enumerate(all_dl):
            inputs = inputs.to(device)
            targets = targets.to(device)
            testoutput = cnn(inputs)
            pred = testoutput.data.cpu().numpy().squeeze() * y_std.numpy() + y_mean.numpy()
            real = targets.cpu().numpy() * y_std.numpy() + y_mean.numpy()
            # 将每一个batch的数据装进df中
            df_temp = pd.DataFrame({'True': real, 'Pred': pred})
            df_temp_cleaned = df_temp.dropna(axis=1, how='all')
            df = pd.concat([df, df_temp_cleaned], axis=0, ignore_index=True)
        print('Predicted number:', df['Pred'])
        print('Real number:', df['True'])
        # 为每一行增加月份信息
        # date_strings = [f"{Y}{str(M).zfill(2)}" for Y in range(2014, 2023) for M in range(1, 13)]
        # df['Month'] = date_strings
        df.to_excel(outexcel)
        # 展示测试数据结果
        x_true, y_pred = df['True'], df['Pred']
        sns.regplot(x=x_true, y=y_pred)
        plt.show()
        return


    EPOCH = 20000
    BATCH_SIZE = 10
    LR = 0.0001

    XTensorpath = Get_File_Path(xy_tensor_path, 'X')
    YTensorpath = Get_File_Path(xy_tensor_path, 'Y')
    XNorm, YNorm, YMean, YStd = Read_tensors(XTensorpath, YTensorpath)
    TrainDL, TestDL = Train_Test_Data_Loader(XNorm, YNorm, BATCH_SIZE)

    # input_shape_size = Get_Input_Shape(TrainDL)

    # 初始化网络
    cnn = CNN()
    # 若CUDA存在，则将网络放到GPU上运算
    Device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = cnn.to(Device)
    print('网络初始化完成！')
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    print('损失函数定义完成！')

    # 开始训练网络
    Trian_Modle(TrainDL, EPOCH, Device)
    # 定义测试结果输出路径
    outResultFolder = os.path.join(xy_tensor_path, 'SA-CNN_Results')
    Check_FolderPath_Exist(outResultFolder)
    testoutCNNResult = os.path.join(outResultFolder, f'{city_name}_test_Result.xlsx')
    # 测试网络结果
    Test_Model(TestDL, YMean, YStd, testoutCNNResult, Device)
    print(f'------------{city_name} Done----------------')

    # 定义结果输出路径
    outCNNResult = os.path.join(outResultFolder, f'{city_name}_All_test_Result.xlsx')
    # 测试全体数据集结果
    AllDL = Allset_Data_Loader(XNorm, YNorm)  # YMean, YStd, outCNNResult, Device
    All_Model(AllDL, YMean, YStd, outCNNResult, Device)


if __name__ == "__main__":
    root_path = r'C:/Users/KJ/Documents/ChinaMonthlyIndustrial/'
    x_data_path = os.path.join(root_path, r'12-工厂缓冲区内各类特征提取/')
    y_data_path = os.path.join(root_path, r'13-Transformer建模_初步训练+测试_Ydata/')
    part_x_data_path = os.path.join(root_path, r'13-Transformer建模_初步训练+测试_Xdata/')
    part_x_data_reshape_path = os.path.join(root_path, r'13-Transformer建模_初步训练+测试_Xdata_Reshape/')
    tensor_root = os.path.join(root_path, r'14-Tensor存储/')
    # cuda = torch.cuda.is_available()
    # print(cuda)

    # 3.Excel数据整合、存储
    # real_pattern = List_Data_Path_Needed_to_Transform(y_data_path, x_data_path)
    # Copy_Partial_X_Data(real_pattern, part_x_data_path)

    city_lsit = ['四川省_成都市'] #'北京','广东省_佛山市', '广东省_广州市', '广东省_深圳市', '上海市',
    for city in city_lsit:
        city_x_data = os.path.join(part_x_data_path, city)
        # 将excel数据输出成方形数组
        out_reshape_excel = os.path.join(part_x_data_reshape_path, city)
        # Shift_Data_to_CNN_Shape(city_x_data, out_reshape_excel)
        # 方形数组存储为tensor
        outXTensor = os.path.join(tensor_root, f'{city}/X_{city}.pt')
        # Read_Xtensors(out_reshape_excel, outXTensor)
        # 读取y值并存储为tensor
        part_y_data_path = os.path.join(y_data_path, f'{city}.xlsx')
        outYTensor = os.path.join(tensor_root, f'{city}/Y_{city}.pt')
        Read_YTensor(part_y_data_path, outYTensor)

        tensor_path = os.path.join(tensor_root, city)
        Train_CNN(tensor_path, city)

























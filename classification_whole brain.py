import numpy as np
import math
import os
import scipy.io as scio
from sklearn.model_selection import train_test_split  # 用于切分训练集和测试集
from sklearn.decomposition import PCA  # PCA降维
from sklearn.svm import SVC    # 支持向量机
from sklearn.metrics import confusion_matrix as CM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import time as time
import matplotlib.pyplot as plt
import xlwt
from xlrd import open_workbook
from xlutils.copy import copy


repeat = 100


data = []  # 存放数据
label = []  # 存放标签
ACC_list = []
AUC_list = []
sensitivity_list = []
specifity_list = []

all_TP = np.zeros((repeat, 100), dtype=int)
all_TN = np.zeros((repeat, 100), dtype=int)
all_FP = np.zeros((repeat, 100), dtype=int)
all_FN = np.zeros((repeat, 100), dtype=int)


def getvalue(mat):
    each_mat = mat
    array_struct = scio.loadmat(each_mat)
    array_data = array_struct['connectivity']  # 取出需要的数字矩阵部分
    array_data_temp = array_data + 1  # 临时给每个元素值都+1
    array_data_temp = np.triu(array_data_temp, k=1)  # 只保留上三角，且对角线的值和下三角的值置为0
    array_data_temp = array_data_temp.flatten()  # 成一维向量
    array_data_temp.tolist()
    array_data_temp = list(filter(lambda x: x != 0, array_data_temp))
    # array_data = list(filter(lambda x: x != -4, array_data_temp))  # 因为原始数据 给不显著的元素值置为了-5
    array_data = np.array(array_data_temp)
    array_data = array_data - 1  # 把剩余上三角的1减回去
    # print(array_data.shape)  # 删除一维向量中的0元素

    return array_data


def getresult(n, m, kenel, dizhi1):
    folder = 'D:/python/mlJNE/MCI/data/' + dizhi1 + '/MCI/'
    folder2 = 'D:/python/mlJNE/MCI/data/' + dizhi1 + '/NC/'

    path_NC = os.listdir(folder)
    path_JME = os.listdir(folder2)

    for each_mat_NC in path_NC:
        mat_nc = os.path.join(folder, each_mat_NC)
        data_new = getvalue(mat_nc)
        data.append(data_new)
        label.append(0)

    for each_mat_JME in path_JME:
        mat_jme = os.path.join(folder2, each_mat_JME)
        data_new = getvalue(mat_jme)
        data.append(data_new)
        label.append(1)

    C_data = np.array(data)
    C_label = np.array(label)

    # 切分数据集
    x_train, x_test, y_train, y_test = train_test_split(C_data, C_label, test_size=0.18)
    pca = PCA(n_components=n, svd_solver='auto').fit(x_train)  # 要求维数相等
    # 降维
    x_train_pca = pca.transform(x_train)
    x_test_pca = pca.transform(x_test)

    model = SVC(kernel=kenel, probability=True)
    # model = KNeighborsClassifier(n_neighbors=kenel)
    # model = RandomForestClassifier(n_estimators=kenel)
    model.fit(x_train_pca, y_train)

    # 测试识别准确度
    acc = model.score(x_test_pca, y_test)
    ACC_list.append(acc)

    pre_proba = model.predict_proba(x_test_pca)[:, 1]  # 分类概率
    fpr, tpr, thresholds = roc_curve(y_test, pre_proba, pos_label=1)
    auc_temp = auc(fpr, tpr)
    AUC_list.append(auc_temp)

    metrix = CM(y_test, model.predict(x_test_pca), labels=[1, 0])
    TP_50 = metrix[0][0]
    FN_50 = metrix[0][1]
    FP_50 = metrix[1][0]
    TN_50 = metrix[1][1]
    sensitivity = TP_50 / (TP_50 + FN_50)
    sensitivity_list.append(sensitivity)
    specifity = TN_50 / (TN_50 + FP_50)
    specifity_list.append(specifity)

    probability = pre_proba * 100
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    number = 0
    for j in range(100):  # 人为设置threshold为1-100中的每个整数
        for k in range(len(y_test)):
            if probability[k] >= j:  # 预测为1
                if y_test[k] == 1:  # 真实为1
                    TP = TP + 1
                elif y_test[k] == 0:  # 真实为0
                    FP = FP + 1
            elif probability[k] < j:  # 预测为0
                if y_test[k] == 1:  # 真实为1
                    FN = FN + 1
                elif y_test[k] == 0:  # 真实为0
                    TN = TN + 1

        all_TP[m][number] = TP
        all_TN[m][number] = TN
        all_FP[m][number] = FP
        all_FN[m][number] = FN
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        number = number + 1


def saveresult(num, PCA_n, kenel, dizhi2):
    result_path = 'D:/python/mlJNE/MCI/result/' + dizhi2 + '/'
    sum_TP = np.zeros(100, dtype=int)
    sum_TN = np.zeros(100, dtype=int)
    sum_FP = np.zeros(100, dtype=int)
    sum_FN = np.zeros(100, dtype=int)

    for j in range(100):
        sum_TP[j] = sum(all_TP[m][j] for m in range(num))
        sum_FP[j] = sum(all_FP[m][j] for m in range(num))
        sum_TN[j] = sum(all_TN[m][j] for m in range(num))
        sum_FN[j] = sum(all_FN[m][j] for m in range(num))

    ave_FPR = sum_FP / (sum_FP + sum_TN)
    ave_TPR = sum_TP / (sum_TP + sum_FN)

    # fig_average = plt.figure()  # 建立可视化图像框
    # ax1 = fig_average.add_subplot(1, 1, 1)  # z子图总行数、列数，位置
    # ax1.set_xlim(0, 1)
    # ax1.set_ylim(0, 1)
    # ax1.plot(ave_FPR, ave_TPR)  # ROC曲线的横轴为假正例率(FPR), 纵轴为真正例率(TPR)
    # ax1.plot([0, 1], [0, 1], linestyle='--', lw=1, color='blue', label='ROC=0.5', alpha=.8)
    # plt.show()

    workbook = xlwt.Workbook(encoding='utf-8')
    booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)
    workbook_path = os.path.join(result_path, ('SVM_' + str(kenel) + '_PCA_' + str(PCA_n) + '.xls'))

    booksheet.write(0, 0, 'ACC')
    booksheet.write(0, 1, 'AUC')
    booksheet.write(0, 2, 'Sen')
    booksheet.write(0, 3, 'Spe')
    booksheet.write(0, 5, 'FPR')
    booksheet.write(0, 6, 'TPR')

    booksheet.write(0, 9, 'Ave_ACC')
    booksheet.write(0, 10, 'Std_ACC')
    booksheet.write(0, 11, 'Sem_ACC')
    booksheet.write(0, 12, 'Ave_Sen')
    booksheet.write(0, 13, 'Ave_Spe')
    booksheet.write(0, 14, 'Ave_AUC')
    Ave_ACC = round(np.mean(ACC_list), 3)
    Std_ACC = round(np.std(ACC_list), 4)
    Sem_ACC = round(np.std(ACC_list) / math.sqrt(repeat), 4)
    Ave_Sen = round(np.mean(sensitivity_list), 3)
    Ave_Spe = round(np.mean(specifity_list), 3)
    Ave_AUC = round(np.mean(AUC_list), 3)
    booksheet.write(1, 9, Ave_ACC)
    booksheet.write(1, 10, Std_ACC)
    booksheet.write(1, 11, Sem_ACC)
    booksheet.write(1, 12, Ave_Sen)
    booksheet.write(1, 13, Ave_Spe)
    booksheet.write(1, 14, Ave_AUC)

    for i in range(num):
        booksheet.write(i+1, 0, AUC_list[i])  # 前两个参数表示位置，第三个参数表示值
        booksheet.write(i+1, 1, ACC_list[i])
        booksheet.write(i+1, 2, sensitivity_list[i])
        booksheet.write(i+1, 3, specifity_list[i])

    for i in range(100):
        booksheet.write(i+1, 5, ave_FPR[i])
        booksheet.write(i+1, 6, ave_TPR[i])

    workbook.save(workbook_path)


# for PCA_n in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]:
for PCA_n in [15]:
    # for kenel in ['linear', 'poly', 'rbf', 'sigmoid']:
    for kenel in ['rbf']:
    # for kenel in [1, 3, 5, 7, 9, 11, 13, 15]:
    # for kenel in [10, 30, 50, 70, 90, 110, 130, 150]:
        # for dizhi in ['adc/all', 'fa/all']:
        for dizhi in ['adc/all']:
            print(dizhi)
            print(PCA_n)
            print(kenel)
            data = []
            label = []
            ACC_list = []
            AUC_list = []
            sensitivity_list = []
            specifity_list = []

            for i in range(repeat):  # 重复实验次数
                if i % 10 == 0:
                    print('已完成 %d 次' % i)

                getresult(PCA_n, i, kenel, dizhi)
                data = []
                label = []

            saveresult(repeat, PCA_n, kenel, dizhi)

            print("重复100次的平均准确率")
            print(round(np.mean(ACC_list), 3))
            print(round(np.std(ACC_list), 4))
            print("sen,spe,AUC")
            print(round(np.mean(sensitivity_list), 3))
            print(round(np.mean(specifity_list), 3))
            print(round(np.mean(AUC_list), 3))






from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from time import time
import datetime
import time
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

# cancers = load_breast_cancer()
# X = cancers.data       #获取特征值
# Y = cancers.target     #获取标签
# print(X.shape)         #查看特征形状
# print(Y.shape)         #查看标签形状
# print(X)
# print(Y)
#
# np.set_printoptions(suppress=True)
# np.set_printoptions(precision=4)   #设精度
# np.savetxt('X.txt', X, fmt='%.03f')
# np.savetxt('Y.txt', Y, fmt='%.00f')

X = np.loadtxt('X_new.txt')
Y = np.loadtxt('Y_new.txt')

x_train_0, x_test_0, y_train_0, y_test_0 = train_test_split(X, Y, test_size=0.1)
x_train_1, x_test_1, y_train_1, y_test_1 = train_test_split(X, Y, test_size=0.1)
# x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(X, Y, test_size=0.1)
# x_train_3, x_test_3, y_train_3, y_test_3 = train_test_split(X, Y, test_size=0.1)
# x_train_4, x_test_4, y_train_4, y_test_4 = train_test_split(X, Y, test_size=0.1)
# x_train_5, x_test_5, y_train_5, y_test_5 = train_test_split(X, Y, test_size=0.1)


time_start = time.time()

model_linear_0 = SVC(C=1.5, kernel='linear')  # 线性核
model_linear_1 = SVC(C=1.1, kernel='linear')  # 线性核
# model_linear_2 = SVC(C=1.2, kernel='linear')  # 线性核
# model_linear_3 = SVC(C=1.3, kernel='linear')  # 线性核
# model_linear_4 = SVC(C=1.4, kernel='linear')  # 线性核
# model_linear_5 = SVC(C=1.3, kernel='linear')  # 线性核

test_predict_label_0 = model_linear_0.fit(x_train_0, y_train_0).decision_function(x_test_0)
test_predict_label_1 = model_linear_1.fit(x_train_1, y_train_1).decision_function(x_test_1)
# test_predict_label_2 = model_linear_2.fit(x_train_2, y_train_2).decision_function(x_test_2)
# test_predict_label_3 = model_linear_3.fit(x_train_3, y_train_3).decision_function(x_test_3)
# test_predict_label_4 = model_linear_4.fit(x_train_4, y_train_4).decision_function(x_test_4)
# test_predict_label_5 = model_linear_5.fit(x_train_5, y_train_5).decision_function(x_test_5)

train_score_0 = model_linear_0.score(x_train_0, y_train_0)
test_score_0 = model_linear_0.score(x_test_0, y_test_0)
print('test_score_0:', test_score_0)
train_score_1 = model_linear_1.score(x_train_1, y_train_1)
test_score_1 = model_linear_1.score(x_test_1, y_test_1)
# train_score_2 = model_linear_2.score(x_train_2, y_train_2)
# test_score_2 = model_linear_2.score(x_test_2, y_test_2)
# train_score_3 = model_linear_3.score(x_train_3, y_train_3)
# test_score_3 = model_linear_3.score(x_test_3, y_test_3)
# train_score_4 = model_linear_4.score(x_train_4, y_train_4)
# test_score_4 = model_linear_4.score(x_test_4, y_test_4)
# train_score_5 = model_linear_5.score(x_train_5, y_train_5)
# test_score_5 = model_linear_5.score(x_test_5, y_test_5)

# print('train_score:{0}; test_score:{1}'.format(train_score, test_score))
#
# preresult = model_linear.predict(x_test)
# #查看第一个样本的预测值
# print(preresult[0])

fpr_0, tpr_0, threshold_0 = roc_curve(y_test_0, test_predict_label_0)  ###计算真正率和假正率
fpr_1, tpr_1, threshold_1 = roc_curve(y_test_1, test_predict_label_1)  ###计算真正率和假正率
# fpr_2, tpr_2, threshold_2 = roc_curve(y_test_2, test_predict_label_2)  ###计算真正率和假正率
# fpr_3, tpr_3, threshold_3 = roc_curve(y_test_3, test_predict_label_3)  ###计算真正率和假正率
# fpr_4, tpr_4, threshold_4 = roc_curve(y_test_4, test_predict_label_4)  ###计算真正率和假正率
# fpr_5, tpr_5, threshold_5 = roc_curve(y_test_5, test_predict_label_5)  ###计算真正率和假正率
# print(fpr)
# print(tpr)
# print(threshold)
roc_auc_0 = auc(fpr_0, tpr_0)  ###计算auc的值，auc就是曲线包围的面积，越大越好
roc_auc_1 = auc(fpr_1, tpr_1)  ###计算auc的值，auc就是曲线包围的面积，越大越好
# roc_auc_2 = auc(fpr_2, tpr_2)  ###计算auc的值，auc就是曲线包围的面积，越大越好
# roc_auc_3 = auc(fpr_3, tpr_3)  ###计算auc的值，auc就是曲线包围的面积，越大越好
# roc_auc_4 = auc(fpr_4, tpr_4)  ###计算auc的值，auc就是曲线包围的面积，越大越好
# roc_auc_5 = auc(fpr_5, tpr_5)  ###计算auc的值，auc就是曲线包围的面积，越大越好

plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr_0, tpr_0, color='red',
         lw=lw, label='Ours (area = %0.2f)' % roc_auc_0)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr_1, tpr_1, color='cyan',
         lw=lw, label='ResNet (area = %0.2f)' % roc_auc_1)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr_2, tpr_2, color='blue',
#          lw=lw, label='NB (area = %0.2f)' % roc_auc_2)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr_3, tpr_3, color='green',
#          lw=lw, label='DT (area = %0.2f)' % roc_auc_3)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr_4, tpr_4, color='magenta',
#          lw=lw, label='KNN (area = %0.2f)' % roc_auc_4)  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr_5, tpr_5, color='yellow',
#          lw=lw, label='RF (area = %0.2f)' % roc_auc_5)  ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 24,
}
plt.xlabel('False Positive Rate', font1)
plt.ylabel('True Positive Rate', font1)
# plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right", prop=font1)
plt.show()

time_end=time.time()
# print('time cost', time_end-time_start, 's')

########################################################################
########################################################################
########################################################################

import numpy as np
from PIL import Image
from sklearn import svm

def test_LinearSVC(train,label,testdata):
    # 选择模型
    cls = svm.LinearSVC()
    # 把数据交给模型训练
    cls.fit(train,label.astype('int'))

    # 预测数据
    #print(cls.predict(testdata))
    results=cls.predict(testdata)
    return results

#输出混淆矩阵
from sklearn.metrics import confusion_matrix
def my_confusion_matrix(y_true, y_pred):
    labels = list(set(y_true))
    conf_mat = confusion_matrix(list(y_true), list(y_pred), labels = labels)
    print ("confusion_matrix(left labels: y_true, up labels: y_pred):")
    print ("labels"," ",end='')
    for i in range(len(labels)):
        print (labels[i]," ",end='')
    print('\n')
    for i in range(len(conf_mat)):
        print (i," ",end='')
        for j in range(len(conf_mat[i])):
            print (conf_mat[i][j]," ",end='')
        print('\n')
    print


if __name__=="__main__":

    #测试集分类结果---列表
    y_pred = test_LinearSVC(x_train_0,y_train_0,x_test_0)
    label_list = [int(i) for i in y_test_0]

    #测试集合的精度
    kk = 0
    for i in range(len(y_test_0)):
        if (y_pred[i] == y_test_0[i]):
            kk = kk + 1
    print('ACC of SVM %f' % float(kk / len(y_test_0)))
    a = list(y_test_0[:40])
    b = list(y_pred[:40])
    confusion_matrix(y_test_0, y_pred) # 输出混淆矩阵
    # print('y_pred', y_pred)
    # print('y_test', y_test)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt

guess = y_pred
fact = y_test_0
classes = list(set(fact))
classes.sort()
confusion = confusion_matrix(guess, fact)
plt.imshow(confusion, cmap=plt.cm.Blues)
indices = range(len(confusion))
plt.xticks(indices, classes)
plt.yticks(indices, classes)
plt.colorbar()
plt.xlabel('prediction')
plt.ylabel('groundtruth')
for first_index in range(len(confusion)):
    for second_index in range(len(confusion[first_index])):
        plt.text(first_index, second_index, confusion[first_index][second_index])

# plt.show()



############ DecisionTree ############

from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0)
estimator = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
estimator.fit(X_train, y_train)

# print('estimator.predict', estimator.predict(X_test))
print('ACC of DecisionTree:', estimator.score(X_test,y_test))
#
#
# ############ MLP ############
#
# from sklearn.neural_network import MLPClassifier
# from sklearn import datasets
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import matplotlib.pyplot as plt
#
# # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
#
# # 测试集，画图对预测值和实际值进行比较
# def test_validate(x_test, y_test, y_predict, classifier):
#     x = range(len(y_test))
#     plt.plot(x, y_test, "ro", markersize=5, zorder=3, label=u"true_v")
#     plt.plot(x, y_predict, "go", markersize=8, zorder=2, label=u"predict_v,$R$=%.3f" % classifier.score(x_test, y_test))
#     plt.legend(loc="upper left")
#     plt.xlabel("number")
#     plt.ylabel("true?")
#     plt.show()
#
# # 神经网络数字分类
# def multi_class_nn():
#     # digits = datasets.load_digits()
#     # x = digits['data']
#     # y = digits['target']
#
#     # 对数据的训练集进行标准化
#     ss = StandardScaler()
#     # x_regular = ss.fit_transform(x)
#     # 划分训练集与测试集
#     x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)
#     clf.fit(x_train, y_train)
#     # 模型效果获取
#     r = clf.score(x_train, y_train)
#     print("ACC of MLP", r)
#     # 预测
#     y_predict = clf.predict(x_test)  # 预测
#     # print(y_predict)
#     # print(y_test)
#     # 绘制测试集结果验证
#     test_validate(x_test=x_test, y_test=y_test, y_predict=y_predict, classifier=clf)
#
# multi_class_nn()


'''
特征工程
1、特征使用方案
2、特征获取方案
3、特征处理
（1）:数据预处理
（2）:特征选择
（3）:数据降维
'''
import numpy as np
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')
# 拉格朗二次插值函数
def Ln2(x, x0, y0, x1, y1, x2, y2):
    f0 = ((x - x1) * (x - x2)) / ((x0 - x1) * (x0 - x2))
    f1 = ((x - x0) * (x - x2)) / ((x1 - x0) * (x1 - x2))
    f2 = ((x - x0) * (x - x1)) / ((x2 - x0) * (x2 - x1))
    return y0 * f0 + y1 * f1 + y2 * f2
# 非线性插值处理连续性缺省值
def interpolation_nan():
    import matplotlib.pyplot as plt
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    xs = np.linspace(0, 2 * np.pi, 100)
    ys = 2 * np.sin(xs) + 0.1
    new_ys = []
    ys[30], ys[31] = np.nan, np.nan  # 假设有两项缺省值
    x0, y0, x1, y1, x2, y2 = xs[29], ys[29], xs[32], ys[32], xs[33], ys[33]
    new_ys.append(y0)
    new_ys.append(Ln2(xs[30], x0, y0, x1, y1, x2, y2))  # 拉格朗二次插值
    new_ys.append(Ln2(xs[31], x0, y0, x1, y1, x2, y2))
    new_ys.append(y1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, color='#AEEEEE', label='原始数据')
    ax.plot([xs[29], xs[30], xs[31], xs[32]], new_ys, color='#FF3030', label='二次插值数据')
    plt.legend()
    plt.show()
if __name__ == '__main__':
    iris = load_iris()
    X, y = iris.data, iris.target
    '''
    1、数据预处理
    '''
    '''
    1.1：无量纲化：
    使不同规格的数据转换到同一规格
    '''
    '''
    1.1.1：标准化：计算特征的均值和标准差
    公式：x = (x - mean) / s 
    mean：均值(列)  s：标准差(列)
    '''
    # from sklearn.preprocessing import StandardScaler
    # new_X = StandardScaler().fit_transform(X)  # 返回标准正态分布N(0,1)
    # ret = np.zeros(X.shape)
    # for i, col in enumerate(X.T):  # 遍历每一列特征
    #     col_mean = np.mean(col)  # 每列特征的均值
    #     col_s = np.sqrt(np.var(col))  # 每列特征标准差
    #     ret[:, i] = (col - col_mean) / col_s  # 标准化
    '''
    1.1.2：区间缩放法：利用两个最值进行缩放
    公式：x = (x - xmin) / (xmax - xmin)
    xmin：最小值(列) xmax：最大值(列)
    '''
    # from sklearn.preprocessing import MinMaxScaler  # 返回[0, 1]区间
    # new_X = MinMaxScaler().fit_transform(X)
    # ret = np.zeros(X.shape)
    # X_min, X_max = np.min(X, axis=0), np.max(X, axis=0)
    # for i, row in enumerate(X):  # 遍历每一行
    #     ret[i] = (row - X_min) / (X_max - X_min)  # 归一化
    '''
    1.2、对定量特征二值化：
    特征值大于阈值为1，否则为0
    '''
    # from sklearn.preprocessing import Binarizer
    # new_X = Binarizer(threshold=3).fit_transform(X)
    # ret = np.zeros(X.shape)
    # for i, row in enumerate(X):  # 遍历每一行
    #     ret[i] = [1 if t else 0 for t in row > 3]
    '''
    1.3、对定性特征哑编码：
    OneHot编码
    '''
    # from sklearn.preprocessing import OneHotEncoder
    # y = y.reshape(-1, 1)
    # new_y = OneHotEncoder(categories='auto').fit_transform(y).toarray()
    # cls = list(set(y[:, 0].tolist()))
    # ret = np.zeros((y.shape[0], len(cls)))
    # for i, row in enumerate(y):  # 遍历每一行
    #     ret[i] = np.array([1 if code == row[0] else 0 for code in cls])
    '''
    1.4、缺失值计算
    '''
    '''
    1.4.1：均值补全
    通过计算缺省值所对应特征的所有特征值的均值/中位数来补全缺省值
    '''
    # from sklearn.preprocessing import Imputer
    # nan_X = np.vstack((X, np.full_like(np.zeros(4), np.nan)))  # 增加一行特征均为缺失值的数据
    # new_X = Imputer(missing_values='NaN', strategy='mean').fit_transform(nan_X)
    '''
    1.4.2：非线性插值补全
    通过选取缺省值的邻域点，采用非线性插值补全缺省值
    '''
    # interpolation_nan()
    '''
    2、特征选择
    （1）特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，则这个特征对于样本的区分并没有什么用。
    （2）特征与目标的相关性：这点比较显见，与目标相关性高的特征，应当优选选择。
    包含：Filter过滤法、Wrapper包装法、Embedded集成法
    '''
    '''
    2.1：Filter过滤方法：
    按照发散性或者相关性对各个特征进行评分'''
    '''
    2.1.1：方差选择法
    计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征
    '''
    # from sklearn.feature_selection import VarianceThreshold
    # new_X = VarianceThreshold(threshold=3).fit_transform(X)
    # ret = []
    # for i, col in enumerate(X.T):  # 遍历每一列特征
    #     col_var = np.var(col)  # 每列特征方差
    #     if col_var > 3:  # 如果特征方差大于阈值
    #         ret.append(X[:, i])  # 选择该特征对应的数据
    # ret = np.array(ret).T
    '''
    2.1.2：相关系数法
    计算各个特征对模型训练目标值的相关系数，然后根据阈值，选择相关系数大于阈值的特征
    '''
    # from sklearn.feature_selection import SelectKBest, f_regression
    # new_X = SelectKBest(f_regression, k=2).fit_transform(X, y)
    # # 计算x和y的有偏协方差
    # def calc_cov(x, y):
    #     x_mean, y_mean = np.mean(x), np.mean(y)  # x和y均值
    #     cov_xy = np.sum([(xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y)])  # 中心化去均值
    #     return 1 / (len(x) - 1) * cov_xy  # x关于y的有偏协方差
    # # 计算x和y的相关系数
    # def calc_pearsonr(x, y):
    #     N_1 = (len(x) - 1) / len(x)
    #     cov_xy = calc_cov(x=x, y=y) * N_1  # x关于y的无偏协方差
    #     Sx = np.sqrt(calc_cov(x=x, y=x) * N_1)  # x的无偏标准差
    #     Sy = np.sqrt(calc_cov(x=y, y=y) * N_1)  # y的无偏标准差
    #     return cov_xy / (Sx * Sy)  # x关于y的相关系数
    # ret, all_col_pearsonr, k = [], [], 2
    # for i, col in enumerate(X.T):  # 遍历每一列特征
    #     col_pearsonr = calc_pearsonr(col, iris.target)  # 每特征列于y的相关系数
    #     all_col_pearsonr.append((col_pearsonr, i))  # 保存每个特征对于y的相关系数
    # # 对相关系数排序，选出前k个大的相关系数对于的特征列索引
    # sort_key_arr = [v[1] for v in sorted(all_col_pearsonr, key=lambda x: x[0], reverse=True)][0:k]
    # # 选出相关系数最大的前k个特征列数据
    # for i, col in enumerate(X.T):  # 遍历每一列特征
    #     if i in sort_key_arr:
    #         ret.append(X[:, i])
    # ret = np.array(ret).T
    # # 比较API和手动实现特征选择算法的筛选结果是否一致
    # print(np.allclose(new_X, ret))
    '''
    2.1.3：卡方检验
    考虑自变量等于i且因变量等于j的样本频数的观察值与期望的差距
    '''
    # from sklearn.feature_selection import SelectKBest, chi2
    # new_X = SelectKBest(chi2, k=2).fit_transform(X, y)
    '''
    2.2：Wrapper包装法：
    根据目标函数，每次选择若干特征，或者排除若干特征'''
    '''
    2.2.1：RFE递归特征消除法
    指定一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练
    '''
    # from sklearn.feature_selection import RFE
    # from sklearn.linear_model import LogisticRegression
    # new_X = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(X, y)
    '''
    2.3：Embedded集成法：
    先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征
    '''
    '''
    2.3.1：基于惩罚项的特征选择法
    结合带L1惩罚项的逻辑回归LR模型
    '''
    # from sklearn.feature_selection import SelectFromModel
    # from sklearn.linear_model import LogisticRegression
    # new_X = SelectFromModel(LogisticRegression(penalty='l1', C=0.1)).fit_transform(X, y)
    '''
    2.3.2：基于树模型的特征选择法
    结合GBDT模型
    '''
    # from sklearn.feature_selection import SelectFromModel
    # from sklearn.ensemble import GradientBoostingClassifier
    # new_X = SelectFromModel(GradientBoostingClassifier()).fit_transform(X, y)
    '''
    3：降维
    当特征选择完成后，可以直接训练模型了，但是可能由于特征矩阵过大，导致计算量大，训练时间长的问题，因此降低特征矩阵维度也是必不可少的
    '''
    '''
    3.1：主成分分析法（PCA）
    '''
    # from sklearn.decomposition import PCA
    # new_X = PCA(n_components=2).fit_transform(X)
    '''
    3.2：线性判别分析法（LDA）
    '''
    # from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # new_X = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)

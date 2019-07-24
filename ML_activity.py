'''
机器学习的工程步骤：
（1）数据获取：数据集的获取和存储
（2）特征工程：特征工程处理原始数据集
（3）建模：输入处理后的数据集进行模型训练
（4）调优：网格搜索对模型参数调优
（5）持久化模型
'''
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    '''1、数据获取：数据集的获取和存储'''
    iris = load_iris()
    X, y = iris.data, iris.target
    X = np.hstack((X, np.random.choice([0, 1, 2], size=X.shape[0]).reshape(-1, 1)))  # 增加一列特征表示花的颜色
    X = np.vstack((X, np.full_like(np.zeros(X.shape[1]), np.nan)))  # 增加一行样本数据（全为nan）
    y = np.hstack((y, np.array([np.median(y)])))  # 为新增加的样本数据增加一行标签值

    '''2、特征工程：特征工程处理原始数据集'''
    # 缺失值处理
    from sklearn.preprocessing import Imputer
    X = Imputer(missing_values='NaN', strategy='mean').fit_transform(X)
    # 无量纲处理
    from sklearn.preprocessing import StandardScaler
    X = StandardScaler().fit_transform(X)
    # 选择特征
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import GradientBoostingClassifier
    X = SelectFromModel(GradientBoostingClassifier()).fit_transform(X, y)
    # 数据降维
    from sklearn.decomposition import PCA
    X = PCA(n_components=min(2, X.shape[1])).fit_transform(X)

    '''3、建模：输入处理后的数据集进行模型训练'''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=66)  # 随机划分训练集和测试集
    lrmodel = LogisticRegression(penalty='l1', C=10)
    lrmodel.fit(X, y)
    y_pred = lrmodel.predict(X_test)
    print('模型在训练集准确率: {:.2%}'.format(accuracy_score(y_train, lrmodel.predict(X_train))))
    print('模型在测试集准确率: {:.2%}'.format(accuracy_score(y_test, lrmodel.predict(X_test))))

    '''4、调优：网格搜索对模型参数调优'''
    # param_grid = {'C': [0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l1', 'l2']}  # 构建模型参数组合
    # grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=10)
    # grid_search.fit(X_train, y_train)  # 穷举不同参数的模型效果
    # print('最优参数组合: {}'.format(grid_search.best_params_))
    # print('最优参数组合训练集准确率: {:.2%}'.format(grid_search.best_score_))

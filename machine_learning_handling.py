import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import xgboost as xgb
import dashscope
import os
from dashscope import TextEmbedding  # 添加明确导入
import threading
from util.threads import ResultThread
from util.embeddings import get_text_embedding
import time

# 随机森林模型加载
def random_forest(x: pd.DataFrame, y: pd.DataFrame) :
    # 实例化模型
    clf_RF = RandomForestClassifier(n_estimators=100, random_state=22)
    # 在训练集上进行训练
    clf_RF.fit(x,y)
    return  clf_RF

# LGBM模型加载
def random_forest_LGBM(x: pd.DataFrame, y: pd.DataFrame) :
    clf_LGBM = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    # 模型训练
    clf_LGBM.fit(x, y)
    return  clf_LGBM

# XGB模型加载
def random_forest_XGB(x: pd.DataFrame, y: pd.DataFrame) :
    clf_XGB = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    # 模型训练
    clf_XGB.fit(x, y-1)
    return  clf_XGB


# 初始化ML数据
def init_ML_data(df: pd.DataFrame):
    embedding_train = pd.DataFrame(df['embedding'].tolist())
    x_train = embedding_train
    y_train = df['type'].values.ravel()

    return x_train, y_train

# 模型训练
def train_model(x_train: pd.DataFrame, y_train: pd.DataFrame,name: str):
    if name == 'RF':
        clf_RF = random_forest(x_train, y_train)
        return clf_RF
    elif name == 'LGBM':
        clf_LGBM = random_forest_LGBM(x_train, y_train)
        return  clf_LGBM
    elif name == 'XGB':
        clf_XGB = random_forest_XGB(x_train, y_train)
        return  clf_XGB

# 模型预测
def machine_learning_prediction(x_test_set: pd.DataFrame,clf,model_name: str):
    if model_name == 'RF':
        pred_RF = clf.predict(x_test_set)
        return pred_RF
    elif model_name == 'LGBM':
        pred_LGBM = clf.predict(x_test_set)
        return pred_LGBM
    elif model_name == 'XGB':
        pred_XGB = clf.predict(x_test_set) + 1
        return pred_XGB

# 模型训练线程
def machine_learning_training(x_train: pd.DataFrame, y_train: pd.DataFrame):
    # 创建带返回值的线程实例
    thread1 = ResultThread(target=train_model, args=(x_train, y_train,'RF'))
    thread2 = ResultThread(target=train_model, args=(x_train, y_train,'LGBM'))
    thread3 = ResultThread(target=train_model, args=(x_train, y_train,'XGB'))

    # 启动线程
    thread1.start()
    thread2.start()
    thread3.start()

    # 等待所有线程完成
    thread1.join()
    thread2.join()
    thread3.join()

    # 获取线程返回值
    clf_RF = thread1.result
    clf_LGBM = thread2.result
    clf_XGB = thread3.result
    return clf_RF, clf_LGBM, clf_XGB

# 模型预测线程
def machine_learning_prediction_thread(x_train: pd.DataFrame,clf_RF,clf_LGBM,clf_XGB) -> pd.DataFrame:
    thread1 = ResultThread(target=machine_learning_prediction, args=(x_train,clf_RF, 'RF'))
    thread2 = ResultThread(target=machine_learning_prediction, args=(x_train,clf_LGBM, 'LGBM'))
    thread3 = ResultThread(target=machine_learning_prediction, args=(x_train,clf_XGB, 'XGB'))

    thread1.start()
    thread2.start()
    thread3.start()

    # 等待所有线程完成
    thread1.join()
    thread2.join()
    thread3.join()

    # 获取线程返回值
    pred_RF = thread1.result
    pred_LGBM = thread2.result
    pred_XGB = thread3.result
    return pd.DataFrame({'RF': pred_RF, 'LGBM': pred_LGBM, 'XGB': pred_XGB})

def main():
    #数据准备，划分测试集和数据集
    combined_df = pd.read_csv('./data/combined_dataset.csv')
    combined_df["embedding"] = combined_df.Conversation.apply(lambda x: get_text_embedding(x))

    # 直接对整个DataFrame进行划分
    train_df, test_df = train_test_split(
        combined_df,
        test_size=0.2,  # 测试集占20%
        random_state=42,  # 固定随机种子，确保结果可复现
        stratify=combined_df['type']  # 按目标变量分层抽样（分类任务）
    )
    X_train, Y_train = init_ML_data(train_df)
    X_test, Y_test = init_ML_data(test_df)

    print("训练集特征形状:", X_train.shape)
    print("测试集特征形状:", X_test.shape)
    print("训练集目标变量形状:", Y_train.shape)
    print("测试集目标变量形状:", Y_test.shape)
    # X_train[0]
    # y_train.head()
    # 模型训练
    clf_RF, clf_LGBM, clf_XGB = machine_learning_training(X_train, Y_train)

    # 模型预测
    pred_df = machine_learning_prediction_thread(X_test,clf_RF,clf_LGBM,clf_XGB)

    print("pred_df  head = ",pred_df.head())
    print("pred_df  shape = ",pred_df.shape)
    print("test_df  shape = ",test_df.shape)
    #axis=1 表示横向合并（按列拼接），默认 axis=0 是纵向合并（按行拼接）
    # merged_df = pd.concat([test_df,pred_df], axis=1)

    merged_df = pd.concat([
        test_df.reset_index(drop=True),
        pred_df.reset_index(drop=True)
    ], axis=1)
    print(merged_df.head())
    merged_df.to_csv('./data/test_dataset_with_pred.csv',index=False)
    #
    # clf_RF = random_forest(X_train, Y_train)
    # test_preds = clf_RF.predict(X_test)
    # test_accuracy_RF = accuracy_score(Y_test, test_preds)
    # print(f"Test-Accuracy: {test_accuracy_RF}")
    #
    # clf_LGBM = random_forest_LGBM(X_train, Y_train)
    # test_preds = clf_LGBM.predict(X_test)
    # test_accuracy_clf = accuracy_score(Y_test, test_preds)
    # print(f"Test-Accuracy: {test_accuracy_clf}")
    #
    # clf_XGB = random_forest_XGB(X_train, Y_train)
    # test_preds = clf_XGB.predict(X_test) + 1
    # test_accuracy_XGB = accuracy_score(Y_test, test_preds)
    # print(f"Test-Accuracy: {test_accuracy_XGB}")



if __name__ == '__main__':
    main()

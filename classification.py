import pickle

import numpy as np
import lightgbm as lgb
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


class Classification():
    '''
        data explanation
        data : sepal length, sepal width, petal length, petal width  (cm)
        label : 0 - setosa  1 - versicolor  2 - virginica
    '''

    def __init__(self):
        iris = load_iris()
        self.iris_data = iris.data
        self.iris_label = iris.target

    def _divide_train_test(self):
        '''
        데이터를 트레이닝 테스트 데이터로 나눠서 반환
        책에서는 train_test_split random_state=11의 파라미터를 주나 코드에서는 제거
        :return:
        '''
        x_train, x_test, y_train, y_test = train_test_split(self.iris_data,self.iris_label,
                                                            test_size=0.2)
        return x_train, x_test, y_train, y_test

    def dt_model_train(self):
        '''
        데이터 트레이닝 및 예측 진행
        :parameter model_save : 모델을 반환받기를 원할경우 True 입력
        :return:
        '''
        print("DecisionTreeClassifier train")
        # 데이터 트레이닝
        x_train, x_test, y_train, y_test = self._divide_train_test()
        dt_clt = DecisionTreeClassifier()
        dt_clt.fit(x_train,y_train)
        # 예측
        pred = dt_clt.predict(x_test)
        print("모델 정확도 : {0:.4f}".format(accuracy_score(y_test,pred)))

    def lr_model_train(self):
        '''
        데이터 트레이닝 및 예측 진행
        :parameter model_save : 모델을 반환받기를 원할경우 True 입력
        :return:
        '''
        print("LogisticRegression train")
        # 데이터 트레이닝
        x_train, x_test, y_train, y_test = self._divide_train_test()
        lr_clt = LogisticRegression()
        lr_clt.fit(x_train,y_train)
        # 예측
        pred = lr_clt.predict(x_test)
        print("모델 정확도 : {0:.4f}".format(accuracy_score(y_test,pred)))

    def lgbm_model_train(self):
        '''
        데이터 트레이닝 및 예측 진행
        :parameter model_save : 모델을 반환받기를 원할경우 True 입력
        :return:
        '''
        print("Light gbm train")
        # 데이터 트레이닝
        x_train, x_test, y_train, y_test = self._divide_train_test()
        # 학습 모델
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': {'multi_logloss'},
            'num_leaves': 63,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.9,
            'bagging_freq': 0,
            'verbose': 0,
            'num_class': 3
        }
        # create dataset for lightgbm
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)
        lgb_model = lgb.train(params, lgb_train, num_boost_round=20,
                              valid_sets=lgb_test, early_stopping_rounds=5)
        # 예측
        pred = lgb_model.predict(x_test)
        pred = [pr.argmax() for pr in pred]
        print("모델 정확도 : {0:.4f}".format(accuracy_score(y_test, pred)))


if __name__ == "__main__":
    classification = Classification()
    # DecisionTreeClassifier
    # classification.dt_model_train()
    # LogisticRegression
    # classification.lr_model_train()
    # light gbm
    classification.lgbm_model_train()
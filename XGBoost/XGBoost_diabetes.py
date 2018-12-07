import pandas as pd
import xgboost as xgb
from  sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot

if __name__ == '__main__':
    data = pd.read_csv('pima-indians-diabetes.csv',header=None)
    x = data[list(range(7))]
    y = data[8]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=1)

    param = {'max_depth':3, 'eta': 0.3, 'objective': 'binary:logistic', 'silent':1}

    data_train = xgb.DMatrix(x_train, label=y_train)
    data_test = xgb.DMatrix(x_test, label=y_test)

    watch_list = [(data_train, 'train'), (data_test, 'test')]

    bst = xgb.train(param, data_train, num_boost_round=10, evals=watch_list)
    y_hat = bst.predict(data_test)
    y_hat = [round(value) for value in y_hat]
    acc = accuracy_score(y_hat, y_test)
    print('精确度为%.4f%%' % (acc*100))

    xgb.plot_importance(bst) # 绘制特征的重要性
    pyplot.show()




import xgboost as xgb
import pandas as pd
import time
from sklearn.decomposition import PCA
import numpy as np


now = time.time()

train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')
pca = PCA(n_components=100,whiten=True)

train = train_set.iloc[:, 1:].values
train = pca.fit_transform(train)
label = train_set.iloc[:, :1].values
test = test_set.iloc[:,:].values
test = pca.fit_transform(test)

param = {
    'objective': 'multi:softmax',
    'num_class': 10,
    'eta': 0.1,
    'silent': 1,
    'max_depth': 6,
    'nthread':4,
}

offset = 35000
num_rounds = 300

data_train = xgb.DMatrix(train[:offset, :], label=label[:offset])
data_test = xgb.DMatrix(test)
data_val = xgb.DMatrix(train[offset:, :], label=label[offset:])
watch_list = [(data_train, 'train'),(data_val, 'val')]

bst = xgb.train(param, data_train,num_boost_round=num_rounds, evals=watch_list, early_stopping_rounds=100)
predict = bst.predict(data_test,ntree_limit=bst.best_iteration)
np.savetxt('submission_xgb_MultiSoftmax.csv',np.c_[range(1,len(test)+1),predict],
                delimiter=',',header='ImageId,Label',comments='',fmt='%d')


cost_time = time.time() - now
print("end ......",'\n',"cost time:",cost_time,"(s)......")



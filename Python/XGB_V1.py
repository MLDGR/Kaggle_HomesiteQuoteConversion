# adapted from:
# https://www.kaggle.com/mpearmain/homesite-quote-conversion/xgboost-benchmark/code

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score

seed = 12345

train = 'train.csv'
test = 'test.csv'

print 'Reading data...'
train = pd.read_csv(train)
test = pd.read_csv(test)

def format_data(d):
    d.drop('QuoteNumber', axis=1, inplace=True)

    # create date features
    d['Date'] = pd.to_datetime(pd.Series(d['Original_Quote_Date']))
    d['Year'] = d['Date'].dt.year
    d['Month'] = d['Date'].dt.month
    d['DayOfWeek'] = d['Date'].dt.dayofweek
    d.drop('Original_Quote_Date', axis=1, inplace=True)
    d.drop('Date', axis=1, inplace=True)

    # fill NaN
    d = d.fillna(-1)
    return d

print 'Formatting data...'
y = np.array(train['QuoteConversion_Flag'])
train.drop('QuoteConversion_Flag', axis=1, inplace=True)
train = format_data(train)

submission = test[['QuoteNumber']]
test = format_data(test)

print 'Creating features...'
features = train.columns
# convert categorical features to numeric
for f in features:
    if train[f].dtype=='object':
        print(f)
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values) + list(test[f].values))
        train[f] = lbl.transform(list(train[f].values))
        test[f] = lbl.transform(list(test[f].values))

# create train/eval
train_X, eval_X, train_y, eval_y = train_test_split(train, y, test_size=.05)

dtrain = xgb.DMatrix(train_X, train_y)
deval = xgb.DMatrix(eval_X, eval_y)

watchlist = [(dtrain, 'train'), (deval, 'eval')]

params = {"objective": "binary:logistic",
          "booster" : "gbtree",
          "eta": 0.08,
          "max_depth": 13,
          "subsample": 0.7,
          "colsample_bytree": 0.7,
          "eval_metric": "auc",
          "silent": 1
}

rounds = 1600

print 'Training model...'
gbm = xgb.train(params, dtrain, rounds, evals=watchlist, early_stopping_rounds=50, verbose_eval=True)

preds = gbm.predict(deval)
score = roc_auc_score(eval_y, preds)
print 'Evaluation set AUC: {0}'.format(score)

print 'Making submission...'
dtest = xgb.DMatrix(test)
submission_preds = gbm.predict(dtest)
submission['QuoteConversion_Flag'] = submission_preds
submission.to_csv('xgb_submission0005.csv', index=False)


# XGB feature importances
#xgb.plot_importance(gbm)
#mpl.pyplot.savefig('foo.png')
x = pd.Series(gbm.get_fscore())
x.to_csv('feature_score5.csv')
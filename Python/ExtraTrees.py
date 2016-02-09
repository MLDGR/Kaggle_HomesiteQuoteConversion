import random
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn import preprocessing

# Reading files 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train_sample = np.random.choice(train.index.values,40000)   
train = train.ix[train_sample]



# Converting date into datetime format
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
# Dropping original date column
train = train.drop('Original_Quote_Date', axis=1)   

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

## Seperating date into 3 columns
train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek 
    
train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)    

## Filing NA values with -1

train = train.fillna(-1)
test = test.fillna(-1)

## Converting categorical variables into numeric variables with label encoder

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl=preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values)+list(test[f].values))
        train[f]=lbl.transform(list(train[f].values))
        test[f]=lbl.transform(list(test[f].values))


y = train.QuoteConversion_Flag.values
train = train.drop(['QuoteNumber', 'QuoteConversion_Flag'], axis=1)
test = test.drop('QuoteNumber', axis=1)          
 
X = train.ix[:, 0:299]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)       
      
extc = ExtraTreesClassifier(n_estimators=580,max_features= 168,criterion= 'entropy',min_samples_split= 3,n_jobs=-1,
                            max_depth= 30, min_samples_leaf= 8)      
extc.fit(X,y)          
        
## Creating submission file

preds = extc.predict_proba(test)[:,1]
sample = pd.read_csv('sample_submission.csv')
sample.QuoteConversion_Flag = preds
sample.to_csv('extc_p_V1.0.csv', index=False)        
      

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "binary:logistic",
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 7, #changed from default of 8
                subsample           = 0.86, # 0.7
                colsample_bytree    = 0.68 # 0.7
                alpha = 0.0001,
                            # lambda = 1
                )
bst.cv = xgb.cv(param=param, data = dtrain,nfold = 4, nrounds=1900)
bst.cv$train.auc.mean




#0.9800338







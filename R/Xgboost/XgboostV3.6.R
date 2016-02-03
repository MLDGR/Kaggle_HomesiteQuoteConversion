library(readr)
library(xgboost)

#my favorite seed^^
set.seed(1718)

cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -1
test[is.na(test)]   <- -1


lista<-c("SalesField5","PersonalField10B","PropertyField37","Field7",
         "SalesField8","GeographicField31B","PropertyField26B")
train<-train[,c("QuoteConversion_Flag",lista)]
test=test[,lista]




feature.names <- names(train)[c(2:dim(train)[2])]
cat("Feature Names\n")
feature.names

cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

nrow(train)
h<-sample(nrow(train),2000)
train$QuoteConversion_Flag<-as.numeric(train$QuoteConversion_Flag)
dval<-xgb.DMatrix(data=data.matrix(train[h,]),label=train$QuoteConversion_Flag[h])
#dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
dtrain<-xgb.DMatrix(data=data.matrix(train),label=train$QuoteConversion_Flag)

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective = "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.15, # 0.06, #0.01,
                max_depth           = 5, #changed from default of 8
                subsample           = 0.83, # 0.7
                colsample_bytree    = 0.77 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 400, 
                    verbose             = 0,  #1
                    #early.stop.round    = 150,
                    #watchlist           = watchlist,
                    maximize            = FALSE
)

bst.cv = xgb.cv(param=param, data = dtrain, nfold = 10, nrounds=400)


test[] <- lapply(test, as.numeric)
pred1 <- predict(clf, (data.matrix(test[,feature.names])))
test  <- read_csv("test.csv")
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb_V3.6.csv")

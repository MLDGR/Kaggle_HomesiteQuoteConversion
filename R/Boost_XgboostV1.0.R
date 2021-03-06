library(readr)
library(xgboost)



cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- 0
test[is.na(test)]   <- 0

cat("train data column names and details\n")
names(train)
str(train)
summary(train)
cat("test data column names and details\n")
names(test)
str(test)
summary(test)


# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$day <- weekdays(as.Date(train$Original_Quote_Date))

# removing the date column
train <- train[,-c(2)]

# seperating out the elements of the date column for the train set
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$day <- weekdays(as.Date(test$Original_Quote_Date))

# removing the date column
test <- test[,-c(2)]


feature.names <- names(train)[c(3:301)]
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

cat("train data column names after slight feature engineering\n")
names(train)
cat("test data column names after slight feature engineering\n")
names(test)
tra2<-train[,feature.names]
listapred=read_csv("sample_submission.csv")
listapred[,2]=0
for(x in 1:20) {
  dat<-sample(nrow(train),26000)
  train2<-train[dat,]
  tra<-tra2[dat,]
  h<-sample(nrow(tra),200)
  dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train2$QuoteConversion_Flag[h])
  dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train2$QuoteConversion_Flag[-h])
  
  watchlist<-list(val=dval,train=dtrain)
  param <- list(  objective           = "binary:logistic", 
                  booster = "gbtree",
                  eta                 = 0.04, # 0.06, #0.01,
                  max_depth           = 7, #changed from default of 8
                  subsample           = 0.6, # 0.7
                  colsample_bytree    = 0.6 # 0.7
                  #num_parallel_tree   = 2
                  # alpha = 0.0001, 
                  # lambda = 1
  )
  
  clf <- xgb.train(   params              = param, 
                      data                = dtrain, 
                      nrounds             = 600, #300, #280, #125, #250, # changed from 300
                      verbose             = 1,
                      early.stop.round    =300,
                      watchlist           = watchlist,
                      maximize            = FALSE
  )
  
  pred1 <- predict(clf, data.matrix(test[,feature.names]))
  listapred[,2]=pred1+listapred[,2]
}   

submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=(listapred[,2]/20)) 
cat("saving the submission file\n")
write_csv(submission, "uBoos_xgboostV1.0.csv",)

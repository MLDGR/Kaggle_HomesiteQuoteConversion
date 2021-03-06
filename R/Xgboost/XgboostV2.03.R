# Based on Ben Hamner script from Springleaf
# https://www.kaggle.com/benhamner/springleaf-marketing-response/random-forest-example

library(readr)
library(xgboost)
library(mice)
library(rms)

#my favorite seed^^
set.seed(172)

cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

train[is.na(train)]   <- 0
test[is.na(test)]   <- 0
#test<-test[,-161]
#test<-test[,-125]



colNumericas<-c()
colNoNumericas<-c()
for(aux in c(1:dim(train)[2])){
  print(aux)
  if(is.numeric(train[,aux])){
    colNumericas<-c(colNumericas,aux)
  }else{
    colNoNumericas<-c(colNoNumericas,aux)   
  }
}

numerTrain<-train[,colNumericas]

# There are some NAs in the integer columns so conversion to zero
imp  <- mice(numerTrain,m = 30)
imp  <- (train)
complete(imp)


colNumericasTest<-c()
colNoNumericasTest<-c()
for(aux in range(1,dim(Test)[2])){
  if(is.numeric(Test[,aux])){
    colNumericasTest<-c(colNumericasTest,aux)
  }else{
    colNoNumericasTest<-c(colNoNumericasTest,aux)   
  }
}

numerTest<-Test[,-colNoNumericasTest]
imptest  <- mice(numerTest)
complete(imptest)
#train<-train[,-162]
#train<-train[,-126] 


#test<-test[,-161]
#test<-test[,-125]


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
tra<-(train[,feature.names])

tra<-train[,feature.names]
for(aux in feature.names){
  print(aux)
  tra[,aux]<-sqrt(train[,aux]+(3/8)+1)
}

nrow(train)
h<-sample(nrow(train),2000)

dval<-xgb.DMatrix(data=data.matrix(tra[h,]),label=train$QuoteConversion_Flag[h])
#dtrain<-xgb.DMatrix(data=data.matrix(tra[-h,]),label=train$QuoteConversion_Flag[-h])
dtrain<-xgb.DMatrix(data=data.matrix(tra),label=train$QuoteConversion_Flag)

watchlist<-list(val=dval,train=dtrain)
param <- list(  objective           = "binary:logistic", 
                booster = "gbtree",
                eval_metric = "auc",
                eta                 = 0.02, # 0.06, #0.01,
                max_depth           = 7, #changed from default of 8
                subsample           = 0.86, # 0.7
                colsample_bytree    = 0.68 # 0.7
                #num_parallel_tree   = 2
                # alpha = 0.0001, 
                # lambda = 1
)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = 1900, 
                    verbose             = 0,  #1
                    #early.stop.round    = 150,
                    #watchlist           = watchlist,
                    maximize            = FALSE
)


tes<-test[,feature.names]
for(aux in feature.names){
  print(aux)
  tes[,aux]<-sqrt(test[,aux]+(3/8)+1)
}
pred1 <- predict(clf, data.matrix(tes[,feature.names]))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb_V2.4_1500_5.csv")

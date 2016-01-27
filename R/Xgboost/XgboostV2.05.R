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

for(aux in feature.names){
  lista<-unique(train[,aux])
  listaNum<-lista
  for(i in 1:length(lista)){
    listaNum[i]=sum(train[,aux]==lista[i])
  }
  for(i in length(lista)){
    #pos<-match(lista[i],train[,aux])
    pos<-which(train[,aux] %in% lista)
    tra[pos,aux]<-as.numeric(listaNum[i])/as.numeric(dim(train)[1])
    #print (tra[pos,aux])
  }
  print(aux)
  #tra[,aux] <-log(tra[,aux])-log(1-(tra[,aux]))
  #print (sum((tra[,aux]<0 )))
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
  lista<-unique(test[,aux])
  listaNum<-lista
  for(i in 1:length(lista)){
    listaNum[i]=sum(test[,aux]==lista[i])
  }
  for(i in length(lista)){
    #pos<-match(lista[i],test[,aux])
    pos<-which(test[,aux] %in% lista)
    tes[pos,aux]<-as.numeric(listaNum[i])/as.numeric(dim(test)[1])
    #print (tes[pos,aux])
  }
  print(aux)
  tes[,aux] <-log(tes[,aux])-log(1-(tes[,aux]))
  print (sum(is.na(tes[,aux] )))
  #print (sum((tra[,aux]<0 )))
}
pred1 <- predict(clf, data.matrix(tes[,feature.names]))
submission <- data.frame(QuoteNumber=test$QuoteNumber, QuoteConversion_Flag=pred1)
cat("saving the submission file\n")
write_csv(submission, "xgb_V2.5_1500_5.csv")

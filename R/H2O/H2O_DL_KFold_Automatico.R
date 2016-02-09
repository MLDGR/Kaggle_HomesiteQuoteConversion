library(readr)
library(h2o)


h2o.shutdown( promp = F)
localH2O = h2o.init(nthreads = -1)

cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -1
test[is.na(test)]   <- -1
lista<-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)
train=train[,-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)]


test=test[,-(lista-1)]


# seperating out the elements of the date column for the train set
train$month <- as.integer(format(train$Original_Quote_Date, "%m"))
train$year <- as.integer(format(train$Original_Quote_Date, "%y"))
train$day <- weekdays(as.Date(train$Original_Quote_Date))

# removing the date column
train <- train[,-c(1)]

# seperating out the elements of the date column for the train set
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$day <- weekdays(as.Date(test$Original_Quote_Date))

# removing the date column
test <- test[,-c(1)]

train[is.na(train)]   <- -1
test[is.na(test)]   <- -1

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


trainFull=train
testFull=test

NumFeatures=c(10,30,50)
Lista_ntrees=c(600,1200)
Lista_max_depth=c(3,5,11)
Lista_balance_classes=T#c(T,F)
Lista_min_rows=c(20)
Lista_binomial_double_trees=F#c(T,F)


submit=data.frame(rbind(1:9))
names(submit)<-c("Fearutes","ntree","depth","balance","try","min_row","binomial","train_auc","test_auc")
for(MinNumber in NumFeatures){
  source("ReduccionVariables_Ismael.R")
  train=trainFull[,-name]
  test=testFull[,-(name-1)]
  feature.names <- names(train)[c(2:dim(train)[2])]
  ds_h2o <- as.h2o( train)
  test <- as.h2o(test)
  train<-ds_h2o
  ds_h2o$QuoteConversion_Flag <- as.factor(ds_h2o$QuoteConversion_Flag)
  
  
  s = h2o.runif(ds_h2o, seed=1)
  target_var <- "QuoteConversion_Flag"
  p=dim(train)[2]
  #Lista_mtries=as.integer(c(sqrt(p),p/3))
  #Lista_mtries[Lista_mtries==0]=-1
  Lista_mtries=-1
  for(numArboles in Lista_ntrees){
    for(depth in Lista_max_depth){
      for(bal in Lista_balance_classes){
        for(MinR in Lista_min_rows){
          for(trye in Lista_mtries){
            for(binomial in Lista_binomial_double_trees){
              model <- h2o.deeplearning(
                nfolds = 5,
                x = setdiff(colnames(ds_h2o), target_var),
                y = target_var, 
                training_frame = ds_h2o
              )
              submit<-data.frame(rbind(submit,
                                       c(MinNumber,numArboles,depth,bal,
                                         trye,MinR,binomial,model@model$training_metrics@metrics$AUC,
                                         model@model$cross_validation_metrics@metrics$AUC)))
              cat("Ultimo Resultado es:\n")
              print(submit[dim(submit)[1],])
              model <- h2o.randomForest(ntrees = numArboles, max_depth = depth,balance_classes = bal,mtries = trye ,
                                        ,min_rows = MinR,
                                        binomial_double_trees = binomial,
                                        
              )
              
              
            }
          }
        }
        
      }
    }
    
  }
}

write.csv(submit, "Result_H2O_RF_5fold_PruebaV3_FewFeatures.csv" , row.names = F)

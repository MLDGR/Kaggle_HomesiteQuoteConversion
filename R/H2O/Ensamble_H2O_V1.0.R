  #Importing libraries
library(readr)
library(xgboost)
library(reshape2)
library(stringr)
library(h2o)

#Initialize H2O
h2o.init()

#Variables to ignore in the model calculations
ignore<-c("QuoteNumber","Original_Quote_Date","QuoteConversion_Flag")

#Loading data
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

#Number of unique values
uniquez<-sapply(train,function(x) length(unique(x)))

#Median of unique values
median_value<-median(uniquez)

#Transformation of variable types. If it's a character, transform it to factor. 
#If it isn't a character and the number of unique values in the variable is lower 
#than the median of all variables, then convert it to a factor.
for (a in setdiff(colnames(train),ignore)){
  if (typeof(train[[a]])=="character"){
    levels <- sort(unique(c(train[[a]],test[[a]])))
    train[[a]]<-factor(train[[a]], levels=levels)
    test[[a]]<-factor(test[[a]], levels=levels)
  } else {
    if (length(unique(train[[a]]))<median_value){
      levels <- sort(unique(c(train[[a]],test[[a]])))
      train[[a]]<-factor(train[[a]], levels=levels)
      test[[a]]<-factor(test[[a]], levels=levels)
    }
  }
}

#h2o modeling
train_hex<-as.h2o(train)
test_hex<-as.h2o(test)

#Training set and test set for model building
train_ind<-sample(seq_len(nrow(train_hex)), size = 35000)
train_hex1<-train_hex[train_ind,]
train_hex2<-train_hex[-train_ind,]

#------------------------DEEP LEARNING-------------------------------
train_dl <- h2o.deeplearning(x = setdiff(colnames(train_hex),ignore), nesterov_accelerated_gradient = T,
                             hidden = c(340,240,120),epochs = 65,
                             y = "QuoteConversion_Flag", 
                             training_frame = train_hex1,
                             validation_frame=train_hex2)

#Model performance
h2o.performance(model = train_dl, data=train_hex2)

#Predictions
predictions_dl <- h2o.predict(train_dl, test_hex)

#------------------------RANDOM FOREST------------------------------
train_rf <- h2o.randomForest(x = setdiff(colnames(train_hex),ignore), ntrees = 1200,balance_classes = T,
                             y = "QuoteConversion_Flag", 
                             training_frame = train_hex1,
                             validation_frame=train_hex2,
                             binomial_double_trees = TRUE)

#Model performance
h2o.performance(model = train_rf, data=train_hex2)

#Predictions
predictions_rf <- h2o.predict(train_rf, test_hex)

#----------------GRADIENT BOOSTED MACHINES-------------------
train_gbm <- h2o.gbm(x = setdiff(colnames(train_hex),ignore), ntrees = 700,balance_classes = T,
                     y = "QuoteConversion_Flag", 
                     training_frame = train_hex1,
                     validation_frame=train_hex2)

#Model performance
h2o.performance(model = train_gbm, data=train_hex2)

#Predictions
predictions_gbm <- h2o.predict(train_gbm, test_hex)

#-----------------------ENSEMBLE-----------------------------
#The final values are the means of the three predictions
predictions_ensemble<-h2o.cbind(predictions_dl, 
                                predictions_rf$predict, 
                                predictions_gbm$predict)

predictions_ensemble$QuoteConversion_Flag<-(predictions_ensemble$predict+predictions_ensemble$predict0+predictions_ensemble$predict1)/3

#If it's a negative value, then make it zero, if it's larger than 1, make it 1.
predictions_ensemble$QuoteConversion_Flag<-ifelse(predictions_ensemble$QuoteConversion_Flag<0,
                                                  0,
                                                  ifelse(predictions_ensemble$QuoteConversion_Flag>1,
                                                         1,
                                                         predictions_ensemble$QuoteConversion_Flag))

#Submission
submission<-h2o.cbind(test_hex$QuoteNumber,predictions_ensemble$QuoteConversion_Flag)
h2o.exportFile(submission, "h2o_submission.csv")

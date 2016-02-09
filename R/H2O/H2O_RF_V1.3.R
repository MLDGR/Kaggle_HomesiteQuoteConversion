library(readr)
library(h2o)
num=150
source("ReduccionVariables_Ismael.R")
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


train=train[,-name]
test=test[,-(name-1)]
ds_h2o <- as.h2o( train)
test <- as.h2o(test)
train<-ds_h2o
ds_h2o$QuoteConversion_Flag <- as.factor(ds_h2o$QuoteConversion_Flag)

s = h2o.runif(ds_h2o, seed=1)

#train = h2o.assign(ds_h2o[s<=0.8,],"train")
#valid = h2o.assign(ds_h2o[s>0.2,],"valid")

target_var <- "QuoteConversion_Flag"
model <- h2o.randomForest(ntrees = 1400,max_depth = 15,balance_classes = T,min_rows = 20,
  x = setdiff(colnames(train), target_var),
  y = target_var, 
 # validation_frame = valid,
  training_frame = train,
)

#model@model$validation_metrics
#perf <- h2o.performance(model, data = valid)
#auc <- h2o.auc(perf)
#print(auc) 0.98

prediction <-h2o.predict(object = model, newdata=test)
submit <- data.frame(QuoteNumber=as.data.frame(test$QuoteNumber), as.data.frame(prediction[,1]))
colnames(submit)[2] <- target_var
write.csv(submit, "H2O_RF_v1.3.csv" , row.names = F)


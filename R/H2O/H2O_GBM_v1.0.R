library(h2o)
library(readr)

h2o.shutdown(localH2O, promp = F)
localH2O = h2o.init()


path <- ""
cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

train[is.na(train)]   <- 0
test[is.na(test)]   <- 0
ds_h2o <- as.h2o(localH2O, train)
test <- as.h2o(localH2O, test)

ds_h2o$QuoteConversion_Flag <- as.factor(ds_h2o$QuoteConversion_Flag)

s = h2o.runif(ds_h2o, seed=1)
train = ds_h2o
train = h2o.assign(ds_h2o[s<=0.8,],"train")
valid = h2o.assign(ds_h2o[s>0.2,],"valid")

target_var <- "QuoteConversion_Flag"
model <- h2o.gbm(
  x = setdiff(colnames(train), target_var),
  y = target_var, 
  training_frame = train,
  #validation_frame = valid,
  #ntrees = 100,
  ntrees = 35,
  max_depth = 6,
  nbins = 40,
  distribution = "bernoulli",
  future=F
  
)



kf.gbm <- h2o.kfold(5, train, setdiff(colnames(train), target_var), target_var, h2o_gbm, h2o.predict, T)#TRUE)  # poll future models

model@model$validation_metrics
#perf <- h2o.performance(model, data = valid)
#auc <- h2o.auc(perf)
#print(auc)

prediction <-h2o.predict(object = model, newdata=test)
submit <- data.frame(QuoteNumber=as.data.frame(test$QuoteNumber), as.data.frame(prediction[,3]))
colnames(submit)[2] <- target_var
View(submit)
options(scipen = 999)
write.csv(submit, paste0(path,"polstein_homesite_quote_conversion_v2.csv") , row.names = F)

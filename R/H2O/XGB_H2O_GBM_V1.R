library(h2o)

h2o.shutdown(localH2O, promp = F)
localH2O = h2o.init()


ds_h2o <- h2o.importFile(localH2O, path = "train.csv")
test <- h2o.importFile(localH2O, path="test.csv")
ds_h2o$QuoteConversion_Flag <- as.factor(ds_h2o$QuoteConversion_Flag)

s = h2o.runif(ds_h2o, seed=1)
train = ds_h2o
#train = h2o.assign(ds_h2o[s<=0.8,],"train")
#valid = h2o.assign(ds_h2o[s>0.2,],"valid")

target_var <- "QuoteConversion_Flag"
model <- h2o.gbm(
  x = setdiff(colnames(train), target_var),
  y = target_var, 
  training_frame = train,
  #validation_frame = valid,
  ntrees = 300,
  max_depth = 6,
  nbins = 40,
  distribution = "bernoulli"
  #score_each_iteration = TRUE
)

#model@model$validation_metrics
#perf <- h2o.performance(model, data = valid)
#auc <- h2o.auc(perf)
#print(auc)

prediction <-h2o.predict(object = model, newdata=test)
submit <- data.frame(QuoteNumber=as.data.frame(test$QuoteNumber), as.data.frame(prediction[,3]))
colnames(submit)[2] <- target_var
View(submit)
options(scipen = 999)
write.csv(submit, paste0(path,"polstein_homesite_quote_conversion_v5.csv") , row.names = F)
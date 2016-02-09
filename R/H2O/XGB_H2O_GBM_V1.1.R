library(h2o)

h2o.shutdown( promp = F)
localH2O = h2o.init(nthreads = -1)

train<- read.csv("train.csv")
test<- read.csv("test.csv")

lista<-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)
train=train[,-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)]
test=test[,-(lista-1)]
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


feature.names <- names(train)[c(3:dim(train)[2])]
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

train2<-train
ds_h2o <- as.h2o( train)
test <- as.h2o(test)
train<-ds_h2o

ds_h2o$QuoteConversion_Flag <- as.factor(ds_h2o$QuoteConversion_Flag)

s = h2o.runif(ds_h2o, seed=1)

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
write.csv(submit, "homesite_quote_conversion_v1.1.csv" , row.names = F)

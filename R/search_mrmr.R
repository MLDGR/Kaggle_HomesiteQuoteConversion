library(mRMRe)
library(readr)
library(randomForest)
library(Metrics)
library(caret)

train=read_csv("train.csv")
test=read_csv("test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -1
test[is.na(test)]   <- -1

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

nombre=colnames(train)[2]
train=data.frame(train[,-2],train[,2])
colnames(train)[ncol(train)]=nombre

#Remove features with one value or even with missing values
train=train[,-c(127,132,137,180,190,191,198,206,212,216,282,290,292,294)]
test=test[,-c(127,132,137,180,190,191,198,206,212,216,282,290,292,294)]

#Create partitions with caret
folds=createDataPartition(y=train[[2]],times=1,p=0.8)
ttrain=train[folds[[1]],]
ttest=train[-folds[[1]],]

ddtrain=as.data.frame(apply(train,2,as.numeric))

#mRMR  
dd=mRMR.data(data =  ddtrain)
features=mRMR.ensemble(data = dd, target_indices = 301, solution_count = 1, feature_count = 301) 
features=as.data.frame(print(solutions(features)))

ttrain[[2]]=as.factor(ttrain[[2]])

#Check order of the features of mRMR through h2o.RF
ffs=sapply(2:286,function(i){
  cat(i,colnames(train)[features[i,2]],"\n")
  ttrain=train[folds[[1]],c(features[1:i,2],2)]
  ttest=train[-folds[[1]],features[1:i,2]]
  ttrain=as.h2o(ttrain,destination_frame = "ttrainframe")
  ttest=as.h2o(ttest,destination_frame = "ttestframe")
  model=h2o.randomForest(x=c(1:i),y=ncol(ttrain),
                         training_frame = "ttrainframe",min_rows=20,
                         max_depth = 23, ntrees=200, balance_classes = T)
  
  pred=h2o.predict(object=model,newdata=ttest)
  
  pred=as.data.frame(pred)
  metrica=auc(as.factor(pred[,1]),train[-folds[[1]],2])
  print(metrica)
  return(rbind(i,features[i,1],metrica))
})

#Perform a CV with n-features 
folds=createDataPartition(y=train[[2]],times=5,p=0.8)

cv=lapply(1:5,function(i){
  train=train[folds[[i]],]
  test=train[-folds[[i]],]
  model=randomForest(x=train[,features[1:190,1]],y=train[[2]],ntree=10,do.trace=T)
  pred=predict(model,test[[-2]])
  metrica=auc(pred,test[[2]])
  return(metrica)
})

#Plot result
final=(rbind(cbind(1,features[1,1],NA),t(ffs)))
final=as.data.frame(final)
ggplot(final[,c(1,3)],aes(x=V1,y=V3))+geom_line() 
write.table(ffs,file="caracteristicas.csv",sep = ",",row.names = F)
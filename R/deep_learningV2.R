library(h2o)
library(caret)
library(AUC)
h2o.init(ip="localhost",nthreads=7,max_mem_size = "24G")


Caracteristicas<-read_csv("caracteristicas.csv")

Caracteristicas=Caracteristicas[-nrow(Caracteristicas)]

MinNumber=286
name<-Caracteristicas[c(1:MinNumber),1]


cat("reading the train and test data\n")
train <- read_csv("train.csv")
test  <- read_csv("test.csv")

QuoteNumber=test$QuoteNumber

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -1
test[is.na(test)]   <- -1

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

nombre=colnames(train)[2]
train=data.frame(train[,-2],train[,2])
colnames(train)[ncol(train)]=nombre

train=train[,-c(127,132,137,180,190,191,198,206,212,216,282,290,292,294)]
test=test[,-c(127,132,137,180,190,191,198,206,212,216,282,290,292,294)]

feature.names <- names(train)[c(1:dim(train)[2])]


cat("assuming text variables are categorical & replacing them with numeric ids\n")
for (f in feature.names) {
  if (class(train[[f]])=="character") {
    levels <- unique(c(train[[f]], test[[f]]))
    train[[f]] <- as.integer(factor(train[[f]], levels=levels))
    test[[f]]  <- as.integer(factor(test[[f]],  levels=levels))
  }
}

train=train[,c(name,ncol(train))]
test=test[,name]

nfilas=nrow(train)
etiquetas=train[,ncol(train)]
full=rbind(train[,-ncol(train)],test)


for(i in 1:ncol(full)){
  if(length(unique(full[[i]]))<=50 & is.integer(full[[i]])){
    full[[i]]=as.factor(full[[i]])
  }
}

#Ejecuta esto para asegurarte que los factores siguen el mismo orden en train y en test
train=full[1:nfilas,]
test=full[(nfilas+1):nrow(full),]
train=data.frame(train,etiquetas)
colnames(train)[ncol(train)]=nombre
train[,ncol(train)]=as.factor(train[,ncol(train)])

train_hex=as.h2o(train,destination_frame = "train")
test_hex=as.h2o(test, destination_frame = "test")

l2=0.0008
hidden=c(400)
hidden_dropout_ratios=c(0.5)
rate=0.00002
epochs=400


deep=h2o.deeplearning(x = colnames(train_hex)[-match("QuoteConversion_Flag",colnames(train_hex))],y ="QuoteConversion_Flag",training_frame=train_hex,
                      hidden=c(hidden), #(80,40,20)
                      seed=1234,
                      epochs = epochs, #20
                      adaptive_rate = F,
                      score_validation_samples = 500,
                      score_training_samples = round(nrow(train_hex)*0.25),
                      train_samples_per_iteration= 100,
                      activation="MaxoutWithDropout",
                      momentum_start = 0.2,
                      momentum_stable = 0.99,
                      momentum_ramp = 100,
                      rate =rate, #0.005
                      l2 = l2, #0.002
                      #l1 = l1,
                      hidden_dropout_ratios = hidden_dropout_ratios, #c(0.5,0.5,0.5)
                      #                        initial_weight_distribution="Normal",
                      #                      distribution="gaussian",
                      loss="CrossEntropy",
                      balance_classes=T,
                      fast_mode = T,
                      stopping_rounds = 10,
                      stopping_metric= "MSE",
                      quiet_mode=F,
                      nesterov_accelerated_gradient=T
)

h2o.saveModel(deep,path="deepmodel")

pred.deep=h2o.predict(object = deep,newdata = test_hex)
submit <- data.frame(QuoteNumber=QuoteNumber, as.data.frame(pred.deep[,3]))
colnames(submit)[2] <- target_var
submit[,2]=round(submit[,2],8)
write.csv(submit, "H2Odeep11.csv" , row.names = F)

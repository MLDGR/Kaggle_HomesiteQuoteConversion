library(sfsmisc)
library(parallel)
library(xgboost)
library(readr)
setwd("/home/ismael/Proyectos/Kaggle/Homesite/")

train=read_csv("train.csv")
test=read_csv("test.csv")

# There are some NAs in the integer columns so conversion to zero
train[is.na(train)]   <- -1
test[is.na(test)]   <- -1

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

train=train[,-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)]
test=test[,-c(128,133,138,181,191,192,199,207,213,217,283,291,293,295)]

columnas=c(126,127,141,142,147,148,151,152,154,155,168,169,170,171,179:205,207:281)
lista=list(c(126,127),c(141,142),c(147,148),c(151,152),c(154,155),c(168,169),c(170,171),c(179:205),c(207:216),c(217:218),c(219:220),c(221:222),
           c(223:224),c(225:226),c(227:228),c(229:230),c(231:232),c(233:234),c(235:236),c(237:238),c(239:240),c(241:242),c(243:244),c(245:250),
           c(251:252),c(253:254),c(255:256),c(257:268),c(269:270),c(271:273),c(274:277),c(278:279),c(280:281))

temp=NULL
for(j in 1:length(lista)){
  cat(length(lista)-j,"\n")
  temp=cbind(temp,matrix(unlist(mclapply(1:nrow(train),function(i){
    as.intBase(as.matrix(as.integer(train[i,lista[[j]]])),base=25)
  },mc.preschedule = TRUE, mc.set.seed = TRUE,
  mc.silent = FALSE, mc.cores = getOption("mc.cores", 6L),
  mc.cleanup = TRUE, mc.allow.recursive = TRUE))))
}


for(i in 1:ncol(temp)){
  minimo=abs(min(temp[,i]))
  temp[,i]=temp[,i]+minimo+1
  temp[,i]=log(temp[,i])
  temp[,i]=as.factor(temp[,i])
  levels(temp[,i])=levels(temp[,i])=1:length(levels(temp[,i]))
}


train=train[,-columnas]
train=cbind(train,temp)
for(i in 172:ncol(train)){
  colnames(train)[i]=paste("X",i,sep="")
  train[,i]=as.integer(train[,i])
}

n_col=ncol(train)
for(i in 172:n_col){
  coeficientes=as.data.frame(table(train[,i]))
  colnames(coeficientes)[1]=paste("X",i,sep="")
  train=merge(train,coeficientes)
  colnames(train)[ncol(train)]=paste("Freq",i,sep="")
}
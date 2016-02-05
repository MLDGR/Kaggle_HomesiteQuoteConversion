library(randomForest)
library(readr)

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
train <- train[,-c(2)]

# seperating out the elements of the date column for the train set
test$month <- as.integer(format(test$Original_Quote_Date, "%m"))
test$year <- as.integer(format(test$Original_Quote_Date, "%y"))
test$day <- weekdays(as.Date(test$Original_Quote_Date))

# removing the date column
test <- test[,-c(2)]

train[is.na(train)]   <- -1
test[is.na(test)]   <- -1

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

sampleSize <- floor(.5 * nrow(train))
set.seed(123)
partitionInd <- sample(seq_len(nrow(train)), size = sampleSize)
train <- train[partitionInd, ]

fol <- formula(QuoteConversion_Flag ~ .)

model <- randomForest(fol,mtry = 17,ntree = 2500, method = "class", data = train)

predictions <- predict(model, test, type = "class")

submissionRF <- data.frame(QuoteNumber = test$QuoteNumber, QuoteConversion_Flag = predictions)
write.csv(submissionRF, "predictions_RF_R_v3.0.csv", row.names = F)

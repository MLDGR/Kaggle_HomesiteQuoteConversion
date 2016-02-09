library(randomForest)

trainData <- train
test <- test

sampleSize <- floor(.5 * nrow(trainData))
set.seed(123)
partitionInd <- sample(seq_len(nrow(trainData)), size = sampleSize)
train <- trainData[partitionInd, ]

fol <- formula(QuoteConversion_Flag ~ .)

model <- randomForest(fol,ntree = 1000, method = "class", data = train)

predictions <- predict(model, test, type = "class")

submissionRF <- data.frame(QuoteNumber = test$QuoteNumber, QuoteConversion_Flag = predictions)
write.csv(submissionRF, "predictions_RF_R_v2.0.csv", row.names = F)

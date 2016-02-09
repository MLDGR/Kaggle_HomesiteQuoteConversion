library(randomForest)

trainData <- read.csv("../input/train.csv", colClasses = c(rep('factor', 299)))
test <- read.csv("../input/test.csv", colClasses = c(rep('factor', 298)))

sampleSize <- floor(.5 * nrow(trainData))
set.seed(123)
partitionInd <- sample(seq_len(nrow(trainData)), size = sampleSize)
train <- trainData[partitionInd, ]

fol <- formula(QuoteConversion_Flag ~ SalesField5 
               + SalesField4 
               + CoverageField9 
               + PersonalField10A
               + SalesField1A 
               + GeographicField16A 
               + CoverageField6B 
               + GeographicField41A 
               + CoverageField6A 
               + GeographicField6A 
               + PersonalField10A 
               + GeographicField64 
               + GeographicField13A 
               + GeographicField11A 
               + Field6 
               + PropertyField34 
               + GeographicField15A 
               + Field10 
               + Field9 
               + SalesField6 
               + Field7 
               + GeographicField8A)

model <- randomForest(fol, method = "class", data = train)

predictions <- predict(model, test, type = "class")

submission <- data.frame(QuoteNumber = test$QuoteNumber, QuoteConversion_Flag = predictions)
write.csv(submission, "predictionsv1.csv", row.names = F)
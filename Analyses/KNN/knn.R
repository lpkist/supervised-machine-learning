######### KNN analysis
library(tidyverse)
data_raw <- read_csv("https://raw.githubusercontent.com/lpkist/supervised-machine-learning/main/Analyses/KNN/KNNAlgorithmDataset.csv")
glimpse(data_raw)
data <- data_raw %>% select(-id) %>% 
  mutate(diagnosis = factor(diagnosis))

set.seed(5)

##### Using directly the method
validationIndex <- createDataPartition(data$diagnosis, p=2/3, list=FALSE)

train <- data[validationIndex,] # 2/3 of data to training
test <- data[-validationIndex,] # remaining 1/3 for test


library(caret)
trainControl <- trainControl(method = "repeatedcv", number=15, repeats=5)
metric <- "Accuracy"
grid <- expand.grid(.k=seq(1,20,by=1))
fit_knn <- train(diagnosis~., data=train, method="knn", 
                 metric=metric, tuneGrid=grid, trControl=trainControl)
k <- fit_knn$bestTune # the optimal k
print(fit_knn)
plot(fit_knn)

prediction <- predict(fit_knn, newdata = test)
cf <- confusionMatrix(prediction, test$diagnosis)
print(cf)

######## Scaling the variables
train_scaled <- cbind(data.frame("diagnosis" = train$diagnosis), scale(train %>% select(-diagnosis)))
test_scaled <- cbind(data.frame("diagnosis" = test$diagnosis), scale(test %>% select(-diagnosis)))

fit_knn_scaled <- train(diagnosis~., data=train_scaled, method="knn", 
                 metric=metric, tuneGrid=grid, trControl=trainControl)
k_scaled <- fit_knn_scaled$bestTune # the optimal k
print(fit_knn_scaled)
plot(fit_knn_scaled)

prediction_scaled <- predict(fit_knn_scaled, newdata = test_scaled)
cf_scaled <- confusionMatrix(prediction_scaled, test_scaled$diagnosis)
print(cf_scaled)

######## Comparing the models
print(cf_scaled)
print(cf)
cf$overall["Accuracy"] < cf_scaled$overall["Accuracy"]
# In almost all aspects, cf_scaled was better. That's why you should always scale
# your variables!






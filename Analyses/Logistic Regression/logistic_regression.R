library(tidyverse)
library(GGally)
library(car)
library(caret)
library(pROC)

data <- read_csv("https://raw.githubusercontent.com/lpkist/supervised-machine-learning/main/Analyses/Logistic%20Regression/diabetes2.csv")

glimpse(data)
summary(data)
# splitting the dataset into train and test
idx_test <- sample(1:nrow(data), nrow(data)/3, replace = F)
test <- data[idx_test,]
train <- data[-idx_test,]

ggpairs(train)
logistic1 <- glm(Outcome ~ ., family = binomial(link = "logit"),
                 data = train) # generic logistic regression
prediction_l1 <- data.frame(prob = predict(logistic1, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_l1 <- confusionMatrix(factor(prediction_l1$Outcome), factor(test$Outcome))

summary(logistic1)
logistic2 <- glm(Outcome ~ Pregnancies+Glucose+BloodPressure+Insulin+BMI+DiabetesPedigreeFunction +Age,
                 family = binomial(link = "logit"),
                 data = train) 
summary(logistic2) # remove Age
prediction_l2 <- data.frame(prob = predict(logistic2, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_l2 <- confusionMatrix(factor(prediction_l2$Outcome), factor(test$Outcome))

logistic3 <- glm(Outcome ~ Pregnancies+Glucose+BloodPressure+Insulin+BMI+DiabetesPedigreeFunction,
                 family = binomial(link = "logit"),
                 data = train) 
summary(logistic3) # remove Insulin
prediction_l3 <- data.frame(prob = predict(logistic3, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_l3 <- confusionMatrix(factor(prediction_l3$Outcome), factor(test$Outcome))

logistic4 <- glm(Outcome ~ Pregnancies+Glucose+BloodPressure+BMI+DiabetesPedigreeFunction,
                 family = binomial(link = "logit"),
                 data = train) 
summary(logistic4) # remove BloodPressure
prediction_l4 <- data.frame(prob = predict(logistic4, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_l4 <- confusionMatrix(factor(prediction_l4$Outcome), factor(test$Outcome))

logistic5 <- glm(Outcome ~ Pregnancies+Glucose+BMI+DiabetesPedigreeFunction,
                 family = binomial(link = "logit"),
                 data = train) 
summary(logistic5) # remove DiabetesPedigreeFunction  
prediction_l5 <- data.frame(prob = predict(logistic5, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_l5 <- confusionMatrix(factor(prediction_l5$Outcome), factor(test$Outcome))

logistic6 <- glm(Outcome ~ Pregnancies+Glucose+BMI,
                 family = binomial(link = "logit"),
                 data = train) 
summary(logistic6) # 
prediction_l6 <- data.frame(prob = predict(logistic6, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_l6 <- confusionMatrix(factor(prediction_l6$Outcome), factor(test$Outcome))
ggpairs(data.frame(logistic6$residuals, train)) # no correlation

qqPlot(logistic6$residuals)

### Trying other link functions

cloglog1 <- glm(Outcome ~ Pregnancies+Glucose+BMI,
                family = binomial(link = "cloglog"), data = train)
summary(cloglog1)
prediction_c1 <- data.frame(prob = predict(cloglog1, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_c1 <- confusionMatrix(factor(prediction_c1$Outcome), factor(test$Outcome))
qqPlot(cloglog1$residuals)



cauchit1 <- glm(Outcome ~ Pregnancies+Glucose+BMI,
                family = binomial(link = "cauchit"), data = train)
summary(cauchit1)
qqPlot(cauchit1$residuals)
prediction_ch1 <- data.frame(prob = predict(cauchit1, test, type = "response")) %>% 
  mutate(Outcome = ifelse(prob>=0.5, 1,0))
CM_ch1 <- confusionMatrix(factor(prediction_ch1$Outcome), factor(test$Outcome))

# What's the best 'cut point'?
change_cut_point <- function(cut){
  logistic1 <- glm(Outcome ~ ., family = binomial(link = "logit"),
                   data = train) # generic logistic regression
  prediction_l1 <- data.frame(prob = predict(logistic1, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_l1 <- confusionMatrix(factor(prediction_l1$Outcome), factor(test$Outcome))
  auc_l1 <- roc(test$Outcome, prediction_l1$prob) %>% auc %>% as.numeric()
  CM_l1$byClass <- c(CM_l1$byClass, "AUC" = roc(test$Outcome, prediction_l1$prob) %>% auc %>% as.numeric())
  
  logistic2 <- glm(Outcome ~ Pregnancies+Glucose+BloodPressure+Insulin+BMI+DiabetesPedigreeFunction +Age,
                   family = binomial(link = "logit"),
                   data = train) 
  prediction_l2 <- data.frame(prob = predict(logistic2, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_l2 <- confusionMatrix(factor(prediction_l2$Outcome), factor(test$Outcome))
  auc_l2 <- roc(test$Outcome, prediction_l2$prob) %>% auc %>% as.numeric()
  CM_l2$byClass <- c(CM_l2$byClass, "AUC" = roc(test$Outcome, prediction_l2$prob) %>%
                       auc %>% as.numeric())
  
  logistic3 <- glm(Outcome ~ Pregnancies+Glucose+BloodPressure+Insulin+BMI+DiabetesPedigreeFunction,
                   family = binomial(link = "logit"),
                   data = train) 
  prediction_l3 <- data.frame(prob = predict(logistic3, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_l3 <- confusionMatrix(factor(prediction_l3$Outcome), factor(test$Outcome))
  auc_l3 <- roc(test$Outcome, prediction_l3$prob) %>% auc %>% as.numeric()
  CM_l3$byClass <- c(CM_l3$byClass, "AUC" = roc(test$Outcome, prediction_l3$prob) %>%
                       auc %>% as.numeric())
  
  logistic4 <- glm(Outcome ~ Pregnancies+Glucose+BloodPressure+BMI+DiabetesPedigreeFunction,
                   family = binomial(link = "logit"),
                   data = train) 
  prediction_l4 <- data.frame(prob = predict(logistic4, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_l4 <- confusionMatrix(factor(prediction_l4$Outcome), factor(test$Outcome))
  auc_l4 <- roc(test$Outcome, prediction_l4$prob) %>% auc %>% as.numeric()
  CM_l4$byClass <- c(CM_l4$byClass, "AUC" = roc(test$Outcome, prediction_l4$prob) %>%
                       auc %>% as.numeric())
  
  logistic5 <- glm(Outcome ~ Pregnancies+Glucose+BMI+DiabetesPedigreeFunction,
                   family = binomial(link = "logit"),
                   data = train) 
  prediction_l5 <- data.frame(prob = predict(logistic5, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_l5 <- confusionMatrix(factor(prediction_l5$Outcome), factor(test$Outcome))
  auc_l5 <- roc(test$Outcome, prediction_l5$prob) %>% auc %>% as.numeric()
  CM_l5$byClass <- c(CM_l5$byClass, "AUC" = roc(test$Outcome, prediction_l5$prob) %>%
                       auc %>% as.numeric())
  
  logistic6 <- glm(Outcome ~ Pregnancies+Glucose+BMI,
                   family = binomial(link = "logit"),
                   data = train) 
  prediction_l6 <- data.frame(prob = predict(logistic6, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_l6 <- confusionMatrix(factor(prediction_l6$Outcome), factor(test$Outcome))
  auc_l6 <- roc(test$Outcome, prediction_l6$prob) %>% auc %>% as.numeric()
  CM_l6$byClass <- c(CM_l6$byClass, "AUC" = roc(test$Outcome, prediction_l6$prob) %>%
                       auc %>% as.numeric())
  
  cloglog1 <- glm(Outcome ~ Pregnancies+Glucose+BMI,
                  family = binomial(link = "cloglog"), data = train)
  prediction_c1 <- data.frame(prob = predict(cloglog1, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_c1 <- confusionMatrix(factor(prediction_c1$Outcome), factor(test$Outcome))
  auc_c1 <- roc(test$Outcome, prediction_c1$prob) %>% auc %>% as.numeric()
  CM_c1$byClass <- c(CM_c1$byClass, "AUC" = roc(test$Outcome, prediction_c1$prob) %>%
                       auc %>% as.numeric())
  
  cauchit1 <- glm(Outcome ~ Pregnancies+Glucose+BMI,
                  family = binomial(link = "cauchit"), data = train)
  prediction_ch1 <- data.frame(prob = predict(cauchit1, test, type = "response")) %>% 
    mutate(Outcome = ifelse(prob>=cut, 1,0))
  CM_ch1 <- confusionMatrix(factor(prediction_ch1$Outcome), factor(test$Outcome))
  auc_ch1 <- roc(test$Outcome, prediction_ch1$prob) %>% auc %>% as.numeric()
  CM_ch1$byClass <- c(CM_ch1$byClass, "AUC" = roc(test$Outcome, prediction_ch1$prob) %>%
                       auc %>% as.numeric())
  
  return(list(CM_l1$byClass, CM_l2$byClass, CM_l3$byClass,
              CM_l4$byClass, CM_l5$byClass, CM_l6$byClass,
              CM_c1$byClass, CM_ch1$byClass))
}

results <- list()
for(i in 1:19){
  aux <- change_cut_point(i*.05)
  results[[i]] <- do.call(rbind,aux) %>% data.frame(Model =
                                      c("l1", "l2", "l3", "l4", "l5", "l6", "c1", "ch1"),
                                    Cut = i*.05)
}
results <- do.call(rbind, results)
results %>% mutate(sum = Sensitivity+Specificity) %>% 
  arrange(-sum) %>% head(1) 

results %>%
  arrange(-F1) %>% head(1)

results %>% arrange(-AUC) %>% head(1)
# Depending on the interest criteria, the best model and cut point
# are different. In this case, I show 3 possible models.

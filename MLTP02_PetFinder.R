# MSBA Team 13
# Christine Luong, Tianming Chen, Kathryn Ziccarelli
# ML TP02 - PetFinder Analysis

# Clear Variables
rm(list=ls())

# Load data from train dataset
petFinder <- read.csv("train.csv")

# Remove columns that are not useful (string and random ID) -- RescuerID, Description, PetID
petFinder <- petFinder[, -c(19, 21, 22)]

# Set all mislabeled data (not int 0 - 4) to NA 
petFinder$AdoptionSpeed <- ifelse(petFinder$AdoptionSpeed %in% c(0, 1, 2, 3, 4), petFinder$AdoptionSpeed, NA)

# Check for NA data points
sum(is.na(petFinder$AdoptionSpeed))
petFinder <- na.omit(petFinder)

# Change all data into numeric or factor for further analysis
attach(petFinder)
petFinder$Name <- as.factor(ifelse(petFinder$Name == "", 0, 1))
petFinder$Type <- as.factor(ifelse(petFinder$Type == 1, "Dog", "Cat"))
petFinder$Age <- as.numeric(Age)
petFinder$Breed1 <- as.numeric(Breed1)
petFinder$Breed2 <- as.numeric(Breed2)
petFinder$Gender <- as.factor(Gender)
petFinder$Color1 <- as.numeric(Color1)
petFinder$Color2 <- as.numeric(Color2)
petFinder$Color3 <- as.numeric(Color3)
petFinder$MaturitySize <- as.factor(MaturitySize)
petFinder$FurLength <- as.factor(FurLength)
petFinder$Vaccinated <- as.factor(Vaccinated)
petFinder$Dewormed <- as.factor(Dewormed)
petFinder$Sterilized <- as.factor(Sterilized)
petFinder$Health <- as.factor(Health)
petFinder$Quantity <-as.numeric(Quantity)
petFinder$Fee <-as.numeric(Fee)
petFinder$State <- as.factor(State)
petFinder$VideoAmt <- as.numeric(VideoAmt)
petFinder$PhotoAmt <-as.numeric(PhotoAmt)
petFinder$AdoptionSpeed <- as.factor(AdoptionSpeed)

str(petFinder)
# 21 variables, all numeric and factor

# Exploratory Data Analysis (EDA)

# Load Libraries
library(ggplot2)
library(tidyverse)

# Type Plot
petFinder %>%
  ggplot(aes(x = Type, fill = Type)) +
  geom_bar(stat = "count", color = "black") +
  theme_minimal() +
  ylab(NULL) +
  scale_fill_brewer(palette="YlGn") +
  theme(legend.position = "none")

# Adoption Speed Plot
petFinder %>%
  ggplot(aes(x = AdoptionSpeed, fill = AdoptionSpeed)) +
  geom_bar(stat = "count", color = "black") +
  theme_minimal() +
  ylab(NULL) +
  scale_fill_brewer(palette="YlGn") +
  scale_x_discrete(labels = c("Same Day", "1st Week", "1st Month", "2nd & 3rd Month", "No Adoption After 100 Days")) +
  theme(legend.position = "none")

# Age Density Plot
petFinder %>%
  ggplot(aes(x = Age, fill = Type)) +
  geom_density(alpha = 0.5, adjust = 2) +
  xlim(0, 100) +
  ylab(NULL) +
  theme_minimal() +
  scale_fill_brewer(palette="YlGn")

# Fee Density Plot
petFinder %>%
  ggplot(aes(x = Fee, fill = Type)) +
  geom_density(alpha = 0.5, adjust = 2) +
  xlim(0, 1000) +
  ylab(NULL) +
  theme_minimal() +
  scale_fill_brewer(palette="YlGn") 

# Split Data - Train vs. Test
train <- sample(1:nrow(petFinder), nrow(petFinder) * 0.7)
train.petFinder <- petFinder[train, ] #10,495 observations
test.petFinder <- petFinder[-train, ] #4,498 observations

# GBM Model

# Load library
library(gbm)
set.seed(100)
boost.model <- gbm(AdoptionSpeed~., data = train.petFinder, distribution="gaussian", n.trees=1000, interaction.depth = 1)
# interaction.depth of 1 often works well for boosting, trees are stumps (1 split)

summary(boost.model)
# Breed1 (primary breed), Age, and State are most important influences on AdoptionSpeed
# Breed1 21.06799776
# Age 20.61227726
# State 17.16768418

# XGBoost Model

# Load libraries
library(xgboost)
library(dplyr)
library(caret)

# Training X and Y in matrix form:
x <- model.matrix(AdoptionSpeed~., train.petFinder)[ ,-21]
y <- train.petFinder$AdoptionSpeed
y <- as.matrix(y)

# Define the number of classes:
numberOfClasses <- unique(y) %>% length()

# Initial Model
set.seed(100)
xgboost1 <- xgboost(data=x, label=y, eta=0.5, nround=5, 
                    max.depth=2, verbose=1, objective="multi:softmax", num_class=numberOfClasses)
# eta = learning rate (step size of each boosting step)
# max.depth = maximum depth of tree, 2 splits
# nrounds = max number of iterations 

# Testing X and Y in matrix form:
x2 <- model.matrix(AdoptionSpeed~., test.petFinder)[ ,-21]
y2 <- petFinder[-train, "AdoptionSpeed"]
y2 <- as.matrix(y2)

# Make Predictions
set.seed(100)
xgb_predictions <- predict(xgboost1, newdata=x2)
(table1 <- table(xgb_predictions, y2))
# y2
#    0   1   2   3   4
# 1  19  81  63  42  33
# 2  55 570 643 416 351
# 3   5  57 117 140  59
# 4  40 261 355 344 847

# Accuracy
(acc_xgboost1 <- (81+643+140+847)/4498)
# 0.3803913 = 38.03%

# Tune eta: Loop through learning rates from 0 through 2, interval=0.1
set.seed(100)
acc_tune <- c()
eta_seq <- seq(0, 2, 0.1) 
for (e in eta_seq){
  set.seed(100)
  xgboost_tune <- xgboost(data=x, label=y, eta=e, nround=5, max.depth=2, verbose=1, objective="multi:softmax", num_class=numberOfClasses)
  xgb_predictions_tune <- predict(xgboost_tune, newdata=x2)
  acc_confusion_matrix <- confusionMatrix(as.factor(xgb_predictions_tune), as.factor(y2))
  acc_confusion_matrix
  acc_tune <- c(acc_tune, acc_confusion_matrix$overall[1])
}

# Results:
acc_tune
rbind(eta_seq, acc_tune)
(acc_xgboost2 <- max(acc_tune))
# highest accuracy is 39.84% at eta = 1.9


# SVM Model

#Load library
library(e1071)

# Set response variable as factor
petFinder$AdoptionSpeed <- as.factor(AdoptionSpeed)
test.petFinder$AdoptionSpeed <- as.factor(test.petFinder$AdoptionSpeed)
train.petFinder$AdoptionSpeed <- as.factor(train.petFinder$AdoptionSpeed)

# Initial model
set.seed(100)
svm.model1 <- svm(AdoptionSpeed~., data=petFinder, kernel="linear", cost=1)
summary(svm.model1)
# SVM-Type:  C-classification 
# SVM-Kernel:  linear 
# cost:  1 
# Number of Support Vectors:  14308
# ( 4037 410 3243 3080 3538 )

# Make Predictions
set.seed(100)
pred1 <- predict(svm.model1, test.petFinder)
confusion_matrix1 <- table(pred1, test.petFinder$AdoptionSpeed)
print(confusion_matrix1)
#     0   1   2   3   4
# 0   0   0   0   0   0
# 1  14  78  44  34  38
# 2  74 715 875 627 638
# 3   8  15  26  54  47
# 4  23 161 233 227 567

# Accuracy:
(acc_svm1 <- (78+875+54+567)/sum(confusion_matrix1)) # 0.3499333 = 35.00%

# Model with list of possible cost values
set.seed(100)
tune.svm <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="linear", ranges = list(cost = c(.01, .1, 1, 5)))
summary(tune.svm)
# - best parameters:
#   cost: 1
# - best performance: 0.6548812 
# - Detailed performance results:
#   cost     error dispersion
# 1 0.01 0.6601207 0.01681641
# 2 0.10 0.6584069 0.01706256
# 3 1.00 0.6548812 0.01675555
# 4 5.00 0.6559295 0.01670862

# Obtain Best Model
tune.svm$best.model
# cost:  1 
# Number of Support Vectors:  10029
tune.svm$best.performance
# 0.6548812 = lowest training CV error

# Make Predictions
set.seed(100)
pred2 <- predict(tune.svm$best.model, test.petFinder)
confusion_matrix2 <- table(pred2, test.petFinder$AdoptionSpeed)
print(confusion_matrix2)
#     0   1   2   3   4
# 0   0   0   0   0   0
# 1  14  76  44  34  36
# 2  73 703 853 612 621
# 3   9  19  39  64  52
# 4  23 171 242 232 581

# Accuracy:
(acc_svm2_linear <- (76+853+64+581)/sum(confusion_matrix2)) # 0.3499333 = 34.99%

# Model with Radial kernel
tune.svm.radial3 <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="radial", ranges = list(cost = c(.01, .1, 1, 5)))
tune.svm.radial3$best.model
# cost:  5 
# Number of Support Vectors:  9856
tune.svm.radial3$best.performance
# 0.618004 --> lower than linear kernel tune

# Make Predictions
set.seed(100)
pred3 <- predict(tune.svm.radial3$best.model, test.petFinder)
confusion_matrix3 <- table(pred3, test.petFinder$AdoptionSpeed)
print(confusion_matrix3) 
#     0   1   2   3   4
# 0   0   1   1   0   0
# 1  32 212 158 104  97
# 2  45 512 635 418 351
# 3   7  47  95 134  72
# 4  35 197 289 286 770
# Accuracy:
(acc_svm3_radial <- (212+635+134+770)/sum(confusion_matrix3)) # 0.3892841 = 38.92% accuracy 

# Model with Radial kernel, add gamma=1
tune.svm.radial4 <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="radial", cost=5, gamma=1)
tune.svm.radial4$best.model
# cost:  5 
# Number of Support Vectors:  10353
tune.svm.radial4$best.performance
# 0.6668885

# Make Predictions
set.seed(100)
pred4 <- predict(tune.svm.radial4$best.model, test.petFinder)
confusion_matrix4 <- table(pred4, test.petFinder$AdoptionSpeed)
print(confusion_matrix4)
#     0   1   2   3   4
# 0   4   6   0   5   1
# 1  18 170 137  85 101
# 2  29 287 388 240 225
# 3  12 105 167 188 142
# 4  56 401 486 424 821

# Accuracy: 
(acc_svm4_radial <- (4+170+388+188+821)/sum(confusion_matrix4))
# 0.3492663 = 34.92% accuracy

# Final Results Summary
#XGBoost1: 38.03%
#XGboost2 with eta = 1.9: 39.84%
#Initial SVM Model: 35.00%
#SVM2 Linear with Cost 1: 34.99% 
#SVM Radial with Cost 5: 38.92%
#SVM Radial with Gamma 1: 34.92% 


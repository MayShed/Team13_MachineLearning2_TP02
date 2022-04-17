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
# Breed1 23.52170765
# Age    21.08859766
# State  15.42976931

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
xgboost1 <- xgboost(data=x, label=y, eta=0.5, nround=5, max.depth=2, verbose=1, objective="multi:softmax", num_class=numberOfClasses)
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
#     0   1   2   3   4
# 1  24 156 110  82  70
# 2  45 486 610 421 335
# 3  11  48 108 132  58
# 4  43 253 380 343 783

# Accuracy
(acc_xgboost1 <- (156+610+132+783)/4498)
# 0.3737217 = 37.37%

# Tune eta: Loop through learning rates from 0 through 2, interval=0.1
set.seed(100)
acc_tune <- c()
eta_seq <- seq(0, 2, 0.1) 
for (e in eta_seq){
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
# highest accuracy is 39.24% at eta = 1.0


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
#   0   1   2   3   4
#0  0   0   0   0   0
#1  12  66  51  41  31
#2  75 719 898 662 637
#3  5  20  27  59  31
#4  25 132 243 222 542

# Accuracy:
(acc_svm1 <- (66+898+59+542)/sum(confusion_matrix1)) # 0.3479 = 34.79%

# Model with list of possible cost values
set.seed(100)
tune.svm <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="linear", ranges = list(cost = c(.01, .1, 1, 5)))
summary(tune.svm)
# - best parameters:
#   cost: 1
# - best performance: 0.6504018 
# - Detailed performance results:
#   cost     error  dispersion
# 1 0.01 0.6628830 0.012829320
# 2 0.10 0.6507840 0.009974294
# 3 1.00 0.6504018 0.011884579
# 4 5.00 0.6512593 0.012396875

# Obtain Best Model
tune.svm$best.model
# cost:  1 
# Number of Support Vectors:  10011
tune.svm$best.performance
# 0.6504018 = lowest training CV error

# Make Predictions
set.seed(100)
pred2 <- predict(tune.svm$best.model, test.petFinder)
confusion_matrix2 <- table(pred2, test.petFinder$AdoptionSpeed)
print(confusion_matrix2)
#     0   1   2   3   4
# 0   0   0   0   0   0
# 1  23 101  85  64  51
# 2  64 686 869 646 624
# 3   4  17  24  53  39
# 4  26 133 241 221 527

# Accuracy:
(acc_svm2_linear <- (101+869+53+527)/sum(confusion_matrix2)) # 0.3445976 = 34.45%

# Model with Radial kernel
tune.svm.radial3 <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="radial", ranges = list(cost = c(.01, .1, 1, 5)))
tune.svm.radial3$best.model
# cost:  5 
# Number of Support Vectors:  9856
tune.svm.radial3$best.performance
# 0.6139086 --> lower than linear kernel tune

# Make Predictions
set.seed(100)
pred3 <- predict(tune.svm.radial3$best.model, test.petFinder)
confusion_matrix3 <- table(pred3, test.petFinder$AdoptionSpeed)
print(confusion_matrix3) 
#     0   1   2   3   4
# 0   0   0   0   0   0
# 1  35 220 200 103 118
# 2  42 460 596 443 318
# 3   4  61  92 136  64
# 4  36 196 331 302 741
# Accuracy:
acc_svm3_radial <- (220+596+136+741)/sum(confusion_matrix3) # 0.3763895 = 37.6% accuracy (highest)

# Model with Radial kernel, add gamma=1
tune.svm.radial4 <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="radial", cost=5, gamma=1)
tune.svm.radial4$best.model
# cost:  5 
# Number of Support Vectors:  10349
tune.svm.radial4$best.performance
# 0.6646007

# Make Predictions
set.seed(100)
pred4 <- predict(tune.svm.radial4$best.model, test.petFinder)
confusion_matrix4 <- table(pred4, test.petFinder$AdoptionSpeed)
print(confusion_matrix4)
#     0   1   2   3   4
# 0   5   5   3   6   2
# 1  16 175 196  81 103
# 2  31 267 365 225 205
# 3  13 109 172 182 106
# 4  52 381 483 490 825

# Accuracy: 
(acc_svm4_radial <- (5 + 175 + 365 + 182 + 825)/sum(confusion_matrix4))
# 0.3450422 = 34.50% accuracy

# Results Summary
#XGBoost1: 37.37%
#XGboost2 with eta = 1: 39.24%
#Initial SVM Model: 34.79%
#SVM2 Linear with Cost 1: 34.45%
#SVM Radial with Cost 5: 37.6% 
#SVM Radial with Gamma 1: 34.50 





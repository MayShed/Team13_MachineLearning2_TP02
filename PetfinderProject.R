# MSBA Team 13
# Christine Luong, Tianming Chen, Kathryn Ziccarelli
# PetFinder Analysis

# Load data from train dataset, we don't use the "test set" for validation because they are not labeled  
petFinder <- read.csv("train/train.csv")

# Remove columns that are hard to be useful (string and random ID) -- Name, RescuerID, Descriptoin, PetID
petFinder <- petFinder[, -c(19, 21, 22)]

# Set all mislabled data (not int 0 - 4) to NA 
petFinder$AdoptionSpeed <- ifelse(petFinder$AdoptionSpeed %in% c(0, 1, 2, 3, 4), petFinder$AdoptionSpeed, NA)

# Remove 739 NA data points. luckily, no data need to be further removed 
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


# EDA

library(ggplot2)
library(tidyverse)

petFinder %>%
  ggplot(aes(x = Type, fill = Type)) +
  geom_bar(stat = "count", color = "black") +
  theme_minimal() +
  ylab(NULL) +
  scale_fill_brewer(palette="YlGn") +
  theme(legend.position = "none")

petFinder %>%
  ggplot(aes(x = AdoptionSpeed, fill = AdoptionSpeed)) +
  geom_bar(stat = "count", color = "black") +
  theme_minimal() +
  ylab(NULL) +
  scale_fill_brewer(palette="YlGn") +
  scale_x_discrete(labels = c("Same Day", "1st Week", "1st Month", "2nd & 3rd Month", "No Adoption After 100 Days")) +
  theme(legend.position = "none")

petFinder %>%
  ggplot(aes(x = Age, fill = Type)) +
  geom_density(alpha = 0.5, adjust = 2) +
  xlim(0, 100) +
  ylab(NULL) +
  theme_minimal() +
  scale_fill_brewer(palette="YlGn")

petFinder %>%
  ggplot(aes(x = Fee, fill = Type)) +
  geom_density(alpha = 0.5, adjust = 2) +
  xlim(0, 1000) +
  ylab(NULL) +
  theme_minimal() +
  scale_fill_brewer(palette="YlGn") 


# Train - test split
train <- sample(1:nrow(petFinder), nrow(petFinder) * 0.7)

train.petFinder <- petFinder[train, ]
test.petFinder <- petFinder[-train, ]


# GBM
library(gbm)
set.seed(100)

boost.model <- gbm(AdoptionSpeed~., data = train.petFinder, distribution="gaussian", n.trees=1000, interaction.depth = 1)
# interaction.depth of 1 often works well for boosting, trees are stumps (1 split)

summary(boost.model)
# Breed1 (primary breed), Age, and State are most important influences on AdoptionSpeed
# Breed1 23.52170765
# Age    21.08859766
# State  15.42976931


# XGBoost

# load libraries
library(xgboost)
library(dplyr)
library(caret)

set.seed(100)
# Training X and Y in matrix form:
x <- model.matrix(AdoptionSpeed~., train.petFinder)[ ,-21]
y <- train.petFinder$AdoptionSpeed
y <- as.matrix(y)
# define the number of classes:
numberOfClasses <- unique(y) %>% length()

# Initial Model
xgboost1 <- xgboost(data=x, label=y, eta=0.5, nround=5, max.depth=2, verbose=1, objective="multi:softmax", num_class=numberOfClasses)
# eta = learning rate (step size of each boosting step)
# max.depth = maximum depth of tree, 2 splits
# nrounds = max number of iterations 

# Predictions
set.seed(100)
# Testing X and Y in matrix form:
x2 <- model.matrix(AdoptionSpeed~., test.petFinder)[ ,-21]
y2 <- petFinder[-train, "AdoptionSpeed"]
y2 <- as.matrix(y2)

xgb_predictions <- predict(xgboost1, newdata=x2)
table1 <- table(xgb_predictions, y2)

#                 y2
#xgb_predictions    0   1   2   3   4
# 1  26 153 147  80  47
# 2  43 463 591 437 331
# 3  12  52 102 137  71
# 4  60 239 384 315 808

accuracy_xgboost <- (153+591+137+808)/4498
print(accuracy_xgboost)
# 0.3755002 = 37.55%

# Tune eta: Loop through learning rates from 0 through 2, interval=0.1
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
# highest accuracy is 38.08% at eta=1.5 




# SVM

petFinder$AdoptionSpeed <- as.factor(AdoptionSpeed)
test.petFinder$AdoptionSpeed <- as.factor(test.petFinder$AdoptionSpeed)
train.petFinder$AdoptionSpeed <- as.factor(train.petFinder$AdoptionSpeed)

library(e1071)

# Initial model
set.seed(100)
svm.model1 <- svm(AdoptionSpeed~., data=petFinder, kernel="linear", cost=1)
summary(svm.model1)
#  SVM-Type:  C-classification 
# SVM-Kernel:  linear 
# cost:  1 
# Number of Support Vectors:  14343
# ( 4037 410 3242 3087 3567 )

# prediction
pred1 <- predict(svm.model1, test.petFinder)
confusion_matrix <- table(pred1, test.petFinder$AdoptionSpeed)
print(confusion_matrix)
#     0   1   2   3   4
# 0   0   0   0   0   0
# 1   1  15  10  10  10
# 2  85 760 927 696 656
# 3  10  13  30  57  42
# 4  27 155 241 215 538
# Accuracy:
(15+927+57+538)/sum(confusion_matrix) # 0.3417 = 34.17%

# Model with list of possible cost values
set.seed(100)
tune.svm <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="linear", ranges = list(cost = c(.01, .1, 1, 5)))
summary(tune.svm)

#- Detailed performance results:
#  cost    error dispersion
# 1 0.01 0.6586009 0.01878026
# 2 0.10 0.6561230 0.01544152
# 3 1.00 0.6550744 0.01793455
# 4 5.00 0.6556459 0.01758748

tune.svm$best.model
# cost:  1 
# Number of Support Vectors:  10025
tune.svm$best.performance
# 0.6550744 = lowest training CV error

# prediction
set.seed(100)
pred2 <- predict(tune.svm$best.model, test.petFinder)
confusion_matrix2 <- table(pred2, test.petFinder$AdoptionSpeed)
print(confusion_matrix2)
#    0   1   2   3   4
#0   0   0   0   0   0
#1   2  11   6   7  11
#2  84 759 924 696 656
#3   9  16  30  56  44
#4  28 157 248 219 535
# Accuracy:
(11+924+56+535)/sum(confusion_matrix2) # 0.33926 = 33.92%


# Model with Radial kernel
tune.svm.radial <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="radial", ranges = list(cost = c(.01, .1, 1, 5)))
tune.svm.radial$best.model
# cost:  5 
# Number of Support Vectors:  9849
tune.svm.radial$best.performance
# 0.617912 --> lower than linear kernel tune

# predictions
set.seed(100)
pred4 <- predict(tune.svm.radial$best.model, test.petFinder)
confusion_matrix4 <- table(pred4, test.petFinder$AdoptionSpeed)
print(confusion_matrix4) 
#    0   1   2   3   4
#0   0   0   0   0   0
#1  36 249 190 130 114
#2  44 458 608 406 339
#3  10  53 113 146  64
#4  33 183 297 296 729
# Accuracy:
(249+608+146+729)/sum(confusion_matrix4) # 0.38506 = 38.5% accuracy (highest)

# Model with Radial kernel, add gamma=1
tune.svm.radial2 <- tune(svm, AdoptionSpeed~., data=train.petFinder, kernel="radial", cost=5, gamma=1)
tune.svm.radial2$best.model
# cost:  5 
# Number of Support Vectors:  10346
tune.svm.radial2$best.performance
# 0.6628863 

# predictions
set.seed(100)
pred5 <- predict(tune.svm.radial2$best.model, test.petFinder)
confusion_matrix5 <- table(pred5, test.petFinder$AdoptionSpeed)
print(confusion_matrix5)
#    0   1   2   3   4
#0   6   6   5   6   3
#1  16 167 167  92 107
#2  24 257 368 224 226
#3  12  87 166 168 108
#4  65 426 502 488 802
(6 + 167 + 368 + 168 + 802)/sum(confusion_matrix5)
# 0.3359271 = 33.59% accuracy
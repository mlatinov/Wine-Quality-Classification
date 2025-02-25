

#### Libraries ####
set.seed(123)
library(tidyverse)
library(caret)
library(h2o)
library(ROCR)
library(earth)
library(ranger)
library(vip)
## Load the data from EDA FE

# Preprocess data
train_reg <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/data/clean_data/train_reg.rds")
test_reg <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/data/clean_data/test_reg.rds")

# Regular data 
train_set <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/data/clean_data/train_set.rds")
testing_set <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/data/clean_data/testing_set.rds")

# Remove the quality feature 
test_reg <- test_reg%>% select(-quality)
train_reg <- train_reg%>% select(-quality)

testing_set <- testing_set%>% select(-quality)
train_set <- train_set %>% select(-quality)

# Set the category as factor 
test_reg$category_good <- as.factor(test_reg$category_good)
train_reg$category_good <- as.factor(train_reg$category_good)

testing_set$category <- as.factor(testing_set$category)
train_set$category <- as.factor(train_set$category)

## Objective : Binary Classification on wine category good or bad 

## Models to Test 
# Logistic Regresion on the Preprocess data and on the Regular data .Compare
# Regularized Logistic Regresion : Start with Elastic Net : on the Preprocess data and Regular data.Compare
# KNN : Only on the Preprocess data
# MARS : Multivariate Adaptive Regression Splines 
# Random Forest :
# GBS : Gradient Boosting Machines :


#### Logistic regression ###

# Make a train control with Cross Validation 
tr_control <- trainControl(method = "cv",number = 10)

## Logistic Regression on Regular data

# Fit the model on the testing_set
log_model_1 <- train(
  category ~ .,
  data = train_set,
  method = "glm",
  family = "binomial",
  trControl = tr_control
)

# Predict on the testing data 
pred_1 <- predict(log_model_1,test_reg)

# Create a confusion Matrix
confusionMatrix(
  data = log_model_1,ref = "good",
  reference = testing_set$category,ref = "good"
)

# Note : 
## No Information Rate : 0.5714 = Accuracy : 0.5714 The model just pick 50/50

## logistic regression the Preprocess data
log_model_2 <- train(
  category_good ~ .,
  data = train_reg,
  method = "glm",
  family = "binomial",
  trControl = tr_control
)

# Predict on the testing set 
pred_2 <- predict(log_model_2,test_reg)

# Create a Confusion Matrix
confusionMatrix(
  data = log_model_2,ref = "0",
  reference = test_reg$category_good,ref = "0"
)

# Note:  No Information Rate : 0.5714 < Accuracy : 0.594
  # Pretty bad Kappa : 0.1718 
  # Low  Sensitivity : 0.6419
  # Low  Specificity : 0.5300

## Compare :

# Compute the prob
m1_prob <- predict(log_model_1,testing_set, type = "prob")[,2]
m2_prob <- predict(log_model_2,test_reg,type = "prob")[,2]

## Compute the AUC metrics
perf_1 <- prediction(m1_prob,testing_set$category)%>%
  performance(measure = "tpr",x.measure = "fpr")

perf_2 <- prediction(m2_prob,test_reg$category_good)%>%
  performance(measure = "tpr",x.measure = "fpr")

## Plot the ROC Curv
plot(perf_1,col = "black",lty = 2)
plot(perf_2,add = TRUE,col = "red")
legend(0.553,0.12,legend = c("Logistic Regression on the Normal Traing Data",
                          "Logistic Regression on the Preprocess Data"),
       col = c("black","red"),
       lty = 2:1,
       cex = 0.6)

## Regularized Logistic Regresion on Regular Data 

# Fit Elastic net on the testing set 
log_m_penalty_1 <- train(
  category ~ .,
  data = train_set,
  method = "glmnet",
  family = "binomial",
  trControl = tr_control,
  tuneLenght = 10
)

# Predict on the testing set
log_m_penalty_pred_1 <- predict(log_m_penalty_1,testing_set)

# Confusion Matrix
confusionMatrix(
  data = relevel(log_m_penalty_pred_1,ref = "good"),
  reference = relevel(testing_set$category,ref = "good")
)

## Regularized Logistic Regresion on Preprocess Data
log_m_penalty_2 <- train(
  category_good ~ .,
  data = train_reg,
  method = "glmnet",
  family = "binomial",
  trControl = tr_control,
  tuneLenght = 10
)

# Predict on the testing data 
log_m_penalty_pred_2 <- predict(log_m_penalty_2,test_reg)

# Confusion Matrix
confusionMatrix(
  data = relevel(log_m_penalty_pred_2, ref = "0"),
  reference = relevel(test_reg$category_good, ref = "0")
)

# KNN : Only on the Preprocess data

# Create a resampling method 
cv <-trainControl(
  
  # Specifies Repeated Cross Validations 
  method ="cv",
  
  # Define Number of Folds for each
  number = 10,
  
  # For Binary Classification
  classProbs = TRUE,
  
  # Evaluation Metric
  summaryFunction = twoClassSummary
  )

# Create a KNN Hyperparameter Grid Search 
k_value <- round(sqrt(nrow(train_reg)))
knn_grid <- expand.grid(k = seq(1, k_value, by = 2))

# Correct the category to follow Caret 
train_reg$category_good <- factor(train_reg$category_good, 
                                  levels = c("0", "1"), 
                                  labels = c("bad", "good"))

test_reg$category_good <- factor(test_reg$category_good, 
                                  levels = c("0", "1"), 
                                  labels = c("bad", "good"))

# Fit a KNN model and perform a Grid Search
knn_model <- train(
  category_good ~ .,  
  data = train_reg,  
  method = "knn",
  trControl = cv,
  tuneGrid = knn_grid,
  metric = "ROC"
)
ggplot(knn_model)

# Predict with KNN
knn_model_pred <- predict(knn_model,test_reg)

# Create a confusion matrix
confusionMatrix(
  data = relevel(knn_model_pred,ref = "bad"),
  reference = relevel(test_reg$category_good,ref = "bad")
)

# Note :
 # KNN Accuracy : 0.7697  
  # Sensitivity : 0.7928        
  # Specificity : 0.7389 

#### MARS ####

# Make a grid search
hyper_grid_mars <- expand.grid(
  degree = 1:3,
  nprune = seq(2,100,10) %>% floor()
)

# Fir a MARS model
mars_model <- train(
  x = subset(train_reg,select = -category_good),
  y = train_reg$category_good,
  method = "earth",
  tuneGrid = hyper_grid_mars,
  trControl = tr_control
)

ggplot(mars_model)

# Predict the the MARS model 
mars_prediction <- predict(mars_model,test_reg)

# Make a Confusion Matrix 
confusionMatrix(
  data = relevel(mars_prediction,ref = "bad"),
  reference = relevel(test_reg$category_good,ref = "bad")
)

# Note : MARS 
 #  Accuracy : 0.6179
 # Sensitivity : 0.4714        
 #  Specificity : 0.8133 

#### Random Forest ####

# Starting H2O
h2o.init()

# Convert the training set as h20 objects
train_h2o <- as.h2o(train_reg)

# Set the response column to category_good
responce <- "category_good"

# Set the predictors names
predictors <- setdiff(colnames(train_reg),responce)

## Make a hyper grid

# Specify n_features
n_features <- ncol(train_reg)-1

h2o_hyper_grid <- list(
  
  # Tune Split-variable randomization
  mtries = pmax(floor(n_features * c(.05, .15, .25, .333, .4)), 1),
  
  # Tune minimum number of observations in a leaf node 
  min_rows = c(1,3,5,10),
  
  # Tune  how many splits the tree can have
  max_depth = c(10,20,30),
  
  # Tune the  fraction of rows to sample for training each tree
  sample_rate = c(.5,.6,.7,.8)
)

## Make a search strategy grid
search_criteria <- list(
  strategy ="RandomDiscrete", # Randomly selects combinations of the hyper parameters from the grid
  
  stopping_metric = "auc", # Metric used to monitor the performance 
  
  stopping_tolerance = 0.001, # Improvement in MSE is less than 0.001, the model will stop
  
  stopping_rounds = 10, # Iterations without improvement before stopping the training process
  
  max_runtime_secs = 60*5 # Max runtime  5min
)

## Execute the grid

random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_random_grid",
  x = predictors,
  y = responce,
  training_frame = train_h2o,
  hyper_params = h2o_hyper_grid,     # h2o_hyper_grid defined earlier
  search_criteria = search_criteria, # Search criteria for the grid search
  ntrees = 10 * n_features,          # Number of trees in the forest
  stopping_metric = "auc",           # Stopping metric: AUC for binary classification
  stopping_rounds = 10,              # Stop after 10 rounds without improvement
  stopping_tolerance = 0.005
)

# Check the best results 
grid_result <- h2o.getGrid(grid_id = "rf_random_grid",
                           sort_by = "auc",
                           decreasing = FALSE)
# max_depth: 30.00000  
# min_rows: 1.00000
# mtries:3.00000     
# sample_rate:0.80000 
# This this the result is AUC:0.88513

## Re-run the model with more ntrees and importance metricts 

# Re-run Impurity based importance
ranger_impurity <- ranger(
  category_good ~ .,
  data = train_reg,
  max.depth = 30.00000,
  num.trees = 2000,
  mtry = 3,
  sample.fraction = 0.80000,
  importance = "impurity",
  min.node.size = 1
)

# Re-run Permutation based importance
ranger_permitaion <- ranger(
  category_good ~ .,
  data = train_reg,
  max.depth = 30.00000,
  num.trees = 2000,
  mtry = 3,
  sample.fraction = 0.8,
  importance = "permutation",
  min.node.size = 1
)

## Feature importance
imp_impurity <- vip(ranger_impurity,num_features = 11,bar = FALSE )
imp_permutation <- vip(ranger_permitaion,num_features = 11,bar = FALSE)

# Viz the importance
gridExtra::grid.arrange(imp_impurity,imp_permutation,nrow = 1)

## Note the top 3 important features are :
 # 1 density
 # 2 residual_sugar
 # 3 free_sulfur_dioxide

## Predictions 
random_forest_prediction <- predict(ranger_permitaion,test_reg)

# Convert the ranger.prediction into factor 
predicted_classes <- as.factor(random_forest_prediction$predictions)

## Create a confusion matrix
confusionMatrix(
  data = relevel(predicted_classes,ref = "bad"),
  reference = relevel(test_reg$category_good,ref = "bad")
)

h2o.removeAll()

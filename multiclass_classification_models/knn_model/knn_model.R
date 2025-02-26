#### Libraries ####

library(tidyverse)
library(caret)



#### KNN #### 

## Load the data 
train_reg <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/data/clean_data/train_reg.rds")

# Drop the category column
train_knn <- train_reg %>% select(-category_good)

# Assign new labels to the levels directly
levels(train_knn$quality) <- c("level_3", "level_4", "level_5", "level_6", "level_7", "level_8", "level_9")

#### Models ####

## Create a resampling method

cv <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary
)

## Create a hyperparameter grid search

# Define the frame of search
k_value <- round(sqrt(nrow(train_knn)))

# Create a grid
hyper_grid <- expand.grid(
  k = seq(1,k_value,by = 2))

## Fit Knn model and execute the grid search
knn_model <- train(
  quality ~.,
  data = train_knn,
  method = "knn",
  trControl = cv,
  tuneGrid = hyper_grid,
  metric = "Accuracy"
)

# Viz the K 
ggplot(knn_model)+
  theme_classic()+
  ggtitle("KNN Model Performance Across K Values")

## Save the model 
saveRDS(knn_model, "knn_model.rds") 


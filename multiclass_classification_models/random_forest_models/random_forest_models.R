
#### Libraries #####
library(tidyverse)
library(caret)
library(h2o)
library(ranger)

## Load the train set 
train_reg <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/data/clean_data/train_reg.rds")

## Remove the category column 
train_random_forest <- train_reg %>% select(-category_good)

#### Model the Random Forest 

# Define the number of featues for mtry hyperparameter 

n_feature <- length(setdiff(names(train_reg),"quality"))

## Train a default Random Forest 
random_forest_def <- ranger(
  formula = quality~.,
  data = train_random_forest,
  mtry = floor(n_feature/3)
)


## Train a tuned model 

# Create a hyperparameter Grid
hyper_grid_rf_1 <- expand.grid(
  mty = floor(n_feature * c(0.05,0.15,0.55,0.25,0.333,0.4)),
  min.node.size = c(1,3,5,10),
  replace = c(TRUE,FALSE),
  sample.fraction = c(.5,.6,.8)
)

# Execute the full grid search 
for (i in seq_len(nrow(hyper_grid_rf_1))) {
  
  
  # Fit the model 
  model_rf_tune <- ranger(
    
    # Model with all the features 
    formula = quality ~.,
    
    # train_random_forest data 
    data = train_random_forest,
    
    # n_featurs * 10 = 120
    num.trees = n_feature * 10 ,
    
    # mty to check (0.05,0.15,0.55,0.25,0.333,0.4)
    mtry = hyper_grid_rf_1$mty[i],
    
    # min.node.size to check  (1,3,5,10)
    min.node.size = hyper_grid_rf_1$min.node.size[i],
    
    # Replace check (True False)
    replace = hyper_grid_rf_1$replace[i],
    
    # Sample fraction to check (0.5,0.6,0.8)
    sample.fraction = hyper_grid_rf_1$sample.fraction[i]
  )
}















#### Libraries #####
library(tidyverse)
library(caret)
library(h2o)
library(ranger)

set.seed(123)

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
  mtry = floor(n_feature * c(0.05,0.15,0.55,0.25,0.333,0.4)),
  min.node.size = c(1,3,5,10),
  replace = c(TRUE,FALSE),
  sample.fraction = c(.5,.6,.8),
  accuracy = NA
)

# Execute the full grid search 
for (i in seq_len(nrow(hyper_grid_rf_1))) {
  
  # Fit the model 
  model_rf_tune <- ranger(
    
    # Model with all the features 
    formula = quality ~.,
    
    # train_random_forest data 
    data = train_random_forest,
    
    # number of featurs
    num.trees = 500 ,
    
    # mty  check (0.05,0.15,0.55,0.25,0.333,0.4)
    mtry = hyper_grid_rf_1$mty[i],
    
    # min.node.size check  (1,3,5,10)
    min.node.size = hyper_grid_rf_1$min.node.size[i],
    
    # Replace check (True False)
    replace = hyper_grid_rf_1$replace[i],
    
    # Sample fraction to check (0.5,0.6,0.8)
    sample.fraction = hyper_grid_rf_1$sample.fraction[i],
    
    # Store the Accuracy
    hyper_grid_rf_1$accuracy[i] <-1 - model_rf_tune$prediction.error,
    
    # Measure is the Gini index for classification,
    importance = "impurity"
  )
}

# Take the best parameters
best_params_rf_tune <- hyper_grid_rf_1 %>%
  arrange(desc(accuracy))%>% 
  head(1)

## Make a model with best_params_rf_tune and increase the number of trees
random_forest_model <- ranger(
  
  # Model with all the features 
  formula = quality ~.,
  data = train_random_forest,

  # Takes the tuned parameters and model with them 
  mtry =  best_params_rf_tune$mtry,
  min.node.size = best_params_rf_tune$min.node.size,
  replace = best_params_rf_tune$replace,
  sample.fraction = best_params_rf_tune$sample.fraction,
  
  # Increase the number of trees to have more stable results 
  num.trees = 2000
)

## Random Forest with h2o and Random Search grid

# Starting H2o
Sys.setenv(JAVA_HOME = "C:/Program Files/Eclipse Adoptium/jdk-11.0.26.4-hotspot")
h2o.init()

# Convert the training data into h2o object
train_h2o <- as.h2o(train_random_forest)

# Set the response
responce <- "quality"

# Set the predictors
predictors <- setdiff(colnames(train_random_forest),responce)


## Make a hyperparameter grid

h2o_grid <- list(
  
  # Fewest allowed observations in a leaf
  min_rows = c(1,3,5,10),
  
  # Row sample rate per tree 
  sample_rate = c(0.5,0.6,0.7,0.8),
  
  # Number of variables randomly sampled as candidates at each split.
  mtries = pmax(floor(n_feature * c(.05, .15, .25, .333, .4)), 1),
  
  # Maximum tree depth 
  max_depth = c(10,20,30)
)

## Random grid search strategy
search_criteria <- list(
  
  strategy = "RandomDiscrete",
  stopping_tolerance = 0.001, # Stop if the model does not improve < 0.1 %
  stopping_metric = "AUC",
  stopping_rounds = 10,       # Over the last 10 runs
  max_runtime_secs = 60 * 10  # Stop after 10 min 
)

## Execute the h2o grid
h2o_random_grid <- h2o.grid(
  
  # Sprecify to run RandomForest
  algorithm = "randomForest",
  
  # Define a Grid Id
  grid_id = "rf_random_grid",
  
  # Trained on train_h2o
  training_frame = train_h2o,
  
  # Set the responce and predictors
  x = predictors,
  y = responce,
  
  # Use the defined grid lists 
  hyper_params = h2o_grid,
  search_criteria = search_criteria,
  
  # Number of trees
  ntrees = 500,
  
  # AUC as the stopping metric
  stopping_metric = "AUC",
  stopping_rounds = 10,      # Stop if last 10 trees added
  stopping_tolerance = 0.005  # Stop if the model does not improve < 0.05 % of AUC
)
# Note
 # Problems with h2o cluster 

h2o.removeAll()

## Save the models

# Define the file path to save the model
save_path <- "C:/Users/Huawei/OneDrive/Wine-Quality-Classification"

# Save the models for later use and evaluation 
saveRDS(random_forest_def, file = file.path(save_path, "rf_model_default.rds"))
saveRDS(random_forest_model, file = file.path(save_path, "rf_model_tuned.rds"))




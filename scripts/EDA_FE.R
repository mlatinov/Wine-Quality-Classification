
#### Libraries ####

library(tidyverse)
library(corrplot)
library(recipes) # For feature engineering
library(caret) # For easy data spliting 

# Load the data 
wine_data <- read_csv("data/raw_data/wine_data.csv")

# Convert the quality score to factor 
wine_data$quality <- as.factor(wine_data$quality)

#### EDA ####

# Convert the wine_data from wide to long format 
wine_data_long <- wine_data %>%
  pivot_longer(cols = -c(quality), # Exclude the target variable quality
               names_to = "features",
               values_to = "value")


## Numerical features distributions
ggplot(data = wine_data_long,aes(x = value))+
  geom_density()+
  facet_wrap(~features,scales = "free")+
  theme_minimal()+
  theme(
    strip.text = element_text(size = 10,face = "bold"),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 10)
  )+
  # Add Title and label the axis
  labs(
    title = "Distribution of Wine Features",
    y = "Count",
    x = "Feature",
    caption = "Some features have highly skewed distributions"
  )

# Observations:
# - Some features have highly skewed distributions.
# - A few variables have long tails, which could indicate outliers.
# - Different scales suggest standardization or transformation (Yeo-Johnson or BoxCox or log) may help.
# 
# Notes:
# - Consider transformation for skewed variables.
# - Standardize numeric features before modeling.
# - Compare models trained on raw vs. preprocessed data to assess impact.

## Cheking for outliers via Box plot

ggplot(data = wine_data_long,aes(x = value))+
  geom_boxplot()+
  
  coord_flip()+ # flip the axis
  
  # Devide the plot for each feature 
  facet_wrap(~features,scales = "free")+
  
  # Add Title and label the axis
  theme_minimal()+
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 10)
  )+
  labs(
    title = "Box plot for Wine Features",
    x = "Value",
    y = "Feature",
    caption = "pH and alcohol have potential outliers"
  )
  
# Observations:
# -  The pH feature has a significant number of outliers

# Notes:
# - Test models with and without outlier removal to assess impact.

## Correlation Matrix to assess potential multicollinearity. 
wine_data %>%
  
  # Remove Quality
  select(-quality) %>%
  
  # Compute the correlations
  cor()%>%
  
  # Plot the Correlation Matrix
  corrplot::corrplot(method = "number",
                     order = "hclust",
                     tl.cex = 0.6,
                     number.cex = 0.7,
                     title = "Correlation Matrix Wine Features")

# Observations:
# - Many features show high correlation with each other, indicating multicollinearity.
# - Strong correlations may lead to redundant informationin the model, causing instability in regression-based models.
# - This could impact models like  logistic regression, making coefficients unreliable.

# Notes Future multicollinearity probelm:
# Potential Solutions:
# - Try regularization  (Ridge ,Lasso regression,Elastic net) to reduce the effect 
# - Use PCA to transform correlated features into independent PCA components.
# - Test tree-based models (Random Forest, GBMs) since they handle correlated features 

## Comparison of Wine Features by Quality 
wine_data %>%
  
  group_by(quality)%>%
  
  # Compute the median features stat for each level of wine 1-9
  summarise(
    alchol_levels = median(alcohol),
    chlorides_levels = median(chlorides),
    citric_acid_levels = median(citric_acid),
    density = median(density),
    pH_levels = median(pH),
    volatile_acidity_levels = median(volatile_acidity),
    free_sulfur_dioxide_levels = median(free_sulfur_dioxide),
    total_sulfur_dioxide = median(total_sulfur_dioxide),
    sugars = median(residual_sugar),
    density = median(density)
  )%>%
  
  # Transform the data wide to long 
  pivot_longer(-quality, names_to = "features", values_to = "median_value") %>%
  ggplot(aes(x  = quality,y = median_value,fill = features))+
  geom_col(show.legend = TRUE)+
  
  # Devide the graf for every feature
  facet_wrap(~ features, scales = "free_y") +
  
  # Add Title and label the axis
  theme_minimal()+
  labs(
    title = "Comparison of Wine Features by Quality ",
    x = "Wine Quality",
    y = "Median Feature Value"
  )+
  theme(
    strip.text = element_text(size = 8)
  )
  
# Notes:
# Some of the features (such as density, pH levels, alcohol_levels ),have similar median values
   # across all wine quality ratings. This suggests that they may be bad predictors for modeling wine quality

# Future Feature selection ig Lasso or other to asses the feature importance 

# Other features have similar median values for both lower-quality and higher-quality wines
 # (e.g., dividing wines into categories such as 3-6 for lower quality and 6-9 for higher quality). 
 # While summary statistics using median values does not fully capture the nuances or any interaction between the features,
 # any real differences between these groups are likely to be small.

## Violin plot to try differntiate between thr wine ratings 

ggplot(data = wine_data_long,aes(x = quality,y = value,fill = quality))+
  
  # Make a violin plot
  geom_violin()+
  
  # Devide the plot for each feature
  facet_wrap(~features,scales = "free_y")+
  
  # Add Title and label the axis
  theme_minimal()+
  labs(
    title = "Violin Plot of Wine Quality Features",
    x  = "Quality Score",
    y  = "",
  )+
  
  # Edit the size of the text 
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    
    # Remove the default legend from fill = quality
    legend.position = "none"
  )

## Boxplot between Wine Features across wine quality 
ggplot(data  = wine_data_long,aes(x = quality,y = value,fill = quality))+
  geom_boxplot()+
  facet_wrap(~features,scales = "free_y")+
  theme_minimal()+
  labs(
    title = "Boxplot Wine Features for each Category",
    x  = "Quality Score",
    y = ""
  )+
  theme(
    strip.text = element_text(size = 10),
    axis.title = element_text(size = 12),
    axis.text = element_text(size  = 10),
    legend.position = "none"
  )

## Convert the Quality score for factor wirt levels 1-9 to factor with leves good and bad
 # - bad wine when the score is below 6  (3,4,5,6)
 # - good wine when the score is above 6 (7,8,9)

# Add the new column
wine_data$category <- ifelse(test = as.numeric(as.character(wine_data$quality)) > 6,
                             yes = "good",
                             no = "bad")

# Convert the category into factor
wine_data$category <- as.factor(wine_data$category)

#### Feature Engineering ####

# Note : 
 # To prevent potential data leakage the data is split before we make transformation 

## Spit the data 

# Create a partition 
partition <- createDataPartition(wine_data$quality,p = 0.7 , list = FALSE)

# Create a training set with 70% of the data
train_set <- wine_data[partition,]

# Create a testing set with the remaining 30% of the data
testing_set <- wine_data[-partition,]

## Make a regular recipe blueprint 

blueprint_reg <- recipe(quality ~ .,data =train_set) %>%
  
  # Filter out near zero variance predictors
  step_nzv(all_predictors()) %>%
  
  # Normalize Numerical Skewness with Yeo Johnson Transformation
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Standardize all numerical predictors
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  
  # Hot-one dummy encoding
  step_dummy(all_of("category"))

## Preparation 
preparation_reg <- prep(blueprint_reg,train_set)
  
## Bake the first data to model with 

# Make a training set 
train_regular <- bake(preparation_reg,train_set)

# Make a testing set 
test_regular <- bake(preparation_reg,testing_set)

## Make a PCA blueprint 

# Make a blueprint with pca step 
blueprint_pca <- recipe(quality ~ .,data =train_set) %>%
  
  # Filter out near zero variance predictors
  step_nzv(all_predictors()) %>%
  
  # Normalize Numerical Skewness with Yeo Johnson Transformation
  step_YeoJohnson(all_numeric_predictors()) %>%
  
  # Standardize all numerical predictors
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  
  # PCA  
  step_pca(all_numeric_predictors()) %>%
  
  # Hot-one dummy encoding
  step_dummy(all_of("category"))

# Note:
 # The PCA is not tuned 

## Prepare
prepare_pca <- prep(blueprint_pca,train_set)

## Bake the Pca testing and training set
train_pca <- bake(prepare_pca,train_set)
test_pca <- bake(prepare_pca,testing_set)

## Save the data 

# Save clean data 
saveRDS(train_set,"train_set.rds")
saveRDS(testing_set,"testing_set.rds")

# Save the preprocessed data
saveRDS(train_regular,"train_reg.rds")
saveRDS(test_regular,"test_reg.rds")

# Save the PCA 
saveRDS(test_pca,"test_pca")
saveRDS(train_pca,"train_pca")



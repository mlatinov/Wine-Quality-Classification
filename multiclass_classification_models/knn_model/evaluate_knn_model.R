
#### Libraries ####
library(tidyverse)
library(caret)
library(pdp)
#### KNN Evaluation #####

# Load the KNN model 
knn_model <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/multiclass_classification_models/knn_model/knn_model.rds")

# Load the testing dataset
test_reg <- readRDS("C:/Users/Huawei/OneDrive/Wine-Quality-Classification/data/clean_data/test_reg.rds")

# Drop the category column
test_knn <- test_reg %>% select(-category_good)

# Assign new labels to the levels directly
levels(test_knn$quality) <- c("level_3", "level_4", "level_5", "level_6", "level_7", "level_8", "level_9")

## Predict wiht knn model
knn_prediction <- predict(knn_model,test_knn)
 
## Confusion Matrix
confusion_matrix<-confusionMatrix(knn_prediction,test_knn$quality)

## Note :
 # Accuracy : 0.5746  and the model does discriminate 
 # All clases are consistant

## Confusion matrix Heatmap

# Convert the confusion Matix into tidy format
conf_matrix_tidy <- as.data.frame(as.table(confusion_matrix))

# Plot the heatmap
ggplot(data = conf_matrix_tidy,aes(Prediction,Reference,fill = Freq))+
  geom_tile()+
  scale_fill_gradient(low = "white",high = "red")+
  theme_minimal()+
  labs(
    title = "Confusion Matrix Heatmap",
    x = "Predicted",
    y = "Actual")





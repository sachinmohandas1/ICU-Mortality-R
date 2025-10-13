# Sachin Mohandas - BU ID: U79542832
# MET CS555 A1 - Foundations of Machine Learning
# Term Project

library(ggplot2)
library(caret)
library(ROCR)
library(itertools)
library(GGally)
library(pROC)

setwd("G:/My Drive/BU MET/555 Foundations of Machine Learning/Term Paper")
df0 <- read.csv("mimic_iii_icu.csv")    # change this to file name listed below in write.csv line

# Preprocessing and saving data (replace NA's with the means of their respective columns)
for (col in names(df0)) {
  df0[[col]][is.na(df0[[col]])] <- mean(df0[[col]], na.rm = TRUE)
}

write.csv(df0, file = "Project_Sachin_Mohandas.csv", row.names = FALSE)

# Analysis on All Columns

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(df0$outcome, p = 0.5, list = FALSE)
training_data <- df0[trainIndex, !colnames(df0) %in% c("group", "ID")]
testing_data <- df0[-trainIndex, !colnames(df0) %in% c("group", "ID")]

# Train the logistic regression model
logistic_model0 <- train(outcome ~ ., data = training_data, method = "glm", family = "binomial")

# Make predictions on the testing set
predictions0 <- predict(logistic_model0, newdata = testing_data, type = "raw")
binary_predictions0 <- ifelse(predictions0 >= 0.5, 1, 0)

# Evaluate the model's accuracy
conf_matrix0 <- confusionMatrix(as.factor(as.integer(binary_predictions0)), 
                                as.factor(testing_data$outcome))
accuracy0 <- conf_matrix0$overall['Accuracy']
print(paste("Accuracy (all columns):", accuracy0))


# Analysis on Ensembles

# Selecting the relevant columns (all non-categorical columns)
cols <- c('age', 'outcome', 'BMI', 'heart.rate', 'Systolic.blood.pressure',
          'Diastolic.blood.pressure', 'Respiratory.rate', 'temperature', 
          'SP.O2', 'Urine.output', 'hematocrit', 'RBC', 'MCH', 'MCHC', 'MCV', 
          'RDW', 'Leucocyte', 'Platelets', 'Neutrophils', 'Basophils', 'Lymphocyte', 
          'PT', 'INR', 'NT.proBNP', 'Creatine.kinase', 'Creatinine', 'Urea.nitrogen', 
          'glucose', 'Blood.potassium', 'Blood.sodium', 'Blood.calcium', 'Chloride', 
          'Anion.gap', 'Magnesium.ion', 'PH', 'Bicarbonate', 'Lactic.acid', 'PCO2')

df1 <- df0[, cols]

# Dealing with missing values using the mean of each column
for (col in names(df1)) {
  df1[[col]][is.na(df1[[col]])] <- mean(df1[[col]], na.rm = TRUE)
}

df1$outcome <- as.integer(df1$outcome)

colsNoOutcome <- cols[!cols %in% 'outcome']

# Generate all possible combinations of 2 columns
allEnsembles <- combn(colsNoOutcome, 3, simplify = FALSE)

# List to store logistic regression models
logistic_models <- list()

# Vector to store accuracies
accuracies <- numeric(length(allEnsembles))

set.seed(123)
trainIndex1 <- createDataPartition(df1$outcome, p = 0.5, list = FALSE)

# Iterate through each combination of columns
for (i in seq_along(allEnsembles)) {
  ensemble <- allEnsembles[[i]]
  
  # Extract the columns for this combination
  trainingSubset <- df1[trainIndex1, c(ensemble, "outcome")]
  testingSubset <- df1[-trainIndex1, c(ensemble, "outcome")]
  
  # Train logistic regression model
  logistic_model <- train(outcome ~ ., data = trainingSubset, method = "glm", family = "binomial")
  
  # Store the model in the list
  logistic_models[[i]] <- logistic_model
  
  # Make predictions on the testing set
  predictions <- predict(logistic_model, newdata = testingSubset, type = "raw")
  binary_predictions <- ifelse(predictions >= 0.5, 1, 0)
  
  # Compute accuracy
  conf_matrix <- confusionMatrix(as.factor(as.integer(binary_predictions)), 
                                 as.factor(testingSubset$outcome))
  accuracy <- conf_matrix$overall['Accuracy']
  accuracies[i] <- accuracy
  print(i)          # Progress check
}

# Find how many ensembles are more accurate than the base model
betterEnsembles <- list()

for (index in seq_along(allEnsembles)) {
  ensemble <- allEnsembles[[index]]
  accuracy <- accuracies[index]
  
  # Check if accuracy is greater than target
  if (accuracy > accuracy0) {
    betterEnsembles[[length(betterEnsembles) + 1]] <- ensemble
  }
}

print(length(betterEnsembles))

# Sort accuracies in descending order and get the top
top10indices <- order(accuracies, decreasing = TRUE)[1:10]
top50indices <- order(accuracies, decreasing = TRUE)[1:50]

cat("Top 10 Most Accurate Ensembles:\n")
for (index in top10indices) {
  ensemble <- allEnsembles[[index]]
  accuracy <- accuracies[index]
  cat("Columns:", paste(ensemble, collapse = ", "), ", Accuracy:", accuracy, "\n")
}



# Time for some exploratory data analysis on the most accurate ensembles


# Create pairplots for each top ensemble and save them
for (index in top50indices) {
  ensemble <- allEnsembles[[index]]
  subset_df <- df1[, c(ensemble, "outcome")]
  g <- ggpairs(subset_df, columns = c(ensemble), mapping = aes(colour = factor(outcome))) +
    scale_color_manual(values = c("red", "blue"))
  ggsave(paste0("pairplot3_", index, ".png"), plot = g)
}

# Function to calculate collinearities for each ensemble
calculate_collinearities <- function(ensemble, data) {
  subset_data <- data[, ensemble]
  cor_matrix <- cor(subset_data)
  # Extract collinearities (upper triangle of correlation matrix)
  collinearities <- cor_matrix[upper.tri(cor_matrix)]
  return(collinearities)
}

# List to store collinearities for each ensemble
collinearities <- list()

# Calculate collinearities for each ensemble
for (i in seq_along(allEnsembles)) {
  ensemble <- allEnsembles[[i]]
  collinearity <- calculate_collinearities(ensemble, df1)
  collinearities[[i]] <- collinearity
}

allColls <- unlist(collinearities)

# Summary statistics of all collinearities
summaryAll <- summary(allColls)
sdAll <- sd(allColls)
print(summaryAll)
print(sdAll)

# Calculate collinearities of top 50 ensembles
collinearitiesOfTop <- list()

for (index in top50indices) {
  ensemble <- allEnsembles[[index]]
  collinearity <- calculate_collinearities(ensemble, df1)
  collinearitiesOfTop[[i]] <- collinearity
}

selectColls <- unlist(collinearitiesOfTop)

# Summary statistics of top ensemble collinearities
summarySelect <- summary(selectColls)
sdSelect <- sd(selectColls)
print(summarySelect)
print(sdSelect)

t_test <- t.test(allColls, selectColls)
print(t_test)

# Find coefficient magnitudes
for (index in top10indices) {
  logistic_model <- logistic_models[[index]]
  ensemble <- allEnsembles[[index]]
  coefficients <- coef(logistic_model$finalModel)
  cat("\n\nColumns:", paste(ensemble, collapse = ", "))
  cat("\nCoefficients:", coefficients[-1])
}






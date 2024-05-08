#!/usr/bin/env python
# coding: utf-8

# # Activity: Perform logistic regression 

# ## Introduction

# In this activity, you will complete an effective bionomial logistic regression. This exercise will help you better understand the value of using logistic regression to make predictions for a dependent variable based on one independent variable and help you build confidence in practicing logistic regression. Because logistic regression is leveraged across a wide array of industries, becoming proficient in this process will help you expand your skill set in a widely-applicable way.
# For this activity, you work as a consultant for an airline. The airline is interested in knowing if a better in-flight entertainment experience leads to higher customer satisfaction. They would like you to construct and evaluate a model that predicts whether a future customer would be satisfied with their services given previous customer feedback about their flight experience.
# The data for this activity is for a sample size of 129,880 customers. It includes data points such as class, flight distance, and in-flight entertainment, among others. Your goal will be to utilize a binomial logistic regression model to help the airline model and better understand this data.
# Because this activity uses a dataset from the industry, you will need to conduct basic EDA, data cleaning, and other manipulations to prepare the data for modeling.

# In this activity, you will practice the following skills:

# * Importing packages and loading data
# * Exploring the data and completing the cleaning process
# * Building a binomial logistic regression model 
# * Evaluating a binomial logistic regression model using a confusion matrix

# ## Step 1: Imports

# ### Import packages

### YOUR CODE HERE ###
import numpy as np
import pandas as pd

# Important imports for preprocessing, modeling, and evaluation.
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics


# Visualization package imports.
import matplotlib.pyplot as plt
import seaborn as sns


# ### Load the dataset
# RUN THIS CELL TO IMPORT YOUR DATA.
### YOUR CODE HERE ###
df_original = pd.read_csv("Invistico_Airline.csv")




# ### Output the first 10 rows
# Output the first 10 rows of data.
### YOUR CODE HERE ###
df_original.head(10)




# ## Step 2: Data exploration, data cleaning, and model preparation

# ### Prepare the data

# After loading the dataset, prepare the data to be suitable for a logistic regression model. This includes: 

# *   Exploring the data
# *   Checking for missing values
# *   Encoding the data
# *   Renaming a column
# *   Creating the training and testing data


# ### Explore the data
# Check the data type of each column. Note that logistic regression models expect numeric data.

### YOUR CODE HERE ###
df_original.dtypes


### YOUR CODE HERE ###
df_original['satisfaction'].value_counts(dropna = False)



# **Question:** How many satisfied and dissatisfied customers were there?
# There were 71,087 satisfied customers and 58,793 dissatisfied customers.

# **Question:** What percentage of customers were satisfied?
# 54.7% (71,087/129,880). (can compare to the final model)




# ### Check for missing values
# An assumption of logistic regression models is that there are no missing values. Check for missing values in the rows of the data.

### YOUR CODE HERE ###
df_original.isnull().sum()



# **Question:** Should you remove rows where the `Arrival Delay in Minutes` column has missing values, even though the airline is more interested in the `inflight entertainment` column?
# Although this column may not obviously, directly influence in-flight entertainments effect on satisfaction, it's good to have it somewhere incase it becomes a key focus for the company in the future.
# For this specific task, it can be removed as it is 393 values missing out of the larger 129,880. Although it's a small fraction, it may still negatively impact the model.


# ### Drop the rows with missing values
# Drop the rows with missing values and save the resulting pandas DataFrame in a variable named `df_subset`.

### YOUR CODE HERE ###
df_subset = df_original.dropna(axis=0).reset_index(drop = True)




# ### Prepare the data
# If you want to create a plot (`sns.regplot`) of your model to visualize results later in the notebook, the independent variable `Inflight entertainment` cannot be "of type int" and the dependent variable `satisfaction` cannot be "of type object." 
# Make the `Inflight entertainment` column "of type float."


### YOUR CODE HERE ###
df_subset = df_subset.astype({"Inflight entertainment": float})




# ### Convert the categorical column `satisfaction` into numeric
# Convert the categorical column `satisfaction` into numeric through one-hot encoding.


### YOUR CODE HERE ###
df_subset['satisfaction'] = OneHotEncoder(drop='first')
fit_transform(df_subset[['satisfaction']]).toarray()





# ### Output the first 10 rows of `df_subset`
# To examine what one-hot encoding did to the DataFrame, output the first 10 rows of `df_subset`.

### YOUR CODE HERE ###
df_subset.head(10)





# ### Create the training and testing data
# Put 70% of the data into a training set and the remaining 30% into a testing set. Create an X and y DataFrame with only the necessary variables.

### YOUR CODE HERE ###
X = df_subset[["Inflight entertainment"]]
y = df_subset["satisfaction"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3,random_state=42)




# **Question:** If you want to consider customer satisfaction with your model, should you train your model to use `inflight entertainment` as your sole independent variable? 
# The question is based around customer satisfaction. Although we are focused on in-flight entertainment, it would be advisable to conduct further analysis in comparison so using a singular variable is not the best course of action.


# ## Step 3: Model building

# ### Fit a LogisticRegression model to the data
# Build a logistic regression model and fit the model to the training data. 


### YOUR CODE HERE ###
clf = LogisticRegression().fit(X_train,y_train)




# ### Obtain parameter estimates
# Make sure you output the two parameters from your model. 

### YOUR CODE HERE ###
clf.coef_



### YOUR CODE HERE ###
clf.intercept_




# ### Create a plot of your model
# Create a plot of your model to visualize results using the seaborn package.

### YOUR CODE HERE ###
sns.regplot(x="Inflight entertainment", y="satisfaction", data=df_subset,logistic=True, ci=None)




# **Question:** What can you tell from the graph?
# The higher the in-flight entertainment, the higher the satisfaction level. Because the in-flight entertainment is categorical, further analysis is needed for data points.



# ## Step 4. Results and evaluation
# ### Predict the outcome for the test dataset
# Now that you've completed your regression, review and analyze your results. First, input the holdout dataset into the `predict` function to get the predicted labels from the model. Save these predictions as a variable called `y_pred`.




### YOUR CODE HERE ###
y_pred = clf.predict(X_test)

print(y_pred)


# Use predict_proba to output a probability.
### YOUR CODE HERE ###
clf.predict_proba(X_test)




# Use predict to output 0's and 1's.
### YOUR CODE HERE ###
clf.predict(X_test)




# ### Analyze the results
# Print out the model's accuracy, precision, recall, and F1 score.

### YOUR CODE HERE ###
print("Accuracy:", "%.6f" % metrics.accuracy_score(y_test, y_pred))
print("Precision:", "%.6f" % metrics.precision_score(y_test, y_pred))
print("Recall:", "%.6f" % metrics.recall_score(y_test, y_pred))
print("F1 Score:", "%.6f" % metrics.f1_score(y_test, y_pred))




# ### Produce a confusion matrix
# Data professionals often like to know the types of errors made by an algorithm. To obtain this information, produce a confusion matrix.

### YOUR CODE HERE ###
cm = metrics.confusion_matrix(y_test, y_pred, labels = clf.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix = cm,display_labels = clf.classes_)
disp.plot()



# **Question:** What stands out to you about the confusion matrix?
# Two of the quadrants are under 4,000, which are quite low numbers. They are possibly false positives and false negatives. (The remaining quadrants are above 13,000)




# **Question:** Did you notice any difference in the number of false positives or false negatives that the model produced?
# The difference is not very large.

# **Question:** What do you think could be done to improve model performance?
# Going back to what I previously mentioned about using more that one independent variable. I think factoring in other variables that are likely to impact satisfaction can generate better insights.




# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged. 

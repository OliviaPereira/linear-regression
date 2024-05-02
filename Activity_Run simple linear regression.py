#!/usr/bin/env python
# coding: utf-8

# # Activity: Run simple linear regression

# ## **Introduction**

# As you're learning, simple linear regression is a way to model the relationship between two variables. By assessing the direction and magnitude of a relationship, data professionals are able to uncover patterns and transform large amounts of data into valuable knowledge. This enables them to make better predictions and decisions. 
# In this lab, you are part of an analytics team that provides insights about your company's sales and marketing practices. You have been assigned to a project that focuses on the use of influencer marketing. For this task, you will explore the relationship between your radio promotion budget and your sales.
# The dataset provided includes information about marketing campaigns across TV, radio, and social media, as well as how much revenue in sales was generated from these campaigns. Based on this information, company leaders will make decisions about where to focus future marketing resources. Therefore, it is critical to provide them with a clear understanding of the relationship between types of marketing campaigns and the revenue generated as a result of this investment.


# ## **Step 1: Imports**

# Import relevant Python libraries and modules.
### YOUR CODE HERE ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols import statsmodels.api as sm

# The dataset provided is a .csv file (named `marketing_sales_data.csv`), which contains information about marketing conducted in collaboration with influencers, along with corresponding sales. Assume that the numerical variables in the data are expressed in millions of dollars. As shown in this cell, the dataset has been automatically loaded in for you. You do not need to download the .csv file, or provide more code, in order to access the dataset and proceed with this lab. Please continue with this activity by completing the following instructions.
# **Note:** This is a fictional dataset that was created for educational purposes and modified for this lab.

# RUN THIS CELL TO IMPORT YOUR DATA.
### YOUR CODE HERE ###
data = pd.read_csv("marketing_sales_data.csv")





# ## **Step 2: Data exploration** 

# To get a sense of what the data includes, display the first 10 rows of the data.




# Display the first 10 rows of the data.

### YOUR CODE HERE ###
data.head(10)




# **Question:** What do you observe about the different variables included in the data?
# How much money is spent on TV, radio, and social media advertising
# The type of influencer used in the promotion, categorized by their number of followers
# The amount of sales generated from the promotion

# Next, to get a sense of the size of the dataset, identify the number of rows and the number of columns.

# Display number of rows, number of columns.

### YOUR CODE HERE ###
data.shape





# **Question:** How many rows and columns exist in the data?

# 572 rows, 5 columns

# Now, check for missing values in the rows of the data. This is important because missing values are not that meaningful when modeling the relationship between two variables. To do so, begin by getting Booleans that indicate whether each value in the data is missing. Then, check both columns and rows for missing values.

# Start with .isna() to get booleans indicating whether each value in the data is missing.

### YOUR CODE HERE ###
data.isna()



# Use .any(axis=1) to get booleans indicating whether there are any missing values along the columns in each row.

### YOUR CODE HERE ###
data.isna().any(axis=1)





# Use .sum() to get the number of rows that contain missing values.

### YOUR CODE HERE ###
data.isna().any(axis=1).sum()



# **Question:** How many rows containing missing values?
# 3

# Next, drop the rows that contain missing values. Data cleaning makes your data more usable for analysis and regression. Then, check to make sure that the resulting data does not contain any rows with missing values.



# Use .dropna(axis=0) to indicate that you want rows which contain missing values to be dropped. To update the DataFrame, reassign it to the result.

### YOUR CODE HERE ###
data = data.dropna(axis=0)




# Start with .isna() to get booleans indicating whether each value in the data is missing.
# Use .any(axis=1) to get booleans indicating whether there are any missing values along the columns in each row.
# Use .sum() to get the number of rows that contain missing values

### YOUR CODE HERE ###
data.isna().any(axis=1).sum()

# The next step for this task is checking model assumptions. To explore the relationship between radio promotion budget and sales, model the relationship using linear regression. Begin by confirming whether the model assumptions for linear regression can be made in this context.
# **Note:** Some of the assumptions can be addressed before the model is built. These will be addressed in this section. After the model is built, you will finish checking the assumptions.
# Create a plot of pairwise relationships in the data. This will help you visualize the relationships and check model assumptions.




# Create plot of pairwise relationships.

### YOUR CODE HERE ###
sns.pairplot(data)



# **Question:** Is the assumption of linearity met?
# The scatter plot shows that as spending on radio advertising increases, sales also tend to increase.
# The points on the plot form a line, indicating a positive relationship between radio spending and sales.
# This suggests that the assumption of a straight-line relationship between radio spending and sales is valid.




# ## **Step 3: Model building** 

# Select only the columns that are needed for the model.

# Select relevant columns.
# Save resulting DataFrame in a separate variable to prepare for regression.

### YOUR CODE HERE ###
ols_data = data[["Radio", "Sales"]]



# Now, display the first 10 rows of the new DataFrame to better understand the data.

### YOUR CODE HERE ###
ols_data.head(10)




# Next, write the linear regression formula for modeling the relationship between the two variables of interest.

# Write the linear regression formula.
# Save it in a variable.

### YOUR CODE HERE ###
ols_formula = "Sales ~ Radio"




# Now, implement the ordinary least squares (OLS) approach for linear regression.

# Implement OLS.

### YOUR CODE HERE ###
OLS = ols(formula = ols_formula, data = ols_data)




# Now, create a linear regression model for the data and fit the model to the data.

# Fit the model to the data.
# Save the fitted model in a variable.

### YOUR CODE HERE ###
model = OLS.fit()



# ## **Step 4: Results and evaluation** 

# Begin by getting a summary of the results from the model.

# Get summary of results.

### YOUR CODE HERE ###
model.summary()




# Next, analyze the bottom table from the results summary. Based on the table, identify the coefficients that the model determined would generate the line of best fit. The coefficients are the y-intercept and the slope. 

# **Question:** What is the y-intercept?
# 41.5326

# **Question:** What is the slope?
# 8.1733

# **Question:** What linear equation would you write to express the relationship between sales and radio promotion budget? Use the form of y = slope * x + y-intercept? 
# y = 8.1733 * radio promotion budget + 41.5326



# **Question:** What does the slope mean in this context?
# On average, companies that invest an additional million dollars in radio promotion tend to earn about 8.1733 million dollars more in sales.

# Now that you've built the linear regression model and fit it to the data, finish checking the model assumptions. This will help confirm your findings. First, plot the OLS data with the best fit regression line.



# Plot the OLS data with the best fit regression line.

### YOUR CODE HERE ###
sns.regplot(x = "Radio", y = "Sales", data = ols_data)



# **Question:** What do you observe from the preceding regression plot?
# It shows a linear relationship.

# Now, check the normality assumption. Get the residuals from the model.

### YOUR CODE HERE ###
residuals = model.resid




# Now, visualize the distribution of the residuals.

# Visualize the distribution of the residuals.

### YOUR CODE HERE ###
fig = sns.histplot(residuals)
fig.set_xlabel("Residual Value")
fig.set_title("Histogram of Residuals")
plt.show()





# **Question:** Based on the visualization, what do you observe about the distribution of the residuals?
# The residuals seem normal.

# Next, create a Q-Q plot to confirm the assumption of normality.
# Create a Q-Q plot.

### YOUR CODE HERE ###
sm.qqplot(residuals, line='s')
plt.title("Q-Q plot of Residuals")
plt.show()




# **Question:** Is the assumption of normality met?
# Because of the straight line from bottom left to top right, it's clear to assume normality is met.

# Now, check the assumptions of independent observation and homoscedasticity. Start by getting the fitted values from the model.

# Get fitted values.

### YOUR CODE HERE ###
fitted_values = model.predict(ols_data["Radio"])




# Next, create a scatterplot of the residuals against the fitted values.

# Create a scatterplot of residuals against fitted values.

### YOUR CODE HERE ###
fig = sns.scatterplot(x=fitted_values, y=residuals)
fig.axhline(0)
fig.set_xlabel("Fitted Values")
fig.set_ylabel("Residuals")
plt.show()




# **Question:** Are the assumptions of independent observation and homoscedasticity met?
# The cloud like shape and randomly spaced residuals show the assumptions of independent observation and homoscedasticity.



# ## **Considerations**

# **What are some key takeaways that you learned during this lab?**
# Data visualizations and exploring the data can help us see if using a straight line to represent the relationship between two things makes sense.
# The output from a linear regression model helps us describe how those two things are related.

# **How would you present your findings from this lab to others?**
# In simple terms, the linear regression model tells us that for every extra million dollars a company spends on radio promotion, they can expect their sales to increase by around 8.1733 million dollars on average.
# The small p-value of 0.000 means there's a very low chance of seeing such extreme data if there was no real relationship between radio promotion budget and sales.
# So, we can confidently say there is indeed a connection between radio promotion spending and sales.
# However, the exact increase in sales might vary slightly within a range, which we estimate to be between 7.791 million and 8.555 million dollars with 95% certainty.

# **What summary would you provide to stakeholders?**
# Based on the data we have and the analysis we've done, it's clear that spending more money on radio promotion is linked to higher sales for the companies in our dataset.
# When a company increases its radio promotion budget by 1 million dollars, they can expect their sales to go up by around 8.1733 million dollars, on average.
# So, it's a good idea to keep investing in radio promotion. It might also be helpful to explore this relationship further, especially in different industries or when promoting different types of products or services.
# Gathering more data could give us a better understanding of how these factors interact.

# **References**
# [Pandas.DataFrame.Any — Pandas 1.4.3 Documentation.](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.any.html)
# [Pandas.DataFrame.Isna — Pandas 1.4.3 Documentation.](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html)
# [Pandas.Series.Sum — Pandas 1.4.3 Documentation.](https://pandas.pydata.org/docs/reference/api/pandas.Series.sum.html)
# [Saragih, H.S. *Dummy Marketing and Sales Data*.](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data)

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.

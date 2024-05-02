#!/usr/bin/env python
# coding: utf-8

# # Activity: Evaluate simple linear regression

# ## Introduction

# In this activity, you will use simple linear regression to explore the relationship between two continuous variables. To accomplish this, you will perform a complete simple linear regression analysis, which includes creating and fitting a model, checking model assumptions, analyzing model performance, interpreting model coefficients, and communicating results to stakeholders.
# For this activity, you are part of an analytics team that provides insights about marketing and sales. You have been assigned to a project that focuses on the use of influencer marketing, and you would like to explore the relationship between marketing promotional budgets and sales. The dataset provided includes information about marketing campaigns across TV, radio, and social media, as well as how much revenue in sales was generated from these campaigns. Based on this information, leaders in your company will make decisions about where to focus future marketing efforts, so it is critical to have a clear understanding of the relationship between the different types of marketing and the revenue they generate.
# This activity will develop your knowledge of linear regression and your skills evaluating regression results which will help prepare you for modeling to provide business recommendations in the future.

# ## Step 1: Imports

# ### Import packages

# Import relevant Python libraries and packages. In this activity, you will need to use `pandas`, `pyplot` from `matplotlib`, and `seaborn`.


# Import pandas, pyplot from matplotlib, and seaborn.

### YOUR CODE HERE ###
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ### Import the statsmodel module and the ols function
# Import the `statsmodels.api` Python module using its common abbreviation, `sm`, along with the `ols()` function from `statsmodels.formula.api`. To complete this, you will need to write the imports as well.



# Import the statsmodel module.

# Import the ols function from statsmodels.

### YOUR CODE HERE ###
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ### Load the dataset
# RUN THIS CELL TO IMPORT YOUR DATA. 

### YOUR CODE HERE ###
data = pd.read_csv('marketing_and_sales_data_evaluate_lr.csv')

# Display the first five rows.

### YOUR CODE HERE ### 
data.head()

# ## Step 2: Data exploration
# ### Familiarize yourself with the data's features
# Start with an exploratory data analysis to familiarize yourself with the data and prepare it for modeling.

# The features in the data are:
# * TV promotion budget (in millions of dollars)
# * Social media promotion budget (in millions of dollars)
# * Radio promotion budget (in millions of dollars)
# * Sales (in millions of dollars)

# Each row corresponds to an independent marketing promotion where the business invests in `TV`, `Social_Media`, and `Radio` promotions to increase `Sales`.

# The business would like to determine which feature most strongly predicts `Sales` so they have a better understanding of what promotions they should invest in in the future. To accomplish this, you'll construct a simple linear regression model that predicts sales using a single independent variable. 

# **Question:** What are some reasons for conducting an EDA before constructing a simple linear regression model?
# It helps us see what information we have in the dataset.
# We can check how the data is spread out, like what's the smallest, average, and largest values.
# By plotting graphs, we can see how different factors relate to each other and pick the best one to focus on.
# EDA also helps us spot any mistakes or missing information in the data, like typos or empty spaces.

# ### Explore the data size

# Calculate the number of rows and columns in the data.

# Display the shape of the data as a tuple (rows, columns).

### YOUR CODE HERE ### 
data.shape()



# ### Explore the independent variables

# There are three continuous independent variables: `TV`, `Radio`, and `Social_Media`. To understand how heavily the business invests in each promotion type, use `describe()` to generate descriptive statistics for these three variables.

# Generate descriptive statistics about TV, Radio, and Social_Media.

### YOUR CODE HERE ###
data[['TV','Radio','Social_Media']].describe()




# ### Explore the dependent variable

# Before fitting the model, ensure the `Sales` for each promotion (i.e., row) is present. If the `Sales` in a row is missing, that row isn't of much value to the simple linear regression model.
# Display the percentage of missing values in the `Sales` column in the DataFrame `data`.


# Calculate the average missing rate in the sales column.
### YOUR CODE HERE ###
missing_sales = data.Sales.isna().mean()

# Convert the missing_sales from a decimal to a percentage and round to 2 decimal place.
### YOUR CODE HERE ###
missing_sales = round(missing_sales*100, 2)

# Display the results (missing_sales must be converted to a string to be concatenated in the print statement).
### YOUR CODE HERE ###
print('Percentage of promotions missing Sales: ' +  str(missing_sales) + '%')

# **Question:** What do you observe about the percentage of missing values in the `Sales` column?
# 0.13% of rows are missing the Sales value.

# ### Remove the missing data
# Remove all rows in the data from which `Sales` is missing.

# Subset the data to include rows where Sales is present.
### YOUR CODE HERE ### 
data = data.dropna(subset = ['Sales'], axis = 0)







# ### Visualize the sales distribution
# Create a histogram to visualize the distribution of `Sales`.

# Create a histogram of the Sales.
### YOUR CODE HERE ### 
fig = sns.histplot(data['Sales'])

# Add a title
fig.set_title('Distribution of Sales');







# **Question:** What do you observe about the distribution of `Sales` from the preceding histogram?
# They are equally distributed between 25 and 350 million.


# ## Step 3: Model building
# Create a pairplot to visualize the relationships between pairs of variables in the data. You will use this to visually determine which variable has the strongest linear relationship with `Sales`. This will help you select the X variable for the simple linear regression.


# Create a pairplot of the data.
### YOUR CODE HERE ###
sns.pairplot(data);







# **Question:** Which variable did you select for X? Why?
# TV clearly has the strongest linear relationship with Sales.
# You could draw a straight line through the scatterplot of TV and Sales that confidently estimates Sales using TV.
# Radio and Sales appear to have a linear relationship, but there is larger variance than between TV and Sales.

# ### Build and fit the model
# Replace the comment with the correct code. Use the variable you chose for `X` for building the model.

# Define the OLS formula.
### YOUR CODE HERE ### 
ols_formula = 'Sales ~ TV'

# Create an OLS model.
### YOUR CODE HERE ### 
OLS = ols(formula = ols_formula, data = data)

# Fit the model.
### YOUR CODE HERE ### 
model = OLS.fit()

# Save the results summary.
### YOUR CODE HERE ###
model_results = model.summary()

# Display the model results.
### YOUR CODE HERE ### 
model_results






# ### Check model assumptions
# To justify using simple linear regression, check that the four linear regression assumptions are not violated. These assumptions are:

# * Linearity
# * Independent Observations
# * Normality
# * Homoscedasticity

# ### Model assumption: Linearity
# The linearity assumption requires a linear relationship between the independent and dependent variables. Check this assumption by creating a scatterplot comparing the independent variable with the dependent variable. 

# Create a scatterplot comparing the X variable you selected with the dependent variable.

# Create a scatterplot comparing X and Sales (Y).

### YOUR CODE HERE ### 
sns.scatterplot(x = data['TV'], y = data['Sales']);







# **QUESTION:** Is the linearity assumption met?
# Yes, there is a linear relationship between TV and Sales.

# ### Model assumption: Independence
# The **independent observation assumption** states that each observation in the dataset is independent. As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# ### Model assumption: Normality

# The normality assumption states that the errors are normally distributed.
# Create two plots to check this assumption:
## * **Plot 1**: Histogram of the residuals
# * **Plot 2**: Q-Q plot of the residuals




# Calculate the residuals.
### YOUR CODE HERE ### 
residuals = model.resid

# Create a 1x2 plot figures.
fig, axes = plt.subplots(1, 2, figsize = (8,4))

# Create a histogram with the residuals.
sns.histplot(residuals, ax=axes[0])

# Set the x label of the residual plot.
axes[0].set_xlabel("Residual Value")

# Set the title of the residual plot.
axes[0].set_title("Histogram of Residuals")

# Create a Q-Q plot of the residuals.
### YOUR CODE HERE ### 
sm.qqplot(residuals, line='s',ax = axes[1])

# Set the title of the Q-Q plot.
axes[1].set_title("Normal Q-Q plot")

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()

# Show the plot.
plt.show()







# **Question:** Is the normality assumption met?
# Yes it is met.


# ### Model assumption: Homoscedasticity
# The **homoscedasticity (constant variance) assumption** is that the residuals have a constant variance for all values of `X`.

# Check that this assumption is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.




# Create a scatterplot with the fitted values from the model and the residuals.

### YOUR CODE HERE ### 
fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x-axis label.
fig.set_xlabel("Fitted Values")

# Set the y-axis label.
fig.set_ylabel("Residuals")

# Set the title.
fig.set_title("Fitted Values v. Residuals")

# Add a line at y = 0 to visualize the variance of residuals above and below 0.
### YOUR CODE HERE ### 
fig.axhline(0)

# Show the plot.
plt.show()







# **QUESTION:** Is the homoscedasticity assumption met?
# The assumption of homoscedasticity is met.



# ## Step 4: Results and evaluation

# ### Display the OLS regression results

# If the linearity assumptions are met, you can interpret the model results accurately.
# Display the OLS regression results from the fitted model object, which includes information about the dataset, model fit, and coefficients.

# Display the model_results defined previously.

### YOUR CODE HERE ###
model_results

# **Question:** The R-squared on the preceding output measures the proportion of variation in the dependent variable (Y) explained by the independent variable (X). What is your intepretation of the model's R-squared?
# The R-squared value is dependent on the variable selected for X.
# When using TV is represented by X, it shows 99.9% of the differences we see in sales.



# ### Interpret the model results
# With the model fit evaluated, assess the coefficient estimates and the uncertainty of these estimates.

# **Question:** Based on the preceding model results, what do you observe about the coefficients?
# If I use TV as the independent variable X, the coefficient for the Intercept is -0.1263 and the coefficient for TV is 3.5614.

# **Question:** How would you write the relationship between X and `Sales` in the form of a linear equation?
# Y = Intercept + Slope * X
# Sales (in millions) = Intercept + Slope * TV (in millions)
# Sales (in millions) = −0.1263 + 3.5614 * TV (in millions)

# **Question:** Why is it important to interpret the beta coefficients?
# Interpreting beta coefficients is important because they provide insights into the relationship between the independent variables and the dependent variable in a regression model.
# They tell us how much the dependent variable is expected to change when the independent variable changes by one unit, holding all other variables constant.
# This helps us understand the relative impact of each independent variable on the outcome, allowing for better understanding and decision-making based on the model's results.


# ### Measure the uncertainty of the coefficient estimates

# Model coefficients are estimated. This means there is an amount of uncertainty in the estimate. A p-value and $95\%$ confidence interval are provided with each coefficient to quantify the uncertainty for that coefficient estimate.
# Display the model results again.

# Display the model_results defined previously.

### YOUR CODE HERE ###
model_results

# **Question:** Based on this model, what is your interpretation of the p-value and confidence interval for the coefficient estimate of X?
# When TV spending is considered, it shows a very low p-value (0.000) and a narrow confidence interval ([3.558, 3.565]).
# This suggests that the estimated effect of TV advertising on sales is highly certain.
# In simpler terms, the business can trust that TV advertising has a significant impact on sales.

# **Question:** Based on this model, what are you interested in exploring?
# See the difference from different independent values, for example, TV or Radio.
# How different TV promotions could change the business's sales.


# **Question:** What recommendations would you make to the leadership at your organization?
# Based on this analysis, increasing spending on TV advertising would likely lead to the highest increase in sales compared to radio and social media advertising.
# Specifically, for every additional million dollars invested in TV advertising, it's estimated an increase of about 3.56 million dollars in sales.
# This estimate is highly reliable, indicating a strong relationship between TV advertising and sales.
# Therefore, it is recommended to focus more on TV advertising to boost sales effectively.


# ## Considerations

# **What are some key takeaways that you learned from this lab?**
# EDA helps find the right variable for predicting outcomes in a simple linear regression.
# Before trusting the model, make sure it meets basic assumptions.
# R-squared tells us how well the model predicts outcomes.
# Always include measures like p-values and confidence intervals to show how certain we are about our predictions.

# **What findings would you share with others?**
# Sales are spread out between $25 and $350 million for all promotion types.
# TV shows the strongest connection with sales, while radio has a decent one but with more variation. Social media has the weakest link.
# When TV spending increases, it almost perfectly predicts sales with an R-squared of 0.999.
# With TV as the main factor, every additional million spent on TV ads is linked to around $3.56 million more in sales. This link is very certain with a p-value of 0.000 and a confidence interval between $3.558 and $3.565 million.

# **How would you frame your findings to stakeholders?**
# TV has the strongest positive connection with sales compared to social media and radio, this is evidenced by the model showing that almost all changes in sales are because of changes in the TV promotional budget.
# For every additional million dollars spent on TV ads, sales are expected to go up by about $3.56 million.
# We are highly confident in this estimate because the interval (between $3.558 million and $3.565 million) is very likely to contain the actual increase in sales for a one million dollar increase in the TV promotional budget.


# #### **References**
# Saragih, H.S. (2020). [*Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data).
# Dale, D.,Droettboom, M., Firing, E., Hunter, J. (n.d.). [*Matplotlib.Pyplot.Axline — Matplotlib 3.5.0 Documentation*](https://matplotlib.org/3.5.0/api/_as_gen/matplotlib.pyplot.axline.html).

# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.

#!/usr/bin/env python
# coding: utf-8

# # Activity: Perform multiple linear regression

# ## Introduction
# As you have learned, multiple linear regression helps you estimate the linear relationship between one continuous dependent variable and two or more independent variables. For data science professionals, this is a useful skill because it allows you to compare more than one variable to the variable you're measuring against. This provides the opportunity for much more thorough and flexible analysis. 
# For this activity, you will be analyzing a small business' historical marketing promotion data. Each row corresponds to an independent marketing promotion where their business uses TV, social media, radio, and influencer promotions to increase sales. They previously had you work on finding a single variable that predicts sales, and now they are hoping to expand this analysis to include other variables that can help them target their marketing efforts.
# To address the business' request, you will conduct a multiple linear regression analysis to estimate sales from a combination of independent variables. This will include:

# * Exploring and cleaning data
# * Using plots and descriptive statistics to select the independent variables
# * Creating a fitting multiple linear regression model
# * Checking model assumptions
# * Interpreting model outputs and communicating the results to non-technical stakeholders



# ## Step 1: Imports
# ### Import packages

# Import relevant Python libraries and modules.

# Import libraries and modules.
### YOUR CODE HERE ### 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols


# ### Load dataset
# RUN THIS CELL TO IMPORT YOUR DATA.
### YOUR CODE HERE ### 
data = pd.read_csv('marketing_sales_data.csv')



# Display the first five rows.

### YOUR CODE HERE ### 
data.head(5)



# ## Step 2: Data exploration

# ### Familiarize yourself with the data's features
# Start with an exploratory data analysis to familiarize yourself with the data and prepare it for modeling.
# The features in the data are:

# * TV promotional budget (in "Low," "Medium," and "High" categories)
# * Social media promotional budget (in millions of dollars)
# * Radio promotional budget (in millions of dollars)
# * Sales (in millions of dollars)
# * Influencer size (in "Mega," "Macro," "Micro," and "Nano" categories)


# **Question:** What are some purposes of EDA before constructing a multiple linear regression model?
# Understanding Relationships: Before building the model, we explore how different factors relate to each other.
# Checking Assumptions: We ensure our data follows certain guidelines required for accurate modeling.
# Detecting Outliers and Missing Values: We identify and address any unusual or missing data points.
# Feature Selection: We choose the most important data to include in our model for accurate predictions.
# Data Cleaning: We tidy up our data, fixing errors and organizing it properly before starting the analysis.


# ### Create a pairplot of the data

# Create a pairplot to visualize the relationship between the continous variables in `data`.
### YOUR CODE HERE ###
sns.pairplot(data);






# **Question:** Which variables have a linear relationship with `Sales`? Why are some variables in the data excluded from the preceding plot?
# They both do but Radio has the most obvious linear relationship. This means they can both be used as independent variables.
# TV and Influencer are excluded from the plot as they are not numeric.



# ### Calculate the mean sales for each categorical variable

# There are two categorical variables: `TV` and `Influencer`. To characterize the relationship between the categorical variables and `Sales`, find the mean `Sales` for each category in `TV` and the mean `Sales` for each category in `Influencer`. 




# Calculate the mean sales for each TV category.
### YOUR CODE HERE ### 
print(data.groupby('TV')['Sales'].mean())


# Calculate the mean sales for each Influencer category.
### YOUR CODE HERE ### 
print(data.groupby('Influencer')['Sales'].mean())







# **Question:** What do you notice about the categorical variables? Could they be useful predictors of `Sales`?
# TV has the potential to be a predictor of sales as the High TV promotions are significantly higher.
# Influencer is a not a good predictor as the averages are not varied enough.
# but we are always able to do further exploration.


# ### Remove missing data
# This dataset contains rows with missing values. To correct this, drop all rows that contain missing data.

# Drop rows that contain missing data and update the DataFrame.
### YOUR CODE HERE ### 
data = data.dropna(axis=0)







# ### Clean column names
# The `ols()` function doesn't run when variable names contain a space. Check that the column names in `data` do not contain spaces and fix them, if needed.

# Rename all columns in data that contain a space.
### YOUR CODE HERE ### 
data = data.rename(columns={'Social Media': 'Social_Media'})






# ## Step 3: Model building

# ### Fit a multiple linear regression model that predicts sales
# Using the independent variables of your choice, fit a multiple linear regression model that predicts `Sales` using two or more independent variables from `data`.




# Define the OLS formula.
### YOUR CODE HERE ### 
ols_formula = 'Sales ~ C(TV) + Radio'

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







# **Question:** Which independent variables did you choose for the model, and why?
# TV was selected because of the strong relationship between TV promotional budget and Sales average.
# Radio was also chosen because it had a strong relationship with Sales.



# ### Check model assumptions

# For multiple linear regression, there is an additional assumption added to the four simple linear regression assumptions: **multicollinearity**.
# Check that all five multiple linear regression assumptions are upheld for your model.

# ### Model assumption: Linearity


# Create scatterplots comparing the continuous independent variable(s) you selected previously with `Sales` to check the linearity assumption. Use the pairplot you created earlier to verify the linearity assumption or create new scatterplots comparing the variables of interest.

# Create a 1x2 plot figure.
fig, axes = plt.subplots(1, 2, figsize = (8,4))

# Create a scatterplot between Radio and Sales.
sns.scatterplot(x = data['Radio'], y = data['Sales'],ax=axes[0])

# Set the title of the first plot.
axes[0].set_title("Radio and Sales")

# Create a scatterplot between Social Media and Sales.
sns.scatterplot(x = data['Social_Media'], y = data['Sales'],ax=axes[1])

# Set the title of the second plot.
axes[1].set_title("Social Media and Sales")

# Set the xlabel of the second plot.
axes[1].set_xlabel("Social Media")

# Use matplotlib's tight_layout() function to add space between plots for a cleaner appearance.
plt.tight_layout()


# **Question:** Is the linearity assumption met?
# Radio and Sales have a strong linear relationship.
# Social Media and Sales also have a linear relationship but it is not as strong as Radio's.



# ### Model assumption: Independence

# The **independent observation assumption** states that each observation in the dataset is independent. As each marketing promotion (i.e., row) is independent from one another, the independence assumption is not violated.

# ### Model assumption: Normality

# Create the following plots to check the **normality assumption**:
# * **Plot 1**: Histogram of the residuals
# * **Plot 2**: Q-Q plot of the residuals



# Calculate the residuals.
### YOUR CODE HERE ### 
residuals = model.resid

# Create a histogram with the residuals.
### YOUR CODE HERE ### 
fig, axes = plt.subplots(1, 2, figsize = (8,4))

sns.histplot(residuals, ax=axes[0])

axes[0].set_xlabel("Residual Value")

axes[0].set_title("Histogram of Residuals")

# Create a Q-Q plot of the residuals.
### YOUR CODE HERE ### 
sm.qqplot(residuals, line='s',ax = axes[1])

axes[1].set_title("Normal QQ Plot")

plt.tight_layout()

plt.show()







# **Question:** Is the normality assumption met?
# Both the histogram and QQ plot show normalcy is met.



# ### Model assumption: Constant variance
# Check that the **constant variance assumption** is not violated by creating a scatterplot with the fitted values and residuals. Add a line at $y = 0$ to visualize the variance of residuals above and below $y = 0$.



# Create a scatterplot with the fitted values from the model and the residuals.
### YOUR CODE HERE ### 
fig = sns.scatterplot(x = model.fittedvalues, y = model.resid)

# Set the x axis label.
fig.set_xlabel("Fitted Values")

# Set the y axis label.
fig.set_ylabel("Residuals")

# Set the title.
fig.set_title("Fitted Values v. Residuals")


# Add a line at y = 0 to visualize the variance of residuals above and below 0.
### YOUR CODE HERE ### 
fig.axhline(0)

plt.show()








# **Question:** Is the constant variance assumption met?
# TV is the dominating factor, so the fitted values are in three groups but the variance is similarly distributed, showing the assumption is met.



# ### Model assumption: No multicollinearity

# The **no multicollinearity assumption** states that no two independent variables ($X_i$ and $X_j$) can be highly correlated with each other. 

# Two common ways to check for multicollinearity are to:

# * Create scatterplots to show the relationship between pairs of independent variables
# * Use the variance inflation factor to detect multicollinearity

# Use one of these two methods to check your model's no multicollinearity assumption.




# Create a pairplot of the data.
### YOUR CODE HERE ### 
sns.pairplot(data)





# Calculate the variance inflation factor (optional).
### YOUR CODE HERE ###

# Import variance_inflation_factor from statsmodels.
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Create a subset of the data with the continous independent variables.
X = data[['Radio','Social_Media']]

# Calculate the variance inflation factor for each variable.
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# Create a DataFrame with the VIF results for the column names in X.
df_vif = pd.DataFrame(vif, index=X.columns, columns = ['VIF'])

# Display the VIF results.
df_vif







# **Question 8:** Is the no multicollinearity assumption met?
#  The model only has a singular continuous independent variable so there are no multicollinearity issues.
# If we include both Radio and Social Media as predictors in the model, we see that they're somewhat related to each other. This connection violates an assumption we make in the model. Also, when we include both in the model, we find that they both show a high level of multicollinearity, meaning they're strongly related to each other.



# ## Step 4: Results and evaluation

# ### Display the OLS regression results
# If the model assumptions are met, you can interpret the model results accurately.

# First, display the OLS regression results.
# Display the model results summary.
### YOUR CODE HERE ### 
model_results

# **Question:** What is your interpretation of the model's R-squared?
# Using TV and Radio as the independent variables results in R2 = 0.904. This means, the model explains 90.4% of the variation in Sales.




# ### Interpret model coefficients

# With the model fit evaluated, you can look at the coefficient estimates and the uncertainty of these estimates.
# Again, display the OLS regression results.

# Display the model results summary.
### YOUR CODE HERE ### 
model_results

# **Question:** What are the model coefficients?
# Because TV and Radio are used to predict the sales, the model coefficients are:
# β0 = 218.5261
# βT V Low = −154.2971
# βTVMedium = −75.3120
# βRadio = 2.9669



# **Question:** How would you write the relationship between `Sales` and the independent variables as a linear equation?
# HINT: Sales=β0 +β1 ∗X1 +β2 ∗X2 +β3 ∗X3
# β0 + βTV Low ∗ XTV Low + βTV Medium ∗ XTV Medium + βRadio ∗ XRadio
# Sales = 218.5261 − 154.2971 ∗ XT V Low − 75.3120 ∗ XT V M edium + 2.9669 ∗ XRadio




# **Question:** What is your intepretation of the coefficient estimates? Are the coefficients statistically significant?
# The model indicates that when comparing TV promotion levels, the average sales are lower for Medium or Low TV promotions compared to High TV promotions, if the Radio promotion level remains the same.
# For example, the model predicts that sales for a Low TV promotion are about $154,297 lower on average compared to a High TV promotion when the Radio promotion level is the same.
# Additionally, the positive coefficient for Radio confirms that there's a positive relationship between Radio promotion and sales, as observed in our earlier analysis.
# All coefficients in the model are statistically significant, meaning they have a strong impact on sales, and their estimated effects are reliable.




# **Question:** Why is it important to interpret the beta coefficients?
# Interpreting the beta coefficients helps understand how each independent variable influences the dependent variable in a regression model.



# **Question:** What are you interested in exploring based on your model?
#  New plots for further analysis, ones that may help me get the findings across better.



# **Question:** Do you think your model could be improved? Why or why not? How?
# Considering how well TV predicts sales, we could make the model better by looking at TV promotions in more detail, like different budget levels or types of ads.
# Also, adding more factors like when the ads run could help the model make even better predictions.





# #### **References**

# Saragih, H.S. (2020). [*Dummy Marketing and Sales Data*](https://www.kaggle.com/datasets/harrimansaragih/dummy-advertising-and-sales-data).


# **Congratulations!** You've completed this lab. However, you may not notice a green check mark next to this item on Coursera's platform. Please continue your progress regardless of the check mark. Just click on the "save" icon at the top of this notebook to ensure your work has been logged.

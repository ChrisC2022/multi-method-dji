#!/usr/bin/env python
# coding: utf-8

# Github Repo: https://github.com/ChrisC2022/multi-method-dji

# ## Identifying a Dataset

# In[ ]:


"""
Downloading a dataset that contains weekly data for the Dow Jones Industrial Index.
Citation:
Brown,Michael. (2014). Dow Jones Index. UCI Machine Learning Repository. https://doi.org/10.24432/C5788V.
@misc{misc_dow_jones_index_312,
  author       = {Brown,Michael},
  title        = {{Dow Jones Index}},
  year         = {2014},
  howpublished = {UCI Machine Learning Repository},
  note         = {{DOI}: https://doi.org/10.24432/C5788V}
}
"""


# ## Identifying the Topic and Goal
# 
# **Project Topic:** Exploring weekly data from the Dow Jones to establish relationships between changes in stock prices and other inputs.
# 
# **Project Goal:** The goal of the project is to find the best of a selected set of supervised learning methods to predict the percentage change in a stock's price from one week to the next.
# 
# **Data (Source and Description):** Weekly data from the Dow Jones Industrial Index. 
# 
# Citation:
# 
# Brown,Michael. (2014). Dow Jones Index. UCI Machine Learning Repository. https://doi.org/10.24432/C5788V.
# 
# @misc{misc_dow_jones_index_312,
# 
#   author       = {Brown,Michael},
#   
#   title        = {{Dow Jones Index}},
#   
#   year         = {2014},
#   
#   howpublished = {UCI Machine Learning Repository},
#   
#   note         = {{DOI}: https://doi.org/10.24432/C5788V}
#   
# }
# 
# The data was downloaded here: https://archive.ics.uci.edu/dataset/312/dow+jones+index
# 
# This dataset has $750$ observations and $16$ features.
# 
# This response variable was explicitly identified by the authors/contributors of the dataset as:
# *percent_change_next_weeks_price*.

# ## Cleaning and EDA

# In[ ]:


#Checking the location of the local directory for my Jupyter notebook to download data in the proper place
import os
os.getcwd()


# In[3]:


#Importing necessary packages
import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split


# In[4]:


file_path = "~\\dow_jones_index.data"
#Based on a quick visual inspection through notepad, this file appears to be comma-delimited
dji_data = pd.read_csv(file_path, delimiter = ",")


# **Data Cleaning**

# ***Inspecting the Data***

# In[5]:


#Checking out the top few rows of data to see how everything loaded
dji_data.head()


# In[6]:


dji_data.info()


# In[9]:


print('Earliest financial quarter:', dji_data.quarter.min(), ', and latest quarter:', dji_data.quarter.max())


# In[10]:


#Checking the number of observations in each quarter to ensure I'll have a balance for training and testing
print(dji_data.groupby('quarter')['stock'].count())


# ### Priorities for Initial Cleaning and Organization
# 
# This dataset has $750$ observations and $16$ variables. The response variable for this model
# (what we want to test) is *percent_change_next_weeks_price*; that leaves $15$ observations.
# As seasons might play a role, I'm going to follow a procedure I've seen followed elsewhere in financial/economic returns
# and treat the quarters as distinct training/testing units. I'll use quarter $1$ for training and quarter $2$ for testing.
# 
# Note: I'm uncomfortable with having a balance in favor of testing. The reason I'm pursuing this split anyway is because I would assume financial trends to flow forward, i.e., the effect of variables should establish themselves in time and either strengthen or diminish accordingly. It doesn't make intuitive sense to me to use the quarter with the higher number of observations (quarter $2$) to train the model and then test the model on an earlier time period.
# 
# Using the quarters to differentiate training vs. testing sets leaves us with $14$ additional dependent variables about which to make decisions. I'll work through those now.

# *Data type* ***munging*** is necessary for 'stock', 'date', and the financial columns. I'll perform that now.

# In[11]:


#Changing the data types.
#'stock' should be a factor.
#'date' should be a datetype.
#'open', 'high', 'low', 'close', 'next_weeks_open', and 'next_weeks_close' should be numeric.

dji_data['stock'] = dji_data['stock'].astype('category')
dji_data['date'] = pd.to_datetime(dji_data['date'])

#Need to remove the '$' to convert the columns with currency values to numeric
dji_data[dji_data.columns[3:7]] = dji_data[dji_data.columns[3:7]].replace('[\$,]', '', regex = True).astype(float)
dji_data[dji_data.columns[11:13]] = dji_data[dji_data.columns[11:13]].replace('[\$,]', '', regex = True).astype(float)

dji_data.info()


# In[12]:


#Investigating the data frame further to determine spot missing values and other analytical challenges
dji_data.describe()


# In[13]:


#No obvious issues turned up using the summary statistics above; checking through for any missing/NA values to be sure
dji_data.isnull().any()


# In[14]:


#2 columns have NAs, so I'm going to look at them to decide what to do with them
dji_data.isna()


# In[15]:


same_obs = sum((dji_data[dji_data['percent_change_volume_over_last_wk'].isna()].index != 
     dji_data[dji_data['previous_weeks_volume'].isna()].index))
if same_obs == 0:
    obs_with_NA = dji_data[dji_data['percent_change_volume_over_last_wk'].isna()].index.to_list()
    print('The same observations have NAs in both columns:', '\n',obs_with_NA)


# How to treat these? As recent trends are widely believed to be reflected in asset values, I don't think they can be ignored.
# Neither do I think they should be imputed as I don't have a solid methodological/theoretical basis to 'assume'
# any particular values.
# I'm going to eliminate these values from the training data so as not to soften the value of recent volume.
# These values are clearly dependent on one another, so it makes sense to remove both.

# In[16]:


#Removing NA observations
dji_data.drop(obs_with_NA, axis = 0, inplace = True)


# In[17]:


dji_data.info()


# ### Exploratory Data Analysis: Visualizations
# 
# Time series data have intriguing issues related to correlation. I'm going to look for some of the most correlated variables.

# In[18]:


#Looking at correlated variables and potential collinearity:
dji_data.corr()


# ***Supplementary visualizations***

# In[19]:


#To make this a bit clearer, I'm going to use a heatmap:
plt.subplots(figsize=(20,20))
sns.heatmap(dji_data.corr(), vmin = -1, vmax = 1, annot=True)


# So that helped a bit; a number of variables are completely correlated, probably dependent, and can likely be omitted.
# Looking at the heatmap as a guide, and after examining what each variable stands for, many variables can be eliminated;
# for example, most high-low values expressed elsewhere as '% change', and 'open' and 'close' are often the same (though
# separated by a trading cycle.)
# 
# Keeping the following variables: *'quarter', 'stock', 'date', 'open', 'percent_change_price', 'percent_change_volume_over_last_wk',
# 'days_to_next_dividend',* and *'percent_return_next_dividend'*.

# In[20]:


dji_red = (dji_data.drop(['high', 'low', 'close', 'volume', 'previous_weeks_volume', 'next_weeks_open', 'next_weeks_close'],
                        axis = 1))


# In[21]:


#Looking at the correlations (or lack thereof) between the remaining variables is encouraging:
dji_red.corr()


# In[22]:


#Now to create the training and testing sets:
df_train = dji_red[dji_red['quarter'] == 1]
df_test = dji_red[dji_red['quarter'] == 2]


# ## Models
# 
# ### Model 1: MLR with forward-stepwise refinement
# 
# Now that the data is ready for analysis, I'll try to find the best method for modeling the changes in a stock's price from one week to the next. These methods are going to target regression as an outcome; as such, it makes sense to start with some simpler supervised-learning regression methods and build upon that effectiveness. I'll begin with *multiple linear regression with forward-stepwise refinement*.  

# In[23]:


#Listing the variables to test:
var_lst = ['date', 'stock', 'open', 'percent_change_price', 'percent_change_volume_over_last_wk', 'days_to_next_dividend',
           'percent_return_next_dividend']


# Using $R^2$ as a performance metric, checking to see which of the remaining predictor variables could be used for the model.

# In[24]:


top_pred = ['',0]
for pred in var_lst:
    mod  = smf.ols(formula='percent_change_next_weeks_price~' + pred, data=df_train).fit()
    print(pred, mod.rsquared)
    if mod.rsquared>top_pred[1]:
        top_pred = [pred, mod.rsquared]
print('top predictor:',top_pred)


# **Analysis of Model 1:**
# 
# None of those did well; the $R^2$ of $0.207$ for 'date' and $0.096$ for 'stock' are low. Furthermore, it only shows that the top-performing predictor variables are the dates and the stock itself; in the case of the stock, this means we can ditch most of these other predictors and focus on a company's fundamentals/financials/performance as well as the timing. And maybe that's true. But rather than continue down this path, I'd like to test some alternative models; I'll move on from MLR for now.

# ### Model 2: Random-Forest Regression
# 
# While random forest models can be used for classification problems, they can also be used for regression. As they are reputed for their effectiveness in modeling noisy, multi-featured data, it makes sense to try random-forest regression for this modeling problem.
# 
# I'll facilitate this with an **additional layer of data munging** by transforming the date to a categorical variable; quarter and stock can be dropped as well for this problem. I'll define X and y as the set of predictor variables and the response variable, respectively.

# In[25]:


df_train['date'] = df_train['date'].astype('category')

X = df_train.drop(['quarter', 'stock', 'percent_change_next_weeks_price'], axis = 1)
y = df_train['percent_change_next_weeks_price']


# In[26]:


#Exploring X to make sure the data types are usable for random-forest regression:
X.info()


# In[27]:


#Let's see how many estimators will be needed to improve the fit on the training data.
#Using root mean squared error (RMSE) to evaluate the random forest predictions
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
rmse_vals = []
for i in range(1, 41, 1):
    reg = RandomForestRegressor(n_estimators = i, random_state = 42)
    reg.fit(X, y)
    ytrain_pred = reg.predict(X)
    rmse_met = mean_squared_error(ytrain_pred, y, squared = False)
    rmse_vals.append(rmse_met)


# In[28]:


plt.plot(range(1, 41, 1), rmse_vals)
plt.title('RMSE by Number of Estimators')
plt.xlabel('Estimators')
plt.ylabel('RMSE')


# In[29]:


#Probably best around k = 21, as it is less complicated than higher values of k, while improvement slows down after that
reg_final = RandomForestRegressor(n_estimators = 21, random_state = 42)
reg_final.fit(X, y)
ytrain_pred = reg.predict(X)
mean_squared_error(ytrain_pred, y, squared = False)


# In[30]:


#And to have some sense of what those errors would look like, 
#I'm plotting the values here of the predicted values and the actual training values:
plt.plot(ytrain_pred, y, 'bo')
plt.title('Random Forest Predicted Values vs. Training Values')
plt.xlabel('Predicted')
plt.ylabel('Actual Training')


# In[31]:


#Now let's see how it performs on the test data:
df_test['date'] = df_test['date'].astype('category')
X_test = df_test.drop(['quarter', 'stock', 'percent_change_next_weeks_price'], axis = 1)
y_test = df_test['percent_change_next_weeks_price']

ytest_pred = reg_final.predict(X_test)
rmse_test = mean_squared_error(ytest_pred, y_test, squared = False)
rmse_test


# In[32]:


plt.plot(ytest_pred, y_test, 'go')
plt.title('Random Forest Predicted Values vs. Test Values')
plt.xlabel('Predicted')
plt.ylabel('Actual Test')


# **Analysis of Model 2:**
# 
# Random-forest regression didn't perform as well as I'd hoped. It fit the training data considerably better than the testing data, which suggests the influence of other variables—perhaps uncaptured—which influenced the movement in stock prices.
# 
# What about pivoting on the problem by re-considering the *essence* of our goal? If the idea of predicting stock returns is to guide investment decisions, then couldn't a binary yes/no response to the question of *whether* to buy a stock be useful? In other words, if the effectiveness of regression modeling methods is doubtful here, then what about reframing the problem as a *classification* problem?
# 
# This implies the target of predicting whether a stock will move up or down. As most investors buy baskets of stocks rather than particular ones, perhaps understanding which to include in a purchase would make better sense and enable the reduction/dispersion of risk. So one way to look at this is as a problem where we're trying to predict whether the return will be positive or negative; I can create another variable to enable this analysis.

# In[33]:


df_train['next_weeks_direction'] = (df_train['percent_change_next_weeks_price'] > 0).astype('int')
df_test['next_weeks_direction'] = (df_test['percent_change_next_weeks_price'] > 0).astype('int')


# In[34]:


df_test['next_weeks_direction']


# ## Model 3: Logistic Regression for Binary Classification
# 
# Logistic regression can be used to model binary responses (e.g., positive/negative price changes) using multiple predictor variables. To handle issues of dependence and collinearity, I'll drop the variable upon which the direction of the price change is based: *'percent_change_next_weeks_price'*.

# In[35]:


#Trying binary classification via logistic regression to model the outcome: positive or negative price changes.
#Because 'percent_change_next_weeks_price' defines 'next_weeks_direction' it is necessary to drop that variable
X1 = df_train.drop(['quarter', 'stock', 'percent_change_next_weeks_price'], axis = 1)
y1 = df_train['next_weeks_direction']

log_reg = LogisticRegression(solver = 'liblinear').fit(X1, y1)


# In[36]:


log_reg.coef_


# In[37]:


logreg_pred = log_reg.predict(X1)


# In[38]:


#Checking to see how well this model did with the training data
print('accuracy is:', accuracy_score(y1, logreg_pred))
print('recall is:', recall_score(y1, logreg_pred))
print('precision is:', precision_score(y1, logreg_pred))


# In[39]:


#Now applying to the test data:
X1_test = df_test.drop(['quarter', 'stock', 'percent_change_next_weeks_price'], axis = 1)
X1_test['date'] = X1_test['date'].astype('category')
y1_test = df_test['next_weeks_direction']

logreg_test_pred = log_reg.predict(X1_test)
print('accuracy is:', accuracy_score(y1_test, logreg_test_pred))
print('recall is:', recall_score(y1_test, logreg_test_pred))
print('precision is:', precision_score(y1_test, logreg_test_pred))


# **Analysis of Model 3:**
# 
# The above was not much better. But what if the quarter is an important part of the data?
# Trying these 3 methods after re-training the data to allow the quarters to be a variable; and increasing the size
# of the training set.

# In[40]:


df_train, df_test = train_test_split(dji_red, test_size = 0.2, random_state = 42)


# ### Model 1, Refined Approach

# In[41]:


#MLR with forward-stepwise refinement:
var_lst = ['quarter', 'date', 'stock', 'open', 'percent_change_price', 'percent_change_volume_over_last_wk', 'days_to_next_dividend',
           'percent_return_next_dividend']

#Examining R^2:
top_pred = ['',0]
for pred in var_lst:
    mod  = smf.ols(formula='percent_change_next_weeks_price~' + pred, data=df_train).fit()
    print(pred, mod.rsquared)
    if mod.rsquared>top_pred[1]:
        top_pred = [pred, mod.rsquared]
print('top predictor:',top_pred)


# **Analysis of Model 1, Refined Approach:**
# 
# That's a little better, although other than 'date' there still aren't any convincing predictors.

# ## Model 2, Refined Approach: Random-Forest Regression with a Larger Training Set

# In[42]:


df_train['date'] = df_train['date'].astype('category')
X = df_train.drop(['stock', 'percent_change_next_weeks_price'], axis = 1)
y = df_train['percent_change_next_weeks_price']


# In[43]:


rmse_vals = []
for i in range(1, 41, 1):
    reg = RandomForestRegressor(n_estimators = i, random_state = 42)
    reg.fit(X, y)
    ytrain_pred = reg.predict(X)
    rmse_met = mean_squared_error(ytrain_pred, y, squared = False)
    rmse_vals.append(rmse_met)

plt.plot(range(1, 41, 1), rmse_vals)
plt.title('RMSE by Number of Estimators')
plt.xlabel('Estimators')
plt.ylabel('RMSE')


# In[44]:


min(rmse_vals)


# **Analysis of Model 2, Refined Approach:**
# 
# Still not much better; no real need to test it as the goal was to find an improvement, which didn't happen with the training data.

# ### Model 2, Classification Approach
# 
# Random forests can also be used for classification; perhaps with the classification version of this problem, a random forest model might yield benefits?

# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


df_train['next_weeks_direction'] = (df_train['percent_change_next_weeks_price'] > 0).astype('int')
df_test['next_weeks_direction'] = (df_test['percent_change_next_weeks_price'] > 0).astype('int')


# In[ ]:


#Changing the date variable to 'category' to enable analysis


# In[74]:


X_test['date'] = X_test['date'].astype('category')


# In[72]:


#choosing 100 learners, which is the default argument

X_train = df_train.drop(['stock', 'percent_change_next_weeks_price', 'next_weeks_direction'], axis = 1)
y_train = df_train['next_weeks_direction']

X_test = df_test.drop(['stock', 'percent_change_next_weeks_price', 'next_weeks_direction'], axis = 1)
y_test = df_test['next_weeks_direction']

rf_class = RandomForestClassifier()
rf_class.fit(X_train, y_train)
ytrain_pred = rf_class.predict(X_train)
print('accuracy is:', accuracy_score(ytrain_pred, y_train))
print('recall is:', recall_score(ytrain_pred, y_train))
print('precision is:', precision_score(ytrain_pred, y_train))


# This looks encouraging; I'll try it on the test data.

# In[76]:


ytest_pred = rf_class.predict(X_test)
print('accuracy is:', accuracy_score(ytest_pred, y_test))
print('recall is:', recall_score(ytest_pred, y_test))
print('precision is:', precision_score(ytest_pred, y_test))


# **Analysis of Model 3, Classification Approach:**
# 
# That certainly represented an improvement. The high **variance** and low **bias** of the decision trees was balanced by the ensemble that the random-forest method enabled, providing an improvement in accuracy, recall, and precision. So far this is the best approach for modeling the data. I'd like to test the logistic regression model next (with the new balance of training and testing data) to see which method is best.

# In[77]:


X2 = df_train.drop(['stock', 'percent_change_next_weeks_price', 'next_weeks_direction'], axis = 1)
y2 = df_train['next_weeks_direction']

log_reg = LogisticRegression(solver = 'liblinear').fit(X2, y2)


# In[78]:


logreg_pred2 = log_reg.predict(X2)


# In[79]:


#Checking with training data
print('accuracy is:', accuracy_score(y2, logreg_pred2))
print('recall is:', recall_score(y2, logreg_pred2))
print('precision is:', precision_score(y2, logreg_pred2))


# **Analysis of Model 3, Refined Approach:**
# 
# Logistic regression was inferior to random-forest classification.

# ## Conclusions:
# 
# This analysis examined multiple methods for modeling stock prices. Much effort has been devoted to modeling and predicting stock prices and financial returns; an entire industry has grown for the purpose of analyzing company fundamentals, forecasting economic headwinds, mitigating risk, and optimizing predictive strategies and tactics. While the analysis conducted here is by no means exhaustive—and seems to call for additional time-series modeling strategies—it did nevertheless yield interesting insight into the inherent noisiness and difficulty of predicting stock returns.
# 
# Multiple methods were used for both classification and regression. Regression strategies included multiple linear regression with forward stepwise selection (to iterate over the most effective variables for prediction) and random-forest regression. Neither method was especially successful after the first pass; a second pass involved retraining the models on a larger training set that did not differentiate between quarters and therefore potentially captured some of the temporal effects of macroeconomic quarterly trends. The analysis was better after the data were divided a second time into training and testing sets, although the models were still not convincing.
# 
# This led to a modification of the intended strategy. Precise return targets are not always required when risk is spread across a basket of stocks; in this case, picking ones that will increase or decrease may be useful. This suggested the possible effectiveness of a classification strategy with binary outcomes (i.e., increase vs. no-increase/decrease). The models used for this included random-forest classification and logistic regression. Model performance improved (for both training and testing sets) when the size of the training set was increased and the size of the test set was reduced. The best and most effective model was random-forest classification, which performed excellently on the training data and had higher accuracy, recall, and precision scores on the reevaluated test set. The scores were not as high as I had hoped, however, from which I drew the following conclusions (and intend to implement on different data sets in future):
# 
# 1) For better performance, specific time-series approaches may be required. The methods here attempted to account for timing periods in some way, but more effective methods can be employed;
# 
# 2) In the case of market returns, the company itself—plus the period in which it reports its fundamentals—may be vastly more important than the variables captured here in determining returns during the following week (i.e., directional movement, up or down). The salience of this fact is reflected in the establishment of an entire industry to evaluate company fundamentals and performance;
# 
# 3) The market may be inherently noisy to the extent that accuracy, recall, and precision are strongly limited under even the best cases.
# 
# 
# It is worth noting that the authors of the original data set ended up using *percent_change_price, percent_change_volume_over_last_wk, days_to_next_dividend,* and *percent_return_next_dividend* for their model. My goal was to find the best combination of variables for the different models I sought to test; combinations of the variables used by the authors of the data set also turned out to be among the most valuable variables in the models that I implemented for this analysis, which suggests their fundamental importance to asset prices and returns.

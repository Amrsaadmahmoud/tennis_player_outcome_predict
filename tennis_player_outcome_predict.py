
# coding: utf-8

# # Project Goals

# ***i will create a linear regression model that predicts the outcome for a tennis player based on their playing habits. By analyzing and modeling the Association of Tennis Professionals (ATP) data,
# i will determine what it takes to be one of the best tennis players in the world***

# # about the data

# **The ATP men’s tennis dataset includes a wide array of tennis statistics, which are described below:
# 
# Identifying Data
# 
# Player: name of the tennis player
#     
# Year: year data was recorded
#     
# Service Game Columns (Offensive)
# 
# Aces: number of serves by the player where the receiver does not touch the ball
#     
# DoubleFaults: number of times player missed both first and second serve attempts
#     
# FirstServe: % of first-serve attempts made
#     
# FirstServePointsWon: % of first-serve attempt points won by the player
#     
# SecondServePointsWon: % of second-serve attempt points won by the player
#     
# BreakPointsFaced: number of times where the receiver could have won service game of the player
#     
# BreakPointsSaved: % of the time the player was able to stop the receiver from winning service game when they had the chance
# ServiceGamesPlayed: total number of games where the player served
#     
# ServiceGamesWon: total number of games where the player served and won
#     
# TotalServicePointsWon: % of points in games where the player served that they won
#     
# Return Game Columns (Defensive)
# 
# FirstServeReturnPointsWon: % of opponents first-serve points the player was able to win
#     
# SecondServeReturnPointsWon: % of opponents second-serve points the player was able to win
#     
# BreakPointsOpportunities: number of times where the player could have won the service game of the opponent
# BreakPointsConverted: % of the time the player was able to win their opponent’s service game when they had the chance
# ReturnGamesPlayed: total number of games where the player’s opponent served
#     
# ReturnGamesWon: total number of games where the player’s opponent served and the player won
#     
# ReturnPointsWon: total number of points where the player’s opponent served and the player won
#     
# TotalPointsWon: % of points won by the player
# 
# Outcomes
# 
# Wins: number of matches won in a year
#     
# Losses: number of matches lost in a year
#     
# Winnings: total winnings in USD($) in a year
#     
# Ranking: ranking at the end of year**

# In[1]:

#import lib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:

#upload the data
df=pd.read_csv('C:/Users/amr/Downloads/code_academy _course/tennis_ace_starting/tennis_stats.csv')


# In[3]:

#show the data
df.head(7)


# # Assessing Data

# In[4]:

df.shape


# In[6]:

df.info()


# In[8]:

df.duplicated().sum()


# In[9]:

df.isnull().sum()


# In[10]:

df.describe()


# # exploratory analysis

# **Aces && Wins**

# In[16]:

#Aces && Wins
plt.scatter(df['Aces'],df['Wins'])
plt.xlabel('Aces')
plt.ylabel('Wins')
plt.title('relationship between the Aces feature and the Wins')
plt.show()


# **BreakPointsOpportunities && Winnings**

# In[17]:

plt.scatter(df['BreakPointsOpportunities'],df['Winnings'])
plt.xlabel('BreakPointsOpportunities')
plt.ylabel('Winnings')
plt.title('relationship between the BreakPointsOpportunities feature and the Winnings')
plt.show()


# **ServiceGamesWon && Winnings**

# In[18]:

plt.scatter(df['ServiceGamesWon'],df['Winnings'])
plt.xlabel('ServiceGamesWon')
plt.ylabel('Winnings')
plt.title('relationship between the ServiceGamesWon feature and the Winnings')
plt.show()


# # build a single feature linear regression model on the data.

# **i used 'FirstServeReturnPointsWon' as our feature and Winnings as our outcome.**

# In[28]:

#split the data
features=df[['FirstServeReturnPointsWon']]
outcomes=df[['Winnings']]


# In[29]:

#split the data into training and testing
features_train,features_test,outcomes_train,outcomes_test=train_test_split(features,outcomes,train_size=0.8,test_size=0.2)


# In[30]:

#created a linear regression model 
mlr=LinearRegression()
mlr.fit(features_train,outcomes_train)


# ** predicted outcome based on our model**

# In[31]:

outcome_predict=mlr.predict(features_test)


# In[32]:

plt.scatter(outcome_test,outcome_predict,alpha=0.4)
plt.show()


# # Create a  linear regression models that use multiple features to predict yearly earnings.

# In[33]:

two_feature=df[['BreakPointsOpportunities','FirstServeReturnPointsWon']]
new_outcom=df[['Winnings']]


# In[34]:

two_feature_train,two_feature_test,new_outcom_train,new_outcom_test=train_test_split(two_feature,new_outcom,train_size=0.8)


# In[35]:

model=LinearRegression()
model.fit(two_feature_train,new_outcom_train)


# In[36]:

new_outcom_predict=model.predict(two_feature_test)


# In[37]:

plt.scatter(new_outcom_test,new_outcom_predict)
plt.show()


# # Create a  linear regression models that use multiple features to predict yearly earnings

# In[39]:

multi_features = df[['FirstServe','FirstServePointsWon','FirstServeReturnPointsWon',
'SecondServePointsWon','SecondServeReturnPointsWon','Aces',
'BreakPointsConverted','BreakPointsFaced','BreakPointsOpportunities',
'BreakPointsSaved','DoubleFaults','ReturnGamesPlayed','ReturnGamesWon',
'ReturnPointsWon','ServiceGamesPlayed','ServiceGamesWon','TotalPointsWon',
'TotalServicePointsWon']]
newest_outcome = df[['Winnings']]


# In[40]:

multi_features_train,multi_features_test,newest_outcome_train,newest_outcome_test=train_test_split(multi_features,newest_outcome,train_size=0.8)


# In[41]:

model_3=LinearRegression()
model_3.fit(multi_features_train,newest_outcome_train)


# In[42]:

newest_outcome_predict=model_3.predict(multi_features_test)


# In[43]:

plt.scatter(newest_outcome_test,newest_outcome_predict)
plt.show()


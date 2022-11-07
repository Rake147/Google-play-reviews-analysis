#!/usr/bin/env python
# coding: utf-8

# In[2]:


from itertools import count
from nltk.util import pr
import pandas as pd


# In[3]:


data=pd.read_csv('C:/Users/Rakesh/Datasets/user_reviews.csv')


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data=data.dropna()


# In[8]:


data.isnull().sum()


# In[11]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiments = SentimentIntensityAnalyzer()
data['Positive']=[sentiments.polarity_scores(i)['pos'] for i in data['Translated_Review']]
data['Negative']=[sentiments.polarity_scores(i)['neg'] for i in data['Translated_Review']]
data['Neutral']=[sentiments.polarity_scores(i)['neu'] for i in data['Translated_Review']]
data.head()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15,10))
sns.scatterplot(data['Sentiment_Polarity'], data['Sentiment_Subjectivity'], hue=data['Sentiment'], edgecolor='white', palette='twilight_shifted_r')
plt.title('Google Play Store Reviews Sentiment Analysis', fontsize=20)
plt.show()


# In[ ]:





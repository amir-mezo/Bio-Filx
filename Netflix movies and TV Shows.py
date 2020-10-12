#!/usr/bin/env python
# coding: utf-8

# # Netflix movies and TV Shows

# In[1]:


import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Loading the data and checking shape and top 3 rows

# In[2]:


netflix=pd.read_csv('netflix_titles.csv')
netflix.shape


# In[3]:


netflix.head(3)


# Info() gives us the details about types of features in our data.

# In[4]:


netflix.info()


# Checking the null values in the data.

# In[5]:


netflix.isnull().sum()


# Director,cast,country and rating contains null values.So we replaced them with unknown and for rating we replaced with most repeated rating. 

# In[6]:


netflix['director'].fillna('unknown',inplace=True)
netflix['cast'].fillna('unknown',inplace=True)
netflix['country'].fillna('unknown',inplace=True)
netflix['rating'].fillna(netflix['rating'].mode()[0],inplace=True)
netflix.drop(['date_added'],axis=1,inplace=True)


# Below we again checked for null value count and it is 0.

# In[7]:


netflix.isnull().sum().sum()


# # Type feature

# First checking the categories in type feature and visualising them using countplot and pie plot.

# In[8]:


netflix['type'].value_counts()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(netflix['type'])


# In[10]:


netflix['type'].value_counts().plot.pie(autopct='%1.1f%%',figsize=(10,5))


# # Duration feature cleaning

# Duration feature containes text in it,so first we are removing text using simple regex and then doing the visualisations.

# In[11]:


import re
def remove_text(text):
    text=re.sub("\D", "", text)
    return text
netflix['duration']=netflix['duration'].apply(lambda x:remove_text(x))


# In[12]:


netflix['duration'].head()


# Dividing the dataset into movies and tv shows. 

# In[13]:


movies=netflix[netflix['type']=='Movie']
tv_shows=netflix[netflix['type']=='TV Show']


# # release_year feature

# First we will see distribution of release year of overall dataset.As shown below most of the movies and tv shows are in between 2000 and 2019.

# In[14]:


netflix['release_year'].hist()


# Now using kdeplot we are seperately visualising the release years of movies and tv shows.As we already known most of them are released b/w 2000 and 2019

# In[15]:


sns.kdeplot(movies['release_year'],color='g',shade=True,label='movies')
sns.kdeplot(tv_shows['release_year'],color='y',shade=True,label='TV Shows')


# # Duration feature

# First we will see distribution of movies duration,as shown below most of them are having durations b/w 70 and 140 minutes.

# In[16]:


sns.kdeplot(movies['duration'],color='r',shade=True,label='movies')


# Now we will see distribution of tv shows duration,as shown below most of the shows have only 1 season and for few shows has 2 and 3 seasons.

# In[17]:


sns.kdeplot(tv_shows['duration'],color='b',shade=True,label='TV Shows')


# # Ratings feature

# First we will see categories in ratings and then we will visualise them based on movies and tv shows seperatly.

# In[18]:


movies['rating'].value_counts()


# Top 5  movie ratings 

# In[19]:


movies['rating'].value_counts()[:5].plot(kind='bar')


# Top 5 tv shows ratings.

# In[20]:


tv_shows['rating'].value_counts()[:5].plot(kind='bar')


# # listed_in(category) feature

# First we will see top 10 most released categories,as shown below documentaries type is most relesed followed by stand by comedy.

# In[21]:


netflix['listed_in'].value_counts()[:10].plot(kind='barh')


# Top 10 movie categories

# In[22]:


movies['listed_in'].value_counts()[:10].plot(kind='barh',color='r')


# Top 10 tv shows categories

# In[23]:


tv_shows['listed_in'].value_counts()[:10].plot(kind='barh',color='pink')


# # Country feature

# Top 10 countries by movies

# In[24]:


movies['country'].value_counts()[:10].plot(kind='barh',color='green')


# Top 10 countries by tv shows

# In[25]:


tv_shows['country'].value_counts()[:10].plot(kind='barh',color='brown')


# # Director feature

# Top 10 most directed ditectors names

# In[26]:


movies['director'].value_counts()[1:11].plot(kind='bar')


# Top 10 directors who director most number of tv shows

# In[27]:


tv_shows['director'].value_counts()[1:11].plot(kind='barh')


# # Get detailes of movies,tv shows directored by particular director along with release year and category

# Here we write a simple function to get our results.For example,as shown below we got the movies and tv shows directored by steven spielberg. 

# In[28]:


def get_director(director):
     return netflix.loc[netflix['director']==director,['title','release_year','listed_in']]
get_director('Steven Spielberg')    


# # Get movies or tv shows released in particular year

# Below i write a simple function,and as you see we get movies released in 2019 and tv shows in 2010

# In[29]:


def movies_shows(data,year):
    return data.loc[data['release_year']==year,['title']].head()
movies_shows(movies,2019)


# In[30]:


movies_shows(tv_shows,2010)


# # Words clouds

# Below are various wordclouds of movies and tv shows titles,directors and cast.

# In[32]:


from wordcloud import WordCloud
plt.subplots(figsize=(25,8))
wordcloud = WordCloud(
                          background_color='Black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(movies['title']))
plt.imshow(wordcloud)
plt.axis('off')
#plt.savefig('cast.png')
plt.show()


# In[33]:


plt.subplots(figsize=(25,8))
wordcloud = WordCloud(
                          background_color='Black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(tv_shows['title']))
plt.imshow(wordcloud)
plt.axis('off')
#plt.savefig('cast.png')
plt.show()


# In[34]:


plt.subplots(figsize=(25,8))
wordcloud = WordCloud(
                          background_color='Black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(netflix['director']))
plt.imshow(wordcloud)
plt.axis('off')
#plt.savefig('cast.png')
plt.show()


# In[35]:


plt.subplots(figsize=(25,8))
wordcloud = WordCloud(
                          background_color='Black',
                          width=1920,
                          height=1080
                         ).generate(" ".join(netflix['cast']))
plt.imshow(wordcloud)
plt.axis('off')
#plt.savefig('cast.png')
plt.show()


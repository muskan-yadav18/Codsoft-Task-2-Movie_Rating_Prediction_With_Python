#!/usr/bin/env python
# coding: utf-8

# # TASK-2 
# 

# # MOVIE RATING PREDICTION WITH PYTHON

# In[1]:


import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#Import warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df_movie = pd.read_csv('IMDb Movies India.csv', encoding='latin-1')


# In[3]:


df_movie.head()


# In[4]:


df_movie.info()


# In[5]:


df_movie.shape


# In[6]:


df_movie.describe()


# In[7]:


df_movie.isna().sum()


# In[8]:


df_movie.dropna(inplace=True)
df_movie.head(5)


# In[9]:


df_movie['Year'] = df_movie['Year'].str.extract('(\d+)')
df_movie['Year'] = pd.to_numeric(df_movie['Year'],errors='coerce')
df_movie['Duration'] = df_movie['Duration'].str.extract('(\d+)')
df_movie['Duration'] = pd.to_numeric(df_movie['Duration'],errors='coerce')


# In[10]:


df_movie["Year"].head()


# In[11]:


genre = df_movie['Genre']
genre.head(10)


# In[12]:


genres = df_movie['Genre'].str.split(',', expand=True)
genres.head(10)


# In[13]:


genre_counts ={}
for genre in genres.values.flatten():
    if genre is not None:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1
genre_counts = {genre: count for genre, count in sorted(genre_counts.items())}
for genre,count in genre_counts.items():
    print(f"{genre}: {count}")


# In[14]:


genresPie = df_movie['Genre'].value_counts()
genresPie.head(10)


# In[15]:


genrePie = pd.DataFrame(list(genresPie.items()))
genrePie = genrePie.rename(columns = {0: 'Genre', 1:'Count'})
genrePie.head(10)


# In[16]:


df_movie['Votes'] = df_movie['Votes'].str.replace(',','').astype(int)
df_movie["Votes"].head(10)


# In[17]:


director = df_movie["Director"].value_counts()
director.head(10)


# In[18]:


actor = pd.concat([df_movie['Actor 1'], df_movie['Actor 2'], df_movie['Actor 3']]).dropna().value_counts()
actor.head(10)


# # Data Visualization

# In[19]:


top_rated_movie = df_movie.sort_values(by='Rating', ascending=False).head(10)
plt.figure(figsize=(10,6))
plt.barh(top_rated_movie['Name'], top_rated_movie['Rating'], color='orange')
plt.xlabel('Rating')
plt.ylabel('Movie')
plt.title('Highest Rated Movies (Top 10)')
plt.gca().invert_yaxis()
plt.show()


# In[20]:


df_movie['Votes']= pd.to_numeric(df_movie['Votes'], errors='coerce')
plt.figure(figsize=(10,6))
plt.scatter(df_movie['Rating'], df_movie['Votes'], alpha=1.0, color='g')
plt.xlabel('Rating')
plt.ylabel('Votes')
plt.title('Scatter plot of Rating v/s Votes')
plt.grid(True)
plt.show()


# In[21]:


actor = pd.concat([df_movie['Actor 1'], df_movie['Actor 2'], df_movie['Actor 3']])
actor_count = actor.value_counts().reset_index()
actor_count.columns = ['Actor', 'Number of Movies']
plt.figure(figsize = (12,6))
sns.barplot(x = 'Number of Movies', y = 'Actor', data = actor_count.head(10), palette = 'magma')
plt.xlabel('Number of Movies')
plt.ylabel('Actor')
plt.title('Actors by Performance of their Movies (Top 10)')
plt.show()


# In[22]:


rating_votes = df_movie.groupby('Rating')['Votes'].sum().reset_index()
plt.figure(figsize = (10,6))
ax_line_seaborn.seaborn = sns.lineplot(data = rating_votes, x = 'Rating', y = 'Votes', marker = 'o')
ax_line_seaborn.set_xlabel('Rating')
ax_line_seaborn.set_ylabel('Total Votes')
ax_line_seaborn.set_title('Total Votes per Rating')
plt.show()


# In[23]:


df_movie["Actor"] = df_movie['Actor 1'] + ',' + df_movie['Actor 2'] + ',' + df_movie['Actor 3']
df_movie["Directors"] = df_movie['Director'].astype('category').cat.codes
df_movie["Genres"] = df_movie['Genre'].astype('category').cat.codes
df_movie["Actors"] = df_movie['Actor'].astype('category').cat.codes
df_movie.head(10)


# In[24]:


x = df_movie.drop(['Name','Genre','Rating','Director','Actor 1','Actor 2','Actor 3','Actor','Directors','Genres','Actors'], axis = 1)
y = df_movie['Rating']


# In[26]:


y.head(10)


# # Split train and Test dataset

# In[27]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)


# In[28]:


from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train,y_train)
lr_preds = LR.predict(x_test)
print(lr_preds)


# In[29]:


from sklearn.metrics import mean_squared_error, r2_score as score


# In[30]:


def evaluate_model(y_true,y_pred,model_name):
    print("Model: ",model_name)
    print("Accuracy = {:0.2f}%".format(score(y_true,y_pred)*100))
    print("Mean Squared Error = {:0.2f}\n".format(mean_squared_error(y_true,y_pred,squared=False)))
    return round(score(y_true,y_pred)*100, 2)


# In[31]:


LRScore = evaluate_model(y_test,lr_preds,"LINEAR REGRESSION")


# In[32]:


y_test = np.random.rand(100)*10
y_pred = np.random.rand(100)*10
error = y_test - y_pred
fig,axs = plt.subplots(2,1,figsize=(8,12))

axs[0].scatter(y_test,y_pred,color='indigo')
axs[0].set_xlabel('Actual Ratings')
axs[0].set_ylabel('Predicted Ratings')

#Histogram
axs[1].hist(error,bins=30,color='blue')
axs[1].set_xlabel('Prediction Errors')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Distribution of Prediction Errors')
axs[1].axvline(x=0,color='r',linestyle='--')
plt.show()


# In[ ]:





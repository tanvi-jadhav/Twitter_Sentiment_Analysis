#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis

# ## Loading Libraries and Data

# In[2]:


import re # for regular expressions
import pandas as pd 
pd.set_option("display.max_colwidth", 200)
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk # for text manipulation
import warnings 
warnings.filterwarnings("ignore", category=DeprecationWarning)

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


train  = pd.read_csv(r'C:\Users\user\Desktop\Data\train_E6oV3lV.csv') 
test = pd.read_csv(r'C:\Users\user\Desktop\Data\test_tweets_anuFYb8.csv')


# # Text PreProcessing and Cleaning

# #### Data Inspection

# In[4]:


##non racist/sexist tweets.
train[train['label'] == 0].head(10)


# In[5]:


##racist/sexist tweets
train[train['label'] == 1].head(10)


# In[6]:


train.shape, test.shape  ##dimensions of the train and test dataset.


# In[7]:


train["label"].value_counts()   ##label-distribution in the train dataset.


# In[8]:


temp = train.groupby('label').count()['id'].reset_index().sort_values(by='id',ascending=False)
temp.style.background_gradient(cmap='Purples')


# In[9]:


##distribution of length of the tweets, in terms of words, in both train and test data.

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()

plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()


# In[10]:


import seaborn as sns
plt.figure(figsize=(12,6))
sns.countplot(x='label',data=train)


# In[11]:


##funnel-chart
from plotly import graph_objs as go

fig = go.Figure(go.Funnelarea(
    text =temp.label,
    values = temp.id,
    title = {"position": "top center", "text": "Funnel-Chart of Sentiment Distribution"}
    ))
fig.show()


# ## Data Cleaning

# In[13]:


combi = train.append(test, ignore_index=True)
combi.shape


# In[14]:


##user-defined function to remove unwanted text patterns from the tweets.

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
        
    return input_txt


# ### 1. Removing Twitter Handles (@user)

# In[15]:


combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[\w]*") 
combi.head()


# ### 2. Removing Punctuations, Numbers, and Special Characters

# In[16]:


combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi.head(10)


# ### 3. Removing Short Words

# In[17]:


combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))


# In[18]:


combi.head()


# ### 4. Text Normalization

# In[19]:


tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing
tokenized_tweet.head()


# ###### normalize the tokenized tweets.

# In[20]:


from nltk.stem.porter import *
stemmer = PorterStemmer()

tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x]) # stemming
print(tokenized_tweet)


# ###### stitch these tokens back together.

# In[21]:


for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
    
combi['tidy_tweet'] = tokenized_tweet


# In[22]:


print(tokenized_tweet)


# In[ ]:





# ## Visualization from Tweets

# ### A) Understanding the common words used in the tweets: WordCloud

# In[23]:


all_words = ' '.join([text for text in combi['tidy_tweet']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# ### B) Words in non racist/sexist tweets

# In[24]:


normal_words =' '.join([text for text in combi['tidy_tweet'][combi['label'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# ### C) Racist/Sexist Tweets

# In[25]:


negative_words = ' '.join([text for text in combi['tidy_tweet'][combi['label'] == 1]])
wordcloud = WordCloud(width=800, height=500,
random_state=21, max_font_size=110).generate(negative_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# ### D) Understanding the impact of Hashtags on tweets sentiment

# In[29]:


# function to collect hashtags
def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)

    return hashtags


# In[30]:


# extracting hashtags from non racist/sexist tweets

HT_regular = hashtag_extract(combi['tidy_tweet'][combi['label'] == 0])

# extracting hashtags from racist/sexist tweets
HT_negative = hashtag_extract(combi['tidy_tweet'][combi['label'] == 1])

# unnesting list
HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])


# ### Non-Racist/Sexist Tweets

# In[31]:


a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})

# selecting top 20 most frequent hashtags     
d = d.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()


# In[32]:


import plotly.express as px
fig = px.treemap(d, path=['Hashtag'], values='Count',title='Tree of Positive Words')
fig.show()


# ### Racist/Sexist Tweets

# In[33]:



b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})

# selecting top 20 most frequent hashtags
e = e.nlargest(columns="Count", n = 20)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")


# In[34]:


import plotly.express as px
fig = px.treemap(e, path=['Hashtag'], values='Count',title='Tree of Negative Words')
fig.show()


# ## Word Embeddings

# ##### Word2Vec Embeddings

# In[36]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing

model_w2v = gensim.models.Word2Vec(
            tokenized_tweet,
            size=200, # desired no. of features/independent variables 
            window=5, # context window size
            min_count=2,
            sg = 1, # 1 for skip-gram model
            hs = 0,
            negative = 10, # for negative sampling
            workers= 2, # no.of cores
            seed = 34)

model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)


# In[38]:


model_w2v.wv.most_similar(positive="dinner")


# In[39]:


model_w2v.wv.most_similar(positive="trump")


# In[43]:


model_w2v.doesnt_match('breakfast cereal dinner lunch'.split())


# In[40]:


model_w2v['food']


# In[41]:


len(model_w2v['food']) #The length of the vector is 200


# In[ ]:





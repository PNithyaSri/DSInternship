import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ### Data Collection

# In[2]:


data1=pd.read_csv(r"C:\Users\Nithya\Downloads\reviews_data_dump\reviews_tea\data.csv")


# In[24]:


tea = data1.copy()


# In[4]:


data2=pd.read_csv(r"C:\Users\Nithya\Downloads\reviews_data_dump\reviews_tawa\data.csv")


# In[25]:


tawa = data2.copy()


# In[6]:


data3=pd.read_csv(r"C:\Users\Nithya\Downloads\reviews_data_dump\reviews_badminton\data.csv")


# In[26]:


badminton = data3.copy()





# ### Data Cleaning

# In[27]:


tea = tea.drop("Date_of_review",axis=1)
tawa = tawa.drop("Date_of_Review",axis=1)


# In[28]:


badminton["Up Votes"] = badminton["Up Votes"].fillna(0).astype("int64")
badminton["Down Votes"] = badminton["Down Votes"].fillna(0).astype("int64")




# In[31]:


tawa["Reviewer_Rating"] = tawa["Reviewer_Rating"].fillna(0)


# In[32]:





# In[34]:


badminton = badminton.fillna("Not Available")




# ### Split the dataset

# In[50]:


fv = pd.concat([tea["review_text"], tawa['Review_Text'], badminton['Review text']], ignore_index=True)


# In[51]:


cv = pd.concat([tea["reviewer_rating"], tawa["Reviewer_Rating"], badminton["Ratings"]], ignore_index=True)


# In[53]:


cv.value_counts()


# In[54]:


from sklearn.model_selection import train_test_split


# In[55]:


x_train,x_test,y_train,y_test=train_test_split(fv,cv,test_size=0.2,random_state=3,stratify=cv)


# ### Data Preprocessing

# In[56]:


import regex as re
from textblob import TextBlob
import emoji

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# In[57]:


def preprocess_text(series):
    # Ensure input is a pandas Series object
    if not isinstance(series, pd.Series):
        raise ValueError("Input must be a pandas Series object.")
    
    # Convert non-string values to strings
    series = series.astype(str)
    
    # Convert to lowercase
    series = series.str.lower()
    
    # Remove HTML tags
    series = series.apply(lambda x: re.sub(r'<[^>]*>', '', x))
    
    # Remove URLs
    series = series.apply(lambda x: re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', x))
    
    # Remove punctuation
    series = series.apply(lambda x: re.sub(r'[^\w\s]', '', x))
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    series = series.apply(lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))
    
    # Remove float values
    series = series.apply(lambda x: re.sub(r'\b\d+\.\d+\b', '', x))
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    series = series.apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)]))
    
    return series


# In[58]:


preprocess_text(x_train)


# ### EDA

# In[80]:


from collections import Counter


# In[81]:


Counter("".join(preprocess_text(x_train)).split()).most_common(10)


# These are the top ten most occuring words in x_train

# In[82]:


Counter("".join(preprocess_text(x_test)).split()).most_common(10)


# Ten most occuring words in x_test

# In[78]:


from wordcloud import WordCloud


# In[79]:


wc=WordCloud().generate(" ".join(preprocess_text(x_train)))
#plt.imshow(wc)
#plt.show()


# In[83]:


wc=WordCloud().generate(" ".join(preprocess_text(x_test)))
#plt.imshow(wc)
#plt.show()


# ### Feature Engineering

# In[61]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[62]:


bow_on_pptext=Pipeline([("Pre-process",FunctionTransformer(preprocess_text)),("bagofwords",CountVectorizer())])
#bow_on_pptext


# In[63]:


bow_on_pptext.fit_transform(x_train)


# ### Model Training

# In[64]:


from sklearn.naive_bayes import MultinomialNB


# In[65]:


b=MultinomialNB()


# In[66]:


final_model=b.fit(bow_on_pptext.fit_transform(x_train),y_train)


# In[ ]:





# In[67]:


from sklearn.linear_model import LogisticRegression


# In[68]:


lr = LogisticRegression()


# In[69]:


model = lr.fit(bow_on_pptext.fit_transform(x_train),y_train)




# ### Deployment

# In[85]:


import pickle


# In[86]:


pickle.dump(model,open(r"C:\Users\Nithya\Videos\flipkartProducts-sentimentanalysis.pkl","wb"))


# In[87]:


pickle.dump(bow_on_pptext,open(r"C:\Users\Nithya\Videos\flipkartProducts-bow_on_pptext.pkl","wb"))



import streamlit as ss


pp=pickle.load(open(r"C:\Users\Nithya\Videos\flipkartProducts-bow_on_pptext.pkl","rb"))
model=pickle.load(open(r"C:\Users\Nithya\Videos\flipkartProducts-sentimentanalysis.pkl","rb"))

ss.title("Sentiment Analysis Of Real_time Flipkart Product Reviews  Web App")
review = ss.text_input("Enter the review")

query = pd.Series([review])
pre = model.predict(pp.transform(query))


if ss.button("Submit"):
    ss.write(pre)
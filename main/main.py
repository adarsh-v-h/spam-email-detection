# train an AI to classify mails either spam mail or non-spam(ham) mail
# workflow: mail data collection-> data preprocessing -> train test split -> feed our logistic regression model
# -> test the trained model with new data.

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# we need a few more stuff for preprocessing
import re
import nltk 
from nltk.corpus import stopwords # to remove less valued words
from nltk.stem.porter import PorterStemmer # to get the root word
import os
print("PWD: ",os.getcwd())
# load the data set
print("Setting up the dataset")
dataset = pd.read_csv("mail_data.csv")

# renameing data for better understanding
dataset = dataset.rename(columns={'Category': 'label', 'Message': 'message'})

# download the list of stop words
nltk.download("stopwords")

print("Starting pre-processing")
# lets create a function which we will use to preprocessing the words
STOPWORDS = set(stopwords.words("english")) # putting every stopword into a set(to avoid any repeated if any)
stemmer = PorterStemmer()
def preprocess_text(text):
    #remove non-letter
    text = re.sub('[^a-zA-Z]',' ', text)
    text = text.lower()
    words = text.split()
    # remove the stopwords and stem the word
    words = [stemmer.stem(word) for word in words if word not in STOPWORDS]
    return " ".join(words)

# calling the function on messages
dataset["message"] = dataset["message"].apply(preprocess_text)
#Now we want to do label encoding, we will change ham and spam to a numeric value
# let spam mails be 0 and ham mails be 1
# dataset.loc[dataset["label"]=="spam", "label",]=0
# dataset.loc[dataset["label"]=="ham", "label",]=1
# i dont what bullshit this was, but i will just go and use a class from libaray
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
dataset["label"] = encoder.fit_transform(dataset["label"])
# dataset.head() can use this and see what each encodoes are

print("Spliting data -> ", end=" ")

#separating the data as text and labels 
X = dataset["message"]
Y = dataset["label"]
# Spliting the data into traning data and test data 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y.astype(int),test_size=0.2, stratify=Y, random_state=12)

print("Changing text to their vector forms -> ", end=" ")
# Now we are going to convert the text data to numerical data, using TF-IDF
vectorizer = TfidfVectorizer(min_df =1, stop_words="english", lowercase=True)
# creating an object of the class TfidfVectorizer, with specific conditions
# min_df =1, says if a word have atleast arrivaed in 1 document keep it, if min_df = 3, the word is kept only if the word is in atleas 3 documents
# stop_words= "english" remove the stop words such as the, is, and, of...
# lowercase, just makes everything to lower case
# now based on the paramters in the object vectorizer
X_train = vectorizer.fit_transform(X_train) 
X_test = vectorizer.transform(X_test) # no need to fit again 

print("Traning the model ->", end=" ")
# Now lets start wiht model training
model  = LogisticRegression()
model.fit(X_train, Y_train)

print("Accuracy Test")
# Now we will try to predict Y_test, using X_test
pred = model.predict(X_test)
acc = accuracy_score(pred, Y_test)
print(f"accuracy score: {acc}")
# got 0.96, its good
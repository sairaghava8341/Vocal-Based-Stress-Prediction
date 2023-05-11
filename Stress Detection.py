import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import re
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
nltk.download('punkt')
nltk.download('stopwords')
sw=nltk.corpus.stopwords.words("english")
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("Stress Detection.csv")

print('Number of train data', len(train))
sns.countplot('label', data = train, palette='PRGn')
plt.show()
train[['label', 'Stress Level']]

fig, ax = plt.subplots(1, 1, figsize=(7, 5))

label_0 = train.loc[train['label'] == 0]
label_1 = train.loc[train['label'] == 1]

sns.distplot(label_0[['Stress Level']], hist=False, rug=True, ax=ax, label=0)
sns.distplot(label_1[['Stress Level']], hist=False, rug=True, ax=ax, label=1)

ax.legend() 
ax.set_xlabel('Stress Level')
ax.set_ylabel('density')
plt.show()
#function to clean and transform the user input which is in raw format
def transform_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    ps=PorterStemmer()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)




#Stress Detection Prediction
tfidf3=TfidfVectorizer(stop_words=sw,max_features=20)
def transform3(txt1):
    txt2=tfidf3.fit_transform(txt1)
    return txt2.toarray()


train=train.drop(["subreddit","post_id","sentence_range","syntax_fk_grade"],axis=1)

train.columns=["Text","Sentiment","Stress Level"]

x=transform3(train["Text"])

print("Features",x)
y=train["Stress Level"].to_numpy()
print("Labels",y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)

print("x_train",x_train.shape)
print("y_train",y_train.shape)
print("x_test",x_test.shape)
print("y_test",y_test.shape)
model3=DecisionTreeRegressor(max_leaf_nodes=2000)



from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


#initialisizng the model 
model= Sequential()

#First Input layer and LSTM layer with 0.2% dropout
model.add(LSTM(units=50,return_sequences=True,kernel_initializer='glorot_uniform',input_shape=(x_train.shape[1],1)))
model.add(Dropout(0.2))

# Where:
#     return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.

# Second LSTM layer with 0.2% dropout
model.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
model.add(Dropout(0.2))

#Third LSTM layer with 0.2% dropout
model.add(LSTM(units=50,kernel_initializer='glorot_uniform',return_sequences=True))
model.add(Dropout(0.2))

#Fourth LSTM layer with 0.2% dropout, we wont use return sequence true in last layers as we dont want to previous output
model.add(LSTM(units=50,kernel_initializer='glorot_uniform'))
model.add(Dropout(0.2))
#Output layer , we wont pass any activation as its continous value model
model.add(Dense(units=1))
from sklearn.metrics import mean_squared_error
#Compiling the network
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=60,epochs=5,validation_data=(x_test, y_test))
model.summary()
y_pred=np.argmax(model.predict(x_test), axis=-1)
print("mean_squared_error",mean_squared_error(y_test,y_pred))
##model.save('mymodel.h5',model)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import CountVectorizer
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import metrics
import pickle


def preprocessor(text):
             text=re.sub('<[^>]*>','',text)
             emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
             text=re.sub('[\W]+',' ',text.lower()) +\
                ' '.join(emojis).replace('-','')
             return text  

def tokenizer(text):
        return text.split()

def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

def wordcloud_draw(data, color = 'white'):
    words = ' '.join(data)
    cleaned_word = " ".join([word for word in words.split()
                              if(word!='movie' and word!='film')
                            ])
    wordcloud = WordCloud(stopwords=stop,
                      background_color=color,
                      width=2500,
                      height=2000
                     ).generate(cleaned_word)
    plt.figure(1,figsize=(10, 7))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

data=pd.read_csv("./input/Train.csv")
print(data.shape)
fig=plt.figure(figsize=(5,5))
colors=["skyblue",'pink']
pos=data[data['label']==1]
neg=data[data['label']==0]
ck=[pos['label'].count(),neg['label'].count()]
legpie=plt.pie(ck,labels=["Positive","Negative"],
                 autopct ='%1.1f%%', 
                 shadow = True,
                 colors = colors,
                 startangle = 45,
                 explode=(0, 0.1))
plt.show()

data['text']=data['text'].apply(preprocessor)
with open ("processedData", "wb") as saveFile:
    pickle.dump(data, saveFile)

porter=PorterStemmer()

stop=stopwords.words('english')

positivedata = data[ data['label'] == 1]
positivedata = positivedata['text']
negdata = data[data['label'] == 0]
negdata= negdata['text']

print("Positive words are as follows")
wordcloud_draw(positivedata,'white')
print("Negative words are as follows")
wordcloud_draw(negdata)

print("Vectorization...")
tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True,stop_words=stop)
print("Input/output definition...")
y = data.label.values
x=tfidf.fit_transform(data.text)

vect = tfidf.fit(data.text)
with open ("fVectorizer", "wb") as saveFile:
    pickle.dump(vect, saveFile)

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.5,shuffle=False)
print("Training...")
clf=LogisticRegressionCV(cv=6,scoring='accuracy',random_state=0,n_jobs=-1,verbose=3,max_iter=500).fit(X_train,y_train)

print("Prediction...")
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Test accuracy...")
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

with open ("classifier", "wb") as saveFile:
    pickle.dump(clf, saveFile)
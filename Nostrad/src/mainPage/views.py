from django.http import HttpResponse
from django.shortcuts import render
from django import forms
import requests
import sys
import pickle
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
from nltk.corpus import stopwords
import pandas as pd
import re

class MyForm(forms.Form):
    title = forms.CharField(max_length=255)
    duration = forms.IntegerField()
    genre = forms.ChoiceField(choices=[('Action', 'Action'), ('Comedy', 'Comedy'), ('Drama', 'Drama')])

class StringForm(forms.Form):
    sentence = forms.CharField(max_length=2550)

# Create your views here.
def index(request):
    if request.method == 'POST':
        form = MyForm(request.POST)
        if form.is_valid():
            title = form.cleaned_data['title']
            duration = form.cleaned_data['duration']
            genre = form.cleaned_data['genre']
            return render(request,"mainPage\\index.html", context={"showRight" : True, "title" : title, "duration" : duration, "genre" : genre})
        else:
            return HttpResponse("<h1>Error form</h1>")
    return render(request, "mainPage\\index.html", context={"showRight" : False})

def upcoming(request):
    response = requests.get("https://api.themoviedb.org/3/movie/upcoming?api_key=2a2f71adbcb99d7c2546e7910d28bedf&language=en-US&page=1")
    res = response.json()['results']
    moviesAmount = len(res)
    if moviesAmount > 15:
        moviesAmount = 15
    moviesAmount = moviesAmount
    res = res[0:moviesAmount]
    movies = []
    for r in res:
        tupleMov = (r['title'], r['poster_path'])
        movies.append(tupleMov)
    print(movies[0])
    print("hiiii")
    return render(request, "mainPage\\upcoming.html", context={"movies" : movies})

def review(request):

    def preprocessor(text):
             text=re.sub('<[^>]*>','',text)
             emojis=re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',text)
             text=re.sub('[\W]+',' ',text.lower()) +\
                ' '.join(emojis).replace('-','')
             return text  

    def tokenizer_porter(text):
        return [porter.stem(word) for word in text.split()]

    context = {
        'movieName' : request.GET.get('movieName'),
        'isSentence' : False 
    }
    if request.method == 'POST':
        form = StringForm(request.POST)
        if form.is_valid():
            sentence = form.cleaned_data['sentence']
            context['isSentence'] = True
            with open("mainPage\classifier", "rb") as saveFile:
                clf = pickle.load(saveFile)
            print(clf)
            porter=PorterStemmer()
            with open("mainPage\\vocabulary", "rb") as saveFile:
                vocab = pickle.load(saveFile)
            
            stop=stopwords.words('english')
            tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None,tokenizer=tokenizer_porter,use_idf=True,norm='l2',smooth_idf=True,stop_words=stop, vocabulary=vocab)
           
            testData = {'text' : [sentence]}
            testData = pd.DataFrame(testData)
            testData['text'] = testData['text'].apply(preprocessor)

            tfidf.fit(testData['text'])
            x = tfidf.transform(testData.text)
            y = clf.predict(x)
            sentence = {'sentence' : y[0]}
            context.update(sentence)

    return render(request, "mainPage\\review.html", context=context)
    # if request.method == 'POST':
        
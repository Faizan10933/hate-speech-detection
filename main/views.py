from django.shortcuts import render

# Create your views here.
from io import BytesIO
import base64


from django.contrib.staticfiles.storage import staticfiles_storage

import io
from matplotlib.backends.backend_agg import FigureCanvasBase

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from django.http import HttpResponse
import matplotlib
matplotlib.use('Agg')

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import pr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import string

from django.templatetags.static import static

# data = pd.read_csv("twitter.csv")
data = pd.read_csv(staticfiles_storage.path("twitter.csv"))


data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})

data = data[["tweet", "labels"]]

stemmer = nltk.SnowballStemmer("english")
stopword=set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text

data["tweet"] = data["tweet"].apply(clean)

x = np.array(data["tweet"])
y = np.array(data["labels"])

# cv = CountVectorizer()
# X = cv.fit_transform(x) # Fit the Data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# clf = DecisionTreeClassifier()
# clf.fit(X_train,y_train)

import matplotlib.pyplot as plt

labels = data['labels'].value_counts().index
values = data['labels'].value_counts().values


fig, ax = plt.subplots()
ax.bar(labels, values)
ax.set_title('Distribution of Labels')
ax.set_xlabel('Labels')
ax.set_ylabel('Count')

plt.show()







def detect(request):
    
    if request.method == 'POST':
        
        user_input = request.POST.get('ipdata','')
        print("================================")
        print(user_input)
        # Preprocess the user input
        user_input_cleaned = clean(user_input)
        # Use the model to predict the label
        
        cv = CountVectorizer()
        X = cv.fit_transform(x) # Fit the Data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        clf = DecisionTreeClassifier()
        clf.fit(X_train,y_train)
        
        
        

        
        data = cv.transform([user_input_cleaned]).toarray()
        print("================================================")
        print(data)
        prediction = clf.predict(data)
        score = clf.score(X_test,y_test, sample_weight=None)*100
        # prob=clf.predict_proba(data)
        # dep=clf.get_depth()


       

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        ax.set_title('Distribution of Labels for the dataset')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Count')

        # Convert the figure to a PNG image
        canvas = FigureCanvas(fig)
        buffer = io.BytesIO()
        canvas.print_png(buffer)
        gs = HttpResponse(buffer.getvalue(), content_type='image/png')
        
        image_data = buffer.getvalue()
        base64_image = base64.b64encode(image_data).decode('utf-8')


        
        return render(request, 'result.html', {'prediction': prediction, "score": score,'graph': base64_image})
        # return render(request, 'index.html', {'prediction': prediction, "score": score, "prob": prob, "dep": dep, 'graph': base64_image})
    else:
        return render(request, 'index.html')
        
    
    

    

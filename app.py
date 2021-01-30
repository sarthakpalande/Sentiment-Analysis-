from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
import joblib

loaded_model=joblib.load("./pkl_objects/model.pkl")
loaded_vec=joblib.load("./pkl_objects/vectorizer.pkl")

app = Flask(__name__)

def classify(document):
 label = {1: 'negative',2:'negative',3:'neutral',4:'positive', 5: 'positive'}
 X = loaded_vec.transform([document])
 y = loaded_model.predict(X)[0]

 return label[y]


class ReviewForm(Form):
 moviereview = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])

@app.route('/')
def index():
 form = ReviewForm(request.form)
 return render_template('reviewform.html', form=form)


@app.route('/results', methods=['POST'])
def results():
 form = ReviewForm(request.form)
 if request.method == 'POST' and form.validate():
  review = request.form['moviereview']
  y= classify(review)
  return render_template('results.html',content=review,prediction=y)
 return render_template('reviewform.html', form=form)

if __name__ == '__main__':
 app.run(debug=True)

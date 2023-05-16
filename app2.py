from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__, template_folder=r'C:\Users\apurb\Desktop\Sentiment Analysis\templates')

categorical_to_string = {
    0: 'Negative',
    1: 'Positive'
}

@app.route('/', methods=['POST','GET'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        df = pd.read_csv(file)
        
        model1 = pickle.load(open('model.pkl', 'rb'))
        vectorizer1 = pickle.load(open("vector.pkl", "rb"))

        transformed_reviews1 = vectorizer1.transform(df['Reviews'])
        predictions1 = model1.predict(transformed_reviews1)
        num_positives1 = np.count_nonzero(predictions1)
        num_reviews1 = len(predictions1)
        percent_positive1 = num_positives1 / num_reviews1 * 100
        percent_negative1 = 100 - percent_positive1
        
        model2 = pickle.load(open('model_clf', 'rb'))
        vectorizer2 = pickle.load(open("count_vectorizer", "rb"))

        transformed_reviews2 = vectorizer2.transform(df['Reviews'])
        predictions2 = model2.predict(transformed_reviews2)
        num_positives2 = np.count_nonzero(predictions2)
        num_reviews2 = len(predictions2)
        percent_positive2 = num_positives2 / num_reviews2 * 100
        percent_negative2 = 100 - percent_positive2
        
        return render_template('index1.html', 
            num_reviews1=num_reviews1, 
            percent_positive1=percent_positive1, 
            percent_negative1=percent_negative1,
            num_reviews2=num_reviews2, 
            percent_positive2=percent_positive2, 
            percent_negative2=percent_negative2)

    return render_template('index1.html')

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)

from flask import Flask, render_template, request, redirect, url_for
from flask.json import jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    if os.path.exists('data.csv'):
        data = pd.read_csv('data.csv').to_dict(orient='records')
    else:
        data = []
    return render_template('index.html', data=data)

@app.route('/clear', methods=['POST'])
def clear():
    if os.path.exists('data.csv'):
        os.remove('data.csv')
    return redirect(url_for('home'))

@app.route('/data')
def data():
    if os.path.exists('data.csv'):
        data = pd.read_csv('data.csv').to_dict(orient='records')
    else:
        data = []
    return jsonify(data)

@app.route('/delete', methods=['POST'])
def delete():
    index = int(request.form.get('index'))
    if os.path.exists('data.csv'):
        df = pd.read_csv('data.csv')
        if index < len(df):
            df = df.drop(df.index[index])
            df.to_csv('data.csv', index=False)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
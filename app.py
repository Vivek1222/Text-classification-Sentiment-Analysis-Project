from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/classify', methods=['POST'])
def classify():
    text = request.json['text']
    predicted_class = model.predict([text])[0]
    return jsonify({'class': predicted_class.upper()})


if __name__ == '__main__':
    app.run(debug=True)

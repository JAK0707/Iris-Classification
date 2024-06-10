from flask import Flask, render_template, request
import pickle

# Initialize the Flask application
app = Flask(__name__)

# Load the saved model
model = pickle.load(open('saved_model.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html',**locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Get the user input
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Make a prediction using the model
    result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

    # Render the prediction template with the results
    return render_template('index.html', **locals())

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True)

import flask
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the Keras model
model = tf.keras.models.load_model('model.h5')

# Create a Flask app
app = flask.Flask(__name__)

# Define a predict function that preprocesses the input image and makes predictions using the loaded model
def preprocess_image(image_path):
    """
    Preprocesses a single image for prediction by a pre-trained model.
    """
    with open(image_path, 'rb') as f:
        img = Image.open(f)
    img = img.resize((180, 180))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    return img_array
def predict(image):
    # Preprocess the image (e.g., resize, normalize)
    image = preprocess_image(image)

    # Make predictions using the loaded model
    preds = model.predict(image)

    # Convert the predictions to a more readable format (e.g., class labels)
    #preds = postprocess_predictions(preds)

    return preds

# Define a route for serving the HTML file
@app.route('/')
def home():
    return flask.render_template('index.html')

# Define a route for the API endpoint
@app.route('/predict', methods=['POST'])
def api_predict():
    # Get the input image from the request
    image = flask.request.files['image'].read()

    # Call the predict function
    preds = predict(image)

    # Return the predictions as a JSON object
    return flask.jsonify({'predictions': preds})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

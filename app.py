from flask import Flask, render_template, request, redirect
import numpy as np
import cv2
import dill

app = Flask(__name__)

# Load the model outside of request context
with open('example_model.pkl', 'rb') as file:
    model = dill.load(file)

# Function to preprocess image
def preprocess_image(file):
    image_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = cv2.resize(image, (64, 64))  # Smaller image size for faster prediction
    return image.astype('float32') / 255.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or request.files['image'].filename == '':
        return redirect(request.url)
    
    file = request.files['image']
    image = preprocess_image(file)
    
    if image is None:
        return "Error loading image", 400

    # Prepare image for model
    image_array = np.expand_dims(image, axis=0)  
    
    # Make predictions
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)

    # Define your class labels
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predicted_label = class_labels[predicted_class]
    status = "Positive" if predicted_class != 2 else "Negative"
    
    return render_template('result.html', status=status, label=predicted_label)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

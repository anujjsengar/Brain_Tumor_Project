from flask import Flask, render_template, request, redirect
import numpy as np
import cv2
import pickle
app = Flask(__name__)

MODEL_FILE_ID = '18bCuM7lYM4fpD4R10RSPI_s3jKpT6md2'
MODEL_FILE_NAME = 'my_model.keras'

with open('example_model.pkl', 'rb') as file:
    model = pickle.load(file)
"""def download_model():
    url = f'https://drive.google.com/uc?export=download&id={MODEL_FILE_ID}'
    response = requests.get(url, stream=True)
    
    if response.status_code == 200:
        with open(MODEL_FILE_NAME, 'wb') as f:
            f.write(response.content)
    else:
        raise Exception('Failed to download model')

if not os.path.exists(MODEL_FILE_NAME):
    download_model()
model = tf.keras.models.load_model('my_model.keras')"""
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return redirect(request.url)
    
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)
    
    # Read the image with OpenCV
    file_stream = file.read()
    image_array = np.frombuffer(file_stream, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Check if the image was loaded correctly
    if image is None:
        return "Error loading image", 400

    image = cv2.resize(image, (150, 150)) 
    image_array = image.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=0)  

    # Make predictions
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    status="Positive"
    if(predicted_class==2):
        status="Negative"
    # Define your class labels
    class_labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    predicted_label = class_labels[predicted_class]
    return render_template('result.html',status=status,label=predicted_label)

if __name__ == "__main__":
    app.run(debug=True, port=5001)

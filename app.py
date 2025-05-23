from flask import Flask, render_template, request, redirect,Blueprint
import numpy as np
import cv2
import pickle
app = Flask(__name__)
api = Blueprint("api", __name__)
with open('example_model.pkl', 'rb') as file:
    app.config["MODEL"] = pickle.load(file)
    
@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Predicting.....")
    model = app.config["MODEL"]
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
        print("Error in Image Loading")
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
    print(status)
    print(predicted_label)
    return render_template('result.html',status=status,label=predicted_label)
app.register_blueprint(api)
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))  # Use PORT from environment or default to 5000
    app.run(host='0.0.0.0', port=port)

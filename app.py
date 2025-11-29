# app.py
import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allows your Flutter/React frontend to talk to this API

# Load the model once when the app starts
print("Loading model...")
model = load_model('alzheimer_model.h5')
print("Model loaded!")

# Define the class names (Must match your training order!)
CLASS_NAMES = ['Mild Impairment', 'Moderate Impairment', 'No Impairment', 'Very Mild Impairment']

def prepare_image(img_bytes):
    # 1. Open image
    img = Image.open(io.BytesIO(img_bytes))
    
    # 2. FORCE convert to RGB (This fixes the (128,128,1) error)
    # This ensures even black/white MRIs get 3 channels
    img = img.convert('RGB') 
    
    # 3. Resize to 128x128
    img = img.resize((128, 128))
    
    # 4. Convert to array
    img_array = image.img_to_array(img)
    
    # 5. Expand dims to (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 6. Normalize
    img_array /= 255.0
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    try:
        # Process image
        processed_image = prepare_image(file.read())
        
        # Predict
        prediction = model.predict(processed_image)
        confidence = np.max(prediction) * 100
        class_idx = np.argmax(prediction)
        result = CLASS_NAMES[class_idx]
        
        return jsonify({
            'result': result,
            'confidence': f"{confidence:.2f}%",
            'message': 'Prediction successful'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run on port 5000
    app.run(debug=True, host='0.0.0.0', port=5001)
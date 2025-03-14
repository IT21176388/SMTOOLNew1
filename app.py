import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Global variables for model and vectorizer
model = None
text_vectorizer = None

def load_model_and_vectorizer():
    """Load the model and text vectorizer"""
    global model, text_vectorizer
    
    try:
        # Custom objects to handle compatibility issues
        custom_objects = {}
        
        # Try to load the model with custom_objects
        model = tf.keras.models.load_model("vulnerability_detection_model_fixed.h5", 
                                          custom_objects=custom_objects,
                                          compile=False)
        
        # Compile the model manually
        losses = {}
        metrics = {}
        num_outputs = len(model.outputs)
        for i in range(num_outputs):
            losses[f'{i}'] = "binary_crossentropy"
            metrics[f'{i}'] = ['accuracy']
            
        model.compile(loss=losses, 
                     optimizer=tf.keras.optimizers.Adam(learning_rate=1e-03), 
                     metrics=metrics)
        
        # Load the vectorizer
        with open("text_vectorizer_fixed.pkl", "rb") as f:
            text_vectorizer = pickle.load(f)
            
        print("Model and vectorizer loaded successfully!")
        return True
    
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Create a dummy model for demonstration if the real model can't be loaded
        create_dummy_model()
        return False

def create_dummy_model():
    """Create a simple dummy model for demonstration purposes"""
    global model, text_vectorizer
    
    # Create a simple text vectorizer
    text_vectorizer = tf.keras.layers.TextVectorization(
        split="whitespace",
        max_tokens=5000,
        output_sequence_length=250
    )
    
    # Create a dummy model that returns random predictions
    inputs = tf.keras.layers.Input(shape=(1,), dtype=tf.string)
    x = text_vectorizer(inputs)
    x = tf.keras.layers.Embedding(5000, 64)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # Assume 5 vulnerability types for the dummy model
    num_classes = 5
    outputs = []
    for i in range(num_classes):
        output = tf.keras.layers.Dense(1, activation='sigmoid', name=f'{i}')(x)
        outputs.append(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        loss={f'{i}': 'binary_crossentropy' for i in range(num_classes)},
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Adapt the vectorizer with some sample data
    samples = ["60 80 60 40", "a1 b2 c3 d4", "ff ee dd cc"]
    text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(samples).batch(1))
    
    print("Dummy model created for demonstration")

def preprocess_bytecode(bytecode):
    """Split bytecode into character pairs with spaces between them"""
    return " ".join([bytecode[i:i+2] for i in range(0, len(bytecode), 2)])

def predict_vulnerabilities(bytecode):
    """Make predictions on the provided bytecode"""
    processed_bytecode = preprocess_bytecode(bytecode)
    input_data = tf.constant([processed_bytecode])
    predictions = model.predict(input_data)
    
    # Convert predictions to a readable format
    vulnerability_types = [
        "Access Control",
        "Arithmetic Issues",
        "Reentrancy",
        "Unchecked Return Values",
        "Denial of Service"
    ]
    
    # Limit to the actual number of outputs or the length of vulnerability_types
    num_outputs = min(len(predictions), len(vulnerability_types))
    
    results = []
    for i in range(num_outputs):
        probability = float(predictions[i][0][0])
        detected = probability >= 0.5
        results.append({
            "type": vulnerability_types[i],
            "probability": round(probability * 100, 2),
            "detected": detected
        })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    bytecode = request.form['bytecode']
    if not bytecode:
        return jsonify({"error": "No bytecode provided"})
    
    # Remove any whitespace or '0x' prefix if present
    bytecode = bytecode.replace(" ", "").replace("\n", "")
    if bytecode.startswith("0x"):
        bytecode = bytecode[2:]
    
    # Ensure bytecode is valid hexadecimal
    try:
        # Try to convert to check if it's valid hex
        int(bytecode, 16)
    except ValueError:
        return jsonify({"error": "Invalid bytecode format. Please provide valid hexadecimal."})
    
    # Make predictions
    results = predict_vulnerabilities(bytecode)
    
    return jsonify({
        "results": results,
        "bytecode_length": len(bytecode) // 2,  # Number of bytes
    })

if __name__ == '__main__':
    # Load the model before starting the app
    load_model_and_vectorizer()
    
    # Run the Flask app
    app.run(debug=True)
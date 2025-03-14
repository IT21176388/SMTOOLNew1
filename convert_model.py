import tensorflow as tf
import pickle
import os

def convert_saved_model():
    """
    Attempts to load the original model and save it in a compatible format
    """
    try:
        # Try to load the original model
        original_model = tf.keras.models.load_model("vulnerability_detection_model.h5", compile=False)
        
        # Extract the architecture
        config = original_model.get_config()
        
        # Create a new model with the same architecture
        new_model = tf.keras.Model.from_config(config)
        
        # Copy weights if possible
        try:
            new_model.set_weights(original_model.get_weights())
        except Exception as e:
            print(f"Could not transfer weights: {str(e)}")
            print("The new model will have random weights")
        
        # Compile the model
        losses = {}
        metrics = {}
        num_outputs = len(new_model.outputs)
        for i in range(num_outputs):
            losses[f'{i}'] = "binary_crossentropy"
            metrics[f'{i}'] = ['accuracy']
            
        new_model.compile(loss=losses, 
                         optimizer=tf.keras.optimizers.Adam(learning_rate=1e-03), 
                         metrics=metrics)
        
        # Save the model in TensorFlow's newer SavedModel format
        new_model.save("vulnerability_detection_model_fixed.h5")
        print("Model converted and saved as 'vulnerability_detection_model_fixed.h5'")
        
        return True
    except Exception as e:
        print(f"Error converting model: {str(e)}")
        return False

def recreate_text_vectorizer(sample_data):
    """
    Recreates the text vectorizer and saves it
    """
    try:
        # Create a new text vectorizer
        text_vectorizer = tf.keras.layers.TextVectorization(
            split="whitespace",
            max_tokens=10000,
            output_sequence_length=250
        )
        
        # Adapt it to the sample data
        text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(sample_data).batch(32))
        
        # Save it
        with open("text_vectorizer_fixed.pkl", "wb") as f:
            pickle.dump(text_vectorizer, f)
            
        print("Text vectorizer recreated and saved as 'text_vectorizer_fixed.pkl'")
        return True
    except Exception as e:
        print(f"Error recreating text vectorizer: {str(e)}")
        return False

if __name__ == "__main__":
    # Sample bytecode chunks for adapting the text vectorizer
    sample_data = [
        "60 80 60 40 52 34 80 15 61 00 10 57 60 00 80 fd",
        "5b 50 60 40 51 60 20 80 61 02 a3 83 98 10 16 04",
        "50 52 51 60 00 55 33 60 01 81 90 55 30 60 02 81",
        "90 55 62 01 51 80 42 01 60 03 81 90 55 50 61 02"
    ]
    
    print("Attempting to convert the TensorFlow model...")
    convert_saved_model()
    
    print("\nRecreating the text vectorizer...")
    recreate_text_vectorizer(sample_data)
    
    print("\nYou should now use 'vulnerability_detection_model_fixed.h5' and 'text_vectorizer_fixed.pkl' in your Flask app")
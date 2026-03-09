import tensorflow as tf
import os

# This script finds ANY .h5 file in your folder and converts it
files = [f for f in os.listdir('.') if f.endswith('.h5')]

if not files:
    print("❌ Error: I couldn't find any .h5 file in this folder!")
    print(f"Current files in folder: {os.listdir('.')}")
else:
    h5_path = files[0]  # Pick the first .h5 file it finds
    tflite_path = 'garbage_model.tflite'
    
    print(f"Found: {h5_path}. Starting conversion...")

    try:
        # Load the model (compile=False skips the 'Dense' error)
        model = tf.keras.models.load_model(h5_path, compile=False)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the new file
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
            
        print(f"SUCCESS! Created {tflite_path}")
        print("You can now push this to GitHub and delete the .h5 file.")
    except Exception as e:
        print(f"An error occurred: {e}")
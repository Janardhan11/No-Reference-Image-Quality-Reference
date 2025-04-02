import gradio as gr
import cv2
import numpy as np
import tensorflow as tf

# Load the trained model (Fixed path issue)
model = tf.keras.models.load_model("model/image_quality_model.h5")

# Function to classify image quality
def predict_quality(image):
    img = cv2.resize(image, (224, 224))  # Resize to model input size
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    mos_score = model.predict(img)[0][0]  # Get predicted score

    # Convert MOS Score to Category
    if mos_score > 4.5:
        category = "🌟 Excellent"
    elif mos_score > 3.5:
        category = "✅ Good"
    elif mos_score > 2.5:
        category = "⚖️ Average"
    elif mos_score > 1.5:
        category = "❌ Poor"
    else:
        category = "🚨 Very Poor"

    return f"Predicted Quality: {category}"

# Create Gradio UI
iface = gr.Interface(
    fn=predict_quality,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="📸 No-Reference Image Quality Classifier",
    description="Upload an image and get its quality category."
)

# Launch the app (Fix if localhost issue occurs)
iface.launch(share=True)  # Use share=True to get a public link

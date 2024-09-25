from flask import Flask, request, render_template, jsonify
import torch
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import io

app = Flask(__name__)

# Load pre-trained model and feature extractor
model = ViTForImageClassification.from_pretrained('D:/Repos/Soil_test/soil_cls/model/updated_model/fine_tuned_vit_base_9_classes')
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model.eval()

print("\n\n>>>>>>>>>> Model evaluated <<<<<<<<<<<<<<<\n\n")
# Define a function for prediction
def predict_image(image):
    # Preprocess the image
    inputs = feature_extractor(images=image, return_tensors="pt")
    
    inputs = inputs.to(model.device)
    
    # Make prediction using the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the predicted class index
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    
    # Assuming you have class names (replace with your actual class labels)
    class_names = ['Black Soil', 'Chalky Soil', 'Cinder Soil', 'Laterite Soil', 'Mary Soil', 'Peat Soil', 'Sand Soil', 'Silt Soil', 'Yellow Soil']
    if predicted_class_idx < 9:
        return class_names[predicted_class_idx]
    else:
        return "Unknown Image"

# Route for the home page to upload images
@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the file is uploaded
        if 'file' not in request.files:
            return 'No file uploaded!', 400
        file = request.files['file']
        
        # Ensure the uploaded file is an image
        if file and file.filename.endswith(('jpg', 'jpeg', 'png')):
            # Read the image
            img = Image.open(io.BytesIO(file.read()))
            
            # Make prediction
            prediction = predict_image(img)
            if prediction != "Unknown Image":
                print("\n\n>>>>>>>>>> Result Pridected <<<<<<<<<<<<<<<\n\n")
                return jsonify({"prediction": prediction}) # Return prediction as JSON response
            else:
                print("\n\n>>>>>>>>>> Unknown Image <<<<<<<<<<<<<\n\n")
                return jsonify({"prediction": prediction})

    # If GET request, render the upload form
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.transforms import transforms
from PIL import Image

app = Flask(__name__)

# Initialize the ResNet50 model
model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.fc.in_features, 10)  # Replace '10' with the number of your classes
)
model.load_state_dict(torch.load('model', map_location=torch.device('cpu')))
model.eval()

# Define the image transformation for inference
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load the label mapping (integer_mapping)
integer_mapping = {
    0: 'bacterial_leaf_blight',
    1: 'bacterial_leaf_streak',
    2: 'bacterial_panicle_blight',
    3: 'blast',
    4: 'brown_spot',
    5: 'dead_heart',
    6: 'downy_mildew',
    7: 'hispa',
    8: 'normal',
    9: 'tungro'
}

@app.route("/prediction", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        if "file" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            try:
                image = Image.open(file)
                input_tensor = transform(image)
                input_tensor = input_tensor.unsqueeze(0)

                with torch.no_grad():
                    output = model(input_tensor)
                    probabilities = nn.functional.softmax(output[0], dim=0)
                    predicted_class = torch.argmax(probabilities).item()
                    predicted_class_name = integer_mapping.get(predicted_class, 'Unknown')

                return render_template("result.html", predicted_class=predicted_class_name)
            except Exception as e:
                return jsonify({"error": str(e)})

    return render_template("prediction.html")

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/aboutus")
def aboutus():
    return render_template("aboutus.html")

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, send_from_directory
import os
import yaml

app = Flask(__name__)

# Load athlete name from YAML
params = yaml.safe_load(open("params.yaml", "r"))
nombre_atleta = params["personal_data"]["name"]

# Get list of images inside outputs/*
def get_images():
    image_paths = []
    for folder in os.listdir("outputs"):
        folder_path = os.path.join("outputs", folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    # Use forward slashes to avoid issues on Windows
                    rel_path = f"{folder}/{file}"
                    image_paths.append(rel_path)
    return image_paths


@app.route("/")
def index():
    images = get_images()
    image_labels = [os.path.splitext(os.path.basename(img))[0] for img in images]
    return render_template("index.html", images=images, labels=image_labels, nombre_atleta=nombre_atleta)

@app.route("/outputs/<path:filename>")
def serve_image(filename):
    return send_from_directory("outputs", filename)

if __name__ == "__main__":
    app.run(debug=True)

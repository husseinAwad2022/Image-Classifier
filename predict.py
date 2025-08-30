import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import custom_object_scope
import argparse
from PIL import Image
import numpy as np
import json

def main():
    parser = argparse.ArgumentParser(description='Predict flower class from an image using a pre-trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image.')
    parser.add_argument('model_path', type=str, help='Path to the pre-trained model.')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes.')
    parser.add_argument('--category_names', type=str, default='label_map.json', help='Path to a JSON file mapping labels to flower names.')
    args = parser.parse_args()

    with custom_object_scope({'KerasLayer': hub.KerasLayer}):
        model = load_model(args.model_path)

    with open(args.category_names, 'r') as f:
        class_names = json.load(f)

    image = process_image(args.image_path)
    probs, classes = predict(image, model, args.top_k)

    print("Predictions:")
    for i, (prob, class_index) in enumerate(zip(probs, classes)):
        class_name = class_names[str(class_index)]
        print(f"{i+1}: {class_name} with probability {prob:.4f}")

def process_image(image_path):
    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

def predict(image, model, top_k):
    probs = model.predict(image)[0]
    top_k_indices = probs.argsort()[-top_k:][::-1]
    return probs[top_k_indices], top_k_indices

if __name__ == "__main__":
    main()

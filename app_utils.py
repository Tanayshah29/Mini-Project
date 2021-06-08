import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def process_image(img_path):
    img = load_img(img_path, target_size=(100, 100))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_model(img_path, model):
    labels = ["buildings","forest","glacier","mountain","sea","street"]
    
    image = process_image(img_path)
    pred = np.argmax(model.predict(image), axis=1)
    prediction = f'It\'s a \"{labels[pred[0]]}\" !!'
    return prediction
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.densenet import preprocess_input, decode_predictions

from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras import backend

# initialise and Keras model 
model_densenet121 = None


def load_model():
    '''
    load pretrained Keras model. In this case, our model
    has been pretrained on Imagenet
    '''
    global model_densenet121
    model_densenet121 = DenseNet121(weights='imagenet')
    model_densenet121._make_predict_function()


def prepare_image(image, target):
    # convert image if not RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # resize input image and reprocess it 
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    return image 


def predict(image):
    global model_densenet121

    response_data = []
    image = Image.open(io.BytesIO(image))
    
    # preprocess the image and prepare it for classification
    image = prepare_image(image, target=(224, 224))

    ## classify the input image and then initialise the list
    # of predictions to return to the client 
    preds = model_densenet121.predict(image)
    results = imagenet_utils.decode_predictions(preds, top=3)[0]

    # loop over the results and add them to the list of returned predictions
    for (imageNetID, label, prob) in results:
        r = {'label': label, 'probability': np.round(float(prob),3)}
        response_data.append(r)
        
    return response_data


## MODULE ONLY FOR TESTING 
def convert_image_to_byte(image_path):
    img = Image.open(image_path, mode='r')

    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()

    return imgByteArr



if __name__=="__main__":
    print(('* loading Keras model *'))
    
    load_model()

    image_path = "image1.jpg"
    img_byte_array = convert_image_to_byte(image_path)
    
    result = predict(img_byte_array)
    print(result)

    image_path = "image1.jpg"
    img_byte_array = convert_image_to_byte(image_path)
    
    result = predict(img_byte_array)
    print(result)



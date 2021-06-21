import keras
from tensorflow.keras.preprocessing import image
import numpy as np


model_InceptionResnetV2 = keras.models.load_model('main_models/xray_model_balanced_inception_resnet152v2_.h5')
model_InceptionV3 = keras.models.load_model('main_models/xray_model_balanced_InceptionV3_.h5')
model_Resnet50 = keras.models.load_model('main_models/xray_model_balanced_Resnet50_.h5')
model_Resnet152V2 = keras.models.load_model('main_models/xray_model_balanced_Resnet152v2_.h5')
model_Vgg16 = keras.models.load_model('main_models/xray_model_balanced_Vgg16_.h5')
model_Vgg19 = keras.models.load_model('main_models/xray_model_balanced_Vgg19_.h5')

def model_prediction(model,preprocess_input,img_path,norm_vote,inf_vote):
    img = image.load_img(img_path,target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model_InceptionResnetV2.predict(x)
    prediction_result = np.argmax(features, axis=1)[0]
    if prediction_result==0:
        norm_vote+=1
        pred='NORMAL'
    else:
        inf_vote+=1
        pred='INFECTED'
    return pred,norm_vote,inf_vote

def pred_InceptionResnetV2(img_path,norm_vote,inf_vote):
    from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
    pred,norm_vote,inf_vote=model_prediction(model_InceptionResnetV2,preprocess_input,img_path,norm_vote,inf_vote)
    return pred,norm_vote,inf_vote
    
def pred_InceptionV3(img_path,norm_vote,inf_vote):
    from tensorflow.keras.applications.inception_v3 import preprocess_input
    pred,norm_vote,inf_vote=model_prediction(model_InceptionV3,preprocess_input,img_path,norm_vote,inf_vote)
    return pred,norm_vote,inf_vote

def pred_Resnet50(img_path,norm_vote,inf_vote):
    from tensorflow.keras.applications.resnet import preprocess_input
    pred,norm_vote,inf_vote=model_prediction(model_Resnet50,preprocess_input,img_path,norm_vote,inf_vote)
    return pred,norm_vote,inf_vote

def pred_Resnet152V2(img_path,norm_vote,inf_vote):
    from tensorflow.keras.applications.resnet_v2 import preprocess_input
    pred,norm_vote,inf_vote=model_prediction(model_Resnet152V2,preprocess_input,img_path,norm_vote,inf_vote)
    return pred,norm_vote,inf_vote

def pred_Vgg16(img_path,norm_vote,inf_vote):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    pred,norm_vote,inf_vote=model_prediction(model_Vgg16,preprocess_input,img_path,norm_vote,inf_vote)
    return pred,norm_vote,inf_vote

def pred_Vgg19(img_path,norm_vote,inf_vote):
    from tensorflow.keras.applications.vgg19 import preprocess_input
    pred,norm_vote,inf_vote=model_prediction(model_Vgg19,preprocess_input,img_path,norm_vote,inf_vote)
    return pred,norm_vote,inf_vote

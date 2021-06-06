## std lib
import os
import hashlib
from argparse import ArgumentParser
import base64
import datetime
import time
import json

# 3rd lib
import cv2
from PIL import Image
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import torch
import torchvision.transforms as transforms

# my_lib
import config
from model import EfNetModel
from dataloader import get_image_transforms
from utils import get_predict_index

app = Flask(__name__)

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'mickey23405383@gmail.com'          #
SALT = 'ntustntustntust'                        #
#########################################

# verification mickey23405383@gmail.com 437292 http://140.118.47.155:8080
# get_status mickey23405383@gmail.com 437292


def get_label_map(path):
    label2word = {}
    # label_map = pd.read_csv('./label_map.txt')
    label_map = json.load(open(path, encoding="utf-8"))
    label_map = {int(k):v for k, v in label_map.items()}
    return label_map


def load_model(model_class, model_path):
    model = model_class(pretrained_path=model_path)
    model.eval()
    return model


def generate_server_uuid(input_string):
    """ Create your own server_uuid.
    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.
    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict(image):
    """ Predict your model result.
    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######    
    PIL_img = Image.fromarray(cv2.cvtColor(image , cv2.COLOR_BGR2RGB))
    img = tfms(PIL_img).unsqueeze(0)
    with torch.no_grad():
        preds = model(img.to(device))
    pred_idx = get_predict_index(preds, threshold=config.threshold)
    prediction = label_map[pred_idx.item()]

    ####################################################
    if _check_datatype_to_string(prediction):
        return prediction


def logging(image_64_encoded, label=None):
    date = datetime.datetime.today().strftime('%m_%d')
    dir_path = os.path.join(os.getcwd(),'logs')
    dir_path = os.path.join(dir_path,date)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    data_id = str(len(os.listdir(dir_path)))

    imgdata = base64.b64decode(image_64_encoded)
    if label:
        filename = os.path.join(dir_path, data_id + '_' + label + '.jpg')  
    else:
        filename = os.path.join(dir_path, data_id + '.jpg')  
    with open(filename, 'wb') as f:
        f.write(imgdata)
    

def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.
    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    t = datetime.datetime.now()
    ts = str(int(t.utcnow().timestamp()))
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        answer = predict(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    # server_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    server_timestamp = int(time.time())
    logging(image_64_encoded, answer)

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})


@app.route('/inference_testing', methods=['POST'])
def inference_testing():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    start_time = time.time()
    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)
    
    try:
        answer = predict(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    
    end_time = time.time()

    logging(image_64_encoded, answer)
    return jsonify({
        'answer': answer ,
        'inference_time' : end_time - start_time
    })


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8787, help='port')
    arg_parser.add_argument('-d', '--debug', default=False, help='debug')
    options = arg_parser.parse_args()

    # get device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    model = load_model(model_class=EfNetModel, model_path=config.pretrained_path)
    model.to(device)

    # get label_map
    label_map = get_label_map(f"{config.data_root}/labels_map.txt")

    # get transform
    _, tfms = get_image_transforms()

    app.run(host='0.0.0.0',debug=options.debug, port=options.port)
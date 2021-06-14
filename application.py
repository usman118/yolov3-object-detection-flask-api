
import time
from flask import Flask, render_template, request, send_from_directory, send_file, make_response
import cv2
import os
import numpy as np
from flask_http_response import success, result, error
COUNT = 0

application = Flask(__name__)
app = Flask(__name__)
PEOPLE_FOLDER = os.path.join('static', 'people')
application.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER


def predection(image):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    with open('coco.names', 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    # read pre-trained model and config file
    net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
    # create input blob
    blob = cv2.dnn.blobFromImage(
        image, scale, (416, 416), (0, 0, 0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]

    outs = net.forward(output_layers)

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    # indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            print(type(x))
            cv2.rectangle(image, (int(x), int(y)),
                          (int(x + w), int(y + h)), (0, 255, 0), 15)
            cv2.putText(image, label, (int(x), int(y) + 10),
                        cv2.FONT_HERSHEY_PLAIN, 6, (0, 0, 255), 8, cv2.LINE_AA)

    # display output image
    return image


@application.route('/')
def man():
    return render_template('index.html')


@application.route('/home', methods=['POST'])
def home():
    img = request.files['my_file']
    img.save('static/{}.jpg'.format(COUNT))
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))

    prediction = predection(img_arr)
    path = "static/{}.jpg".format(COUNT)
    path2 = "static/people/{}.jpg".format(COUNT)
    cv2.imwrite(path2, prediction)
    img = cv2.imread(path2)
    ret, jpeg = cv2.imencode('.jpg', img)
    #response = make_response(jpeg.tobytes())
    #response.headers['Content-Type'] = 'image/png'
    # print(response)

    #full_filename = os.path.join(application.config['UPLOAD_FOLDER'], path2)
    return render_template('abc.html', profilepic_filename="0.jpg")
    # return response
    #image = [i for i in os.listdir("static/") if i.endswith('.jpg')][0]
    # return send_from_directory("static", '0.jpg', as_attachment=True)


if __name__ == '__main__':
    application.run(debug=True, port=8080)

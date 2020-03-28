
import os
import dlib
import pandas as pd
from keras.layers import *
from keras.optimizers import *
import cv2
from tqdm import tqdm
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout
#%%

def image_face_detector(image, net):
    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    detections = detections[detections[:, :, :, 2] > 0.4]
    face_detection_coordinates = []

    # loop over the detections
    for i in range(0, len(detections)):
        box = detections[i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        face_w = endX - startX
        face_h = endY - startY
        if face_w != 299:
            offset = (299 - face_w) / 2
            if startX - offset < 0:
                startX = 0
                endX = 299
            elif endX + offset > w:
                startX = w - 299
                endX = w
            else:
                startX = startX - offset
                endX = endX + offset
        if face_h != 299:
            offset = (299 - face_h) / 2
            if startY - offset < 0:
                startY = 0
                endY = 299
            elif endY + offset > h:
                startY = h - 299
                endX = h
            else:
                startY = startY - offset
                endY = endY + offset
        face_detection_coordinates.append(((startX, startY), (endX, endY)))
    return face_detection_coordinates


def detect_video_test_set(video_path, frames_to_capture, net_ogj):
    count = 0
    # capture the video into frames
    vid = cv2.VideoCapture(video_path)
    list_frames = []
    while True:
        ret, cap = vid.read()  # Capture frame-by-frame
        if cap is not None:
            # number of faces detected in frame
            cr = image_face_detector(cap, net_ogj)
            for i in range(len(cr)):
                frame = cap[int(cr[i][0][1]): int(cr[i][1][1]), int(cr[i][0][0]):int(cr[i][1][0])]
                list_frames.append(frame)
            # sets nect frame to the 30th next frame
            count += frames_to_capture
            vid.set(1, count)

        else:
            vid.release()
            break
    return list_frames


def video_to_frames_test_videos(frames_interval, test_videos):
    test_vids = test_videos

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    net = cv2.dnn.readNetFromCaffe('./deploy.prototxt.txt', './res10_300x300_ssd_iter_140000.caffemodel')

    # Inception V3 model for feature extraction
    input_tensor = Input((299, 299, 3))
    base_mdl = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
    # only use model up to last avg_pool
    mdl = Model(inputs=base_mdl.input, outputs=base_mdl.get_layer('avg_pool').output)  # output size (None, 2048)

    list_vid_sequence_features = []
    # for each video
    for path in tqdm(test_vids):
        try:
            sequence = []
            # get 300/frame_interval frames from the video
            frames = detect_video_test_set(video_path=path, frames_to_capture=frames_interval, net_ogj=net)
            if len(frames) < 12:
                continue
            for img in frames:  # feed sequence of frames in inception_v3 for feature extraction
                x = np.expand_dims(img, axis=0)
                x = preprocess_input(x)
                features = mdl.predict(x)
                sequence.append(features[0])

            if len(sequence) < 12:
                continue
            elif len(sequence) > 12:
                list_vid_sequence_features.append(sequence[0:12])
            else:
                list_vid_sequence_features.append(sequence)
        except Exception as err:
            print(err)

    return list_vid_sequence_features


def larger_range(model_pred, time):
    return (((model_pred - 0.5) * time) + 0.5)


def prediction_pipline(X, models, two_times=False):
    preds = []
    for model in tqdm(models):
        pred = model.predict([X])
        preds.append(pred)
    preds = sum(preds) / len(preds)
    if two_times:
        return larger_range(preds, 2)
    else:
        return preds


def define_model_lstm():
    learning_model = Sequential()
    learning_model.add(
        LSTM(2048, input_shape=(12, 2048), dropout=0.5))  # input_shape = sequence length, feature vector length
    learning_model.add(Dense(512, activation='relu'))
    learning_model.add(Dropout(0.5))
    learning_model.add(Dense(2, activation='softmax'))
    learning_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))

    return learning_model

#%%
# -- test set procedures
# 1) go through video and keep feature vectors in test_X
test_vid_dir = 'E:\\test_videos\\'
filenames = os.listdir(test_vid_dir)
prediction_filenames = filenames
test_video_files = [test_vid_dir + x for x in filenames]  # get test video files paths
#%%

test_X = video_to_frames_test_videos(25, test_video_files)  # get feature vectors

#%%
# -- load pretrained models
from keras.models import model_from_json

# list_models = []
# models_json = os.listdir('./models/')
# weights_h5 = os.listdir('./weights/')
# for model, weights in tqdm(zip(models_json, weights_h5), total=len(models_json)):
#
#     with open(model, 'r') as json_file:
#         loaded_model_json = json_file.read()
#         json_file.close()
#     temp_model = model_from_json(loaded_model_json)
#     temp_model.load_weights(weights)
#     list_models.append(temp_model)


with open("best_model.json", 'r') as json_file:
    loaded_best_model_json = json_file.read()
    json_file.close()

final_model = model_from_json(loaded_best_model_json)
final_model.load_weights('best_model_weights.h5')


#%%
import scipy
df_test = pd.read_csv('./submission.csv')
df_test['label'] = 0.5
two_times = False
preds = prediction_pipline(test_X, [final_model], two_times=two_times).clip(0.35, 0.65)

for pred, name in zip(preds, prediction_filenames):
    name = name.replace(test_vid_dir, '')
    df_test.iloc[list(df_test['filename']).index(name), 1] = pred

print(preds.clip(0.35, 0.65).mean())
print(scipy.stats.median_absolute_deviation(preds.clip(0.35, 0.65))[0])
print(preds[:10])
#%%
print(df_test.head())
#%%
df_test.to_csv('submission.csv', index=False)

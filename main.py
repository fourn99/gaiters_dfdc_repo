import numpy as np
import os
import glob
from time import process_time
from imutils import face_utils
import dlib
import pandas as pd
from sklearn.metrics import log_loss
from keras import Model, Sequential
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import shutil
import random
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout


# %%


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
            offset = (299 - face_w)/2
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
            offset = (299 - face_h)/2
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


def detect_video(video_path, video_name, frames_to_capture, destination, net_ogj):
    count = 0
    # capture the video into frames
    vid = cv2.VideoCapture(video_path + video_name)
    video_name = video_name.replace('.mp4', '')
    os.makedirs(destination + video_name + '_frames' + '\\', exist_ok=True)
    list_frames = []
   # j = 0
    while True:
        ret, cap = vid.read()  # Capture frame-by-frame
        if cap is not None:
            # number of faces detected in frame

            cr = image_face_detector(cap, net_ogj)
            for i in range(len(cr)):

                frame = cap[int(cr[i][0][1]): int(cr[i][1][1]), int(cr[i][0][0]):int(cr[i][1][0])]
                list_frames.append(frame)

               # cv2.resize(frame, (299, 299))

               # cv2.imwrite(destination + video_name + '_frames' + '\\' + video_name + "_cropped_frame_%d.jpg" % j,
                    #       frame)
               # j+=1

                # sets nect frame to the 30th next frame
            count += frames_to_capture
            #print(count)
            vid.set(1, count)

        else:
            vid.release()
            break
    return list_frames


def video_to_frames(frames_interval):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    net = cv2.dnn.readNetFromCaffe('./deploy.prototxt.txt', './res10_300x300_ssd_iter_140000.caffemodel')
    # source folder with all videos
    all_train_dir = 'D:\\Deep_Fake\\dfdc_train_all\\'

    # array of all the subdirectories
    vid_sub_dir = [all_train_dir + x for x in os.listdir(all_train_dir)]
    test_video_files = []
    os.makedirs('./data/deepfake_jpegs', exist_ok=True)
    os.makedirs('./data/deepfake_features', exist_ok=True)

    # Inception V3 model for feature extraction
    base_mdl = InceptionV3(input_shape=(299,299,3), weights='imagenet', include_top=True)
    # only use model up to last avg_pool
    mdl = Model(inputs=base_mdl.input, outputs=base_mdl.get_layer('avg_pool').output)  # output size (None, 2048)

    # going through each subdirectory
    for i in range(len(vid_sub_dir)):
        # go inside folder with videos
    #for i in range(1):
        test_video_dir = vid_sub_dir[i] + '\\' + str(os.listdir(vid_sub_dir[i])[0]) + '\\'
        # e.g.: test_video_dir[0] -> D:\Deep_Fake\dfdc_train_all\dfdc_train_part_00\dfdc_train_part_0\

        # an array of all the videos in tht sub directory
        test_video_files = os.listdir(test_video_dir)

        # makes directory for the subdirectory of the training video
        os.makedirs('./data/deepfake_jpegs/' + str(os.listdir(vid_sub_dir[i])[0]), exist_ok=True)
        os.makedirs('./data/deepfake_features/'+ str(os.listdir(vid_sub_dir[i])[0]), exist_ok=True)
        # get directory name of the directory just made
        destination_dir = './data/deepfake_jpegs/' + str(os.listdir(vid_sub_dir[i])[0]) + '/'

        destination_dir_features = './data/deepfake_features/' + str(os.listdir(vid_sub_dir[i])[0]) + '/'

        # for each video in the training subdirectory
        for video in tqdm(test_video_files):
            #print(video)
            try:
                if video == 'metadata.json':
                    shutil.copyfile(test_video_dir + video, './data/deepfake_jpegs/metadata' + str(i) + '.json')
                # start = process_time()
                frames = detect_video(video_path=test_video_dir, video_name=video, frames_to_capture=frames_interval, destination=destination_dir, net_ogj=net)
                video = video.replace('.mp4', '')
                os.makedirs(destination_dir_features + video + '_features/', exist_ok=True)
                # create inpute sequence for LSTM to use later on
                sequence = []
                for img in frames:
                    x = np.expand_dims(img, axis=0)
                    x = preprocess_input(x)
                    features = mdl.predict(x)
                    sequence.append(features)

                np.save(destination_dir_features + video + '_features/' + video, sequence)
                # print("total time: ", process_time() - start)
            except Exception as err:
                print(err)


def get_path(num, x):
    path = './data/deepfake_jpegs/dfdc_train_part_' + str(num) + '/' + x.replace('.mp4', '_frames')
    if not os.path.exists(path):
        raise Exception

    if not (os.listdir(path)): #if empty delete subdirectory
        try:
            shutil.rmtree(path)
        except Exception as err:
            print(err)
        return -1

    return path


def preprocessing():
    # -- get metadata
    list_meta = sorted(glob.glob('./data/deepfake_jpegs/meta*'))
    df_trains = []
    df_vals = []
    counter = 0
    for meta_json in list_meta:
        if counter < 47:
            df_trains.append(pd.read_json(meta_json))
        else:
            df_vals.append(pd.read_json(meta_json))
        counter += 1
    nums = list(range(len(df_trains) + 1))
    LABELS = ['REAL', 'FAKE']
    val_nums = [47, 48, 49]

    # go through all dataframes, add path of video frames to paths and label to y
    paths = []
    y = []
    for df_train, num in tqdm(zip(df_trains, nums), total=len(df_trains)):
        images = list(df_train.columns.values)
        for x in images:
            try:
                p = get_path(num, x)
                if not (p == -1):  # if -1 then we didnt capture frames for that video
                    paths.append(p)
                    y.append(LABELS.index(df_train[x]['label']))
            except Exception as err:
                # print(err)
                pass

    val_paths = []
    val_y = []
    for df_val, num in tqdm(zip(df_vals, val_nums), total=len(df_vals)):
        images = list(df_val.columns.values)
        for x in images:
            try:
                p_val = get_path(num, x)
                if not (p_val == -1):  # if -1 then we didnt capture frames for that video
                    val_paths.append(p_val)
                    val_y.append(LABELS.index(df_val[x]['label']))
            except Exception as err:
                # print(err)
                pass

    print('There are ' + str(y.count(1)) + ' fake train samples')
    print('There are ' + str(y.count(0)) + ' real train samples')
    print('There are ' + str(val_y.count(1)) + ' fake val samples')
    print('There are ' + str(val_y.count(0)) + ' real val samples')
    print("Applying Underbalancing Technique")
    import random

    real = []
    fake = []
    for m, n in zip(paths, y):
        if n == 0:
            real.append(m)
        else:
            fake.append(m)
    fake = random.sample(fake, len(real))
    paths, y = [], []
    for x in real:
        paths.append(x)
        y.append(0)
    for x in fake:
        paths.append(x)
        y.append(1)

    print('There are ' + str(y.count(1)) + ' fake train samples')
    print('There are ' + str(y.count(0)) + ' real train samples')
    print('There are ' + str(val_y.count(1)) + ' fake val samples')
    print('There are ' + str(val_y.count(0)) + ' real val samples')

    real = []
    fake = []
    for m, n in zip(val_paths, val_y):
        if n == 0:
            real.append(m)
        else:
            fake.append(m)
    fake = random.sample(fake, len(real))
    val_paths, val_y = [], []
    for x in real:
        val_paths.append(x)
        val_y.append(0)
    for x in fake:
        val_paths.append(x)
        val_y.append(1)

    # Apply Underbalancing Techinique
    print('There are ' + str(y.count(1)) + ' fake train samples')
    print('There are ' + str(y.count(0)) + ' real train samples')
    print('There are ' + str(val_y.count(1)) + ' fake val samples')
    print('There are ' + str(val_y.count(0)) + ' real val samples')

    #  go throught all frames and convert img into array
    from keras.preprocessing.image import load_img, img_to_array
    def read_img(path):
        img = load_img(path, target_size=(299, 299))
        img = img_to_array(img)
        return img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))

    X = []  # input array for model
    for img in tqdm(paths):
        for frames in os.listdir(img): #todo this will failed since sequences are saved in same folder
            # print(img + '/' + frames)
            X.append(read_img(img + '/' + frames))
    val_X = []
    for img in tqdm(val_paths):
        for frames in os.listdir(img):
            # print(img + '/' + frames)
            X.append(read_img(img + '/' + frames))

    # shuffle train data
    def shuffle(X, y):
        new_train = []
        for m, n in zip(X, y):
            new_train.append([m, n])
        random.shuffle(new_train)
        X, y = [], []
        for x in new_train:
            X.append(x[0])
            y.append(x[1])
        return X, y

    X, y = shuffle(X, y)
    val_X, val_y = shuffle(val_X, val_y)

    return X, y, val_X, val_y


def define_model_lstm():

    learning_model = Sequential()
    learning_model.add(LSTM(2048, input_shape=(80,256)))  #input_shape = sequence length, feature vector length
    learning_model.add(Dense(2, activation='softmax'))
    #todo implemente fully connected 512 neurons networks for predictions
    return learning_model


def main():

    # 1) go through videos and save jpegs - Create sequence array in frames directory
    video_to_frames(frames_interval=25)  #todo for Chris run only this file and function

    # # 2) pre-processing todo probably need to modify to adapt to sequence
    # X_train, y_train, X_val, y_val = preprocessing()
    #
    # # 3) Get features of images using inception_v3 model
    #
    #
    # new_model = define_model_lstm()
    # print(new_model.summary())
    #
    # # 4) train model (LSTM) on feature vector
    # from keras.callbacks import LearningRateScheduler
    # lrs = [1e-3, 5e-4, 1e-4]
    #
    # def schedule(epoch):
    #     return lrs[epoch]
    #
    # LOAD_PRETRAIN = False
    #
    # import gc
    # kfolds = 5
    # losses = []
    # models = []
    # i = 0
    # while len(models) < kfolds:
    #     model = define_model((150, 150, 3))
    #     if i == 0:
    #         model.summary()
    #     model.fit([X], [y], epochs=2, callbacks=[LearningRateScheduler(schedule)])
    #     pred = model.predict([val_X])
    #     loss = log_loss(val_y, pred)
    #     losses.append(loss)
    #     print('fold ' + str(i) + ' model loss: ' + str(loss))
    #     if loss < 0.68:
    #         models.append(model)
    #     else:
    #         print('loss too bad, retrain!')
    #     K.clear_session()
    #     del model
    #     gc.collect()
    #     i += 1




if __name__ == "__main__":
    main()



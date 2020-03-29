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
from keras.callbacks import LearningRateScheduler


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


def detect_video(video_path, video_name, frames_to_capture, destination, net_ogj):
    count = 0
    # capture the video into frames
    vid = cv2.VideoCapture(video_path)
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
            # cv2.imwrite(destination + video_name + '_frames' + '\\' + video_name + "_cropped_frame_%d.jpg" % j,
            #       frame)
            # j+=1
            # sets nect frame to the 30th next frame
            count += frames_to_capture
            # print(count)
            vid.set(1, count)

        else:
            vid.release()
            break
    return list_frames


def video_to_frames(frames_interval, train_videos, train_labels, valid_videos, valid_labels):
    train_vid = train_videos
    valid_vid = valid_videos
    extract_list = [train_vid, valid_vid]
    extract_list_labels = [train_labels, valid_labels]

    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    net = cv2.dnn.readNetFromCaffe('./deploy.prototxt.txt', './res10_300x300_ssd_iter_140000.caffemodel')
    # source folder with all videos

    all_train_dir = 'D:\\Deep_Fake\\dfdc_train_all\\'
    #all_train_dir = 'E:\\dfdc_train_all\\'

    # array of all the subdirectories
    vid_sub_dir = [all_train_dir + x for x in os.listdir(all_train_dir)]

    os.makedirs('./data/deepfake_features', exist_ok=True)

    # Inception V3 model for feature extraction
    input_tensor = Input((299, 299, 3))
    base_mdl = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
    # only use model up to last avg_pool
    mdl = Model(inputs=base_mdl.input, outputs=base_mdl.get_layer('avg_pool').output)  # output size (None, 2048)

    # going through each data set
    video_dir_to_delete = []
    for i in range(len(extract_list)):
        temp_paths = extract_list[i]
        temp_labels = extract_list_labels[i]
        parent_dir = None
        if i == 0:
            os.makedirs('./data/deepfake_features/train', exist_ok=True)
            parent_dir = './data/deepfake_features/train/'

        else:
            os.makedirs('./data/deepfake_features/validation', exist_ok=True)
            parent_dir = './data/deepfake_features/validation/'

        # for each video
        for path, label in tqdm(zip(temp_paths[0:5], temp_labels[0:5]), total=len(temp_paths[0:5])):
            video_parts = path.split(
                '\\')  # split paths into array: ['E:','dfdc_train_all', 'dfdc_train_part_46', 'dfdc_train_part_46', 'iffowzafje.mp4']
            video_name = video_parts[4].replace('.mp4', '')

            if label == 0:
                video_name = video_name + '_REAL'
            else:
                video_name = video_name + '_FAKE'

            os.makedirs(parent_dir + video_name + '_feature_vector', exist_ok=True)
            destination_dir_features = parent_dir + video_name + '_feature_vector/'
            try:
                # if video == 'metadata.json':
                #     shutil.copyfile(test_video_dir + video, './data/deepfake_jpegs/metadata' + str(i) + '.json')
                # start = process_time()
                frames = detect_video(video_path=path, video_name=video_parts[4], frames_to_capture=frames_interval,
                                      destination=destination_dir_features, net_ogj=net)
                if len(frames) < 12:
                    video_dir_to_delete.append(destination_dir_features)
                    continue

                # create input sequence for LSTM to use later on
                sequence = []
                for img in frames:
                    x = np.expand_dims(img, axis=0)
                    x = preprocess_input(x)
                    features = mdl.predict(x)
                    sequence.append(features[0])

                if len(sequence) < 12:
                    video_dir_to_delete.append(destination_dir_features)
                    continue
                else:
                    np.save(destination_dir_features + video_name, sequence)
                # print("total time: ", process_time() - start)
            except Exception as err:
                print(err)

    for dir in video_dir_to_delete:
        os.rmdir(dir)
        print("Deleted: " + dir)

    return video_dir_to_delete


def get_path_videos_nic_computer(subdir_num, video_name):
    if subdir_num < 10:
        path = 'E:\\dfdc_train_all\\dfdc_train_part_0' + str(subdir_num) + '\\' + 'dfdc_train_part_' + str(
            subdir_num) + '\\' + video_name
    else:
        path = 'E:\\dfdc_train_all\\dfdc_train_part_' + str(subdir_num) + '\\' + 'dfdc_train_part_' + str(
            subdir_num) + '\\' + video_name

    if not os.path.exists(path):
        return -1
        # raise Exception
    return path


def get_path_videos(subdir_num, video_name):
    if subdir_num < 10:
        path = 'D:\\Deep_Fake\\dfdc_train_all\\dfdc_train_part_0' + str(subdir_num) + '\\' + 'dfdc_train_part_' + str(
            subdir_num) + '\\' + video_name
    else:
        path = 'D:\\Deep_Fake\\dfdc_train_all\\dfdc_train_part_' + str(subdir_num) + '\\' + 'dfdc_train_part_' + str(
            subdir_num) + '\\' + video_name

    if not os.path.exists(path):
        return -1
        # raise Exception
    return path


def preprocessing(metadata_dir):
    # -- get metadata
    df_train0 = pd.read_json(metadata_dir + 'metadata0.json')
    df_train1 = pd.read_json(metadata_dir + 'metadata1.json')
    df_train2 = pd.read_json(metadata_dir + 'metadata2.json')
    df_train3 = pd.read_json(metadata_dir + 'metadata3.json')
    df_train4 = pd.read_json(metadata_dir + 'metadata4.json')
    df_train5 = pd.read_json(metadata_dir + 'metadata5.json')
    df_train6 = pd.read_json(metadata_dir + 'metadata6.json')
    df_train7 = pd.read_json(metadata_dir + 'metadata7.json')
    df_train8 = pd.read_json(metadata_dir + 'metadata8.json')
    df_train9 = pd.read_json(metadata_dir + 'metadata9.json')
    df_train10 = pd.read_json(metadata_dir + 'metadata10.json')
    df_train11 = pd.read_json(metadata_dir + 'metadata11.json')
    df_train12 = pd.read_json(metadata_dir + 'metadata12.json')
    df_train13 = pd.read_json(metadata_dir + 'metadata13.json')
    df_train14 = pd.read_json(metadata_dir + 'metadata14.json')
    df_train15 = pd.read_json(metadata_dir + 'metadata15.json')
    df_train16 = pd.read_json(metadata_dir + 'metadata16.json')
    df_train17 = pd.read_json(metadata_dir + 'metadata17.json')
    df_train18 = pd.read_json(metadata_dir + 'metadata18.json')
    df_train19 = pd.read_json(metadata_dir + 'metadata19.json')
    df_train20 = pd.read_json(metadata_dir + 'metadata20.json')
    df_train21 = pd.read_json(metadata_dir + 'metadata21.json')
    df_train22 = pd.read_json(metadata_dir + 'metadata22.json')
    df_train23 = pd.read_json(metadata_dir + 'metadata23.json')
    df_train24 = pd.read_json(metadata_dir + 'metadata24.json')
    df_train25 = pd.read_json(metadata_dir + 'metadata25.json')
    df_train26 = pd.read_json(metadata_dir + 'metadata26.json')
    df_train27 = pd.read_json(metadata_dir + 'metadata27.json')
    df_train28 = pd.read_json(metadata_dir + 'metadata28.json')
    df_train29 = pd.read_json(metadata_dir + 'metadata29.json')
    df_train30 = pd.read_json(metadata_dir + 'metadata30.json')
    df_train31 = pd.read_json(metadata_dir + 'metadata31.json')
    df_train32 = pd.read_json(metadata_dir + 'metadata32.json')
    df_train33 = pd.read_json(metadata_dir + 'metadata33.json')
    df_train34 = pd.read_json(metadata_dir + 'metadata34.json')
    df_train35 = pd.read_json(metadata_dir + 'metadata35.json')
    df_train36 = pd.read_json(metadata_dir + 'metadata36.json')
    df_train37 = pd.read_json(metadata_dir + 'metadata37.json')
    df_train38 = pd.read_json(metadata_dir + 'metadata38.json')
    df_train39 = pd.read_json(metadata_dir + 'metadata39.json')
    df_train40 = pd.read_json(metadata_dir + 'metadata40.json')
    df_train41 = pd.read_json(metadata_dir + 'metadata41.json')
    df_train42 = pd.read_json(metadata_dir + 'metadata42.json')
    df_train43 = pd.read_json(metadata_dir + 'metadata43.json')
    df_train44 = pd.read_json(metadata_dir + 'metadata44.json')
    df_train45 = pd.read_json(metadata_dir + 'metadata45.json')
    df_train46 = pd.read_json(metadata_dir + 'metadata46.json')
    df_val1 = pd.read_json(metadata_dir + 'metadata47.json')
    df_val2 = pd.read_json(metadata_dir + 'metadata48.json')
    df_val3 = pd.read_json(metadata_dir + 'metadata49.json')

    df_trains = [df_train0, df_train1, df_train2, df_train3, df_train4,
                 df_train5, df_train6, df_train7, df_train8, df_train9, df_train10,
                 df_train11, df_train12, df_train13, df_train14, df_train15, df_train16,
                 df_train17, df_train18, df_train19, df_train20, df_train21, df_train22,
                 df_train23, df_train24, df_train25, df_train26, df_train27, df_train28,
                 df_train29, df_train30, df_train31, df_train32, df_train33, df_train34,
                 df_train35, df_train36, df_train37, df_train38, df_train39, df_train40,
                 df_train41, df_train42, df_train43, df_train44, df_train45, df_train46]

    df_vals = [df_val1, df_val2, df_val3]
    nums = list(range(len(df_trains)))
    LABELS = ['REAL', 'FAKE']
    val_nums = [47, 48, 49]

    # go through all dataframes, add path of videos to paths and label to y
    paths = []
    y = []
    not_found = []
    for df_train, num in tqdm(zip(df_trains, nums), total=len(df_trains)):
        images = list(df_train.columns.values)
        for x in images:
            try:
                p = get_path_videos(num, x)
                if not (p == -1):  # if -1 then we didnt find video
                    paths.append(p)
                    y.append(LABELS.index(df_train[x]['label']))

                else:
                    not_found.append((num, x))
            except Exception as err:
                # print(err)
                pass

    val_paths = []
    val_y = []
    val_not_found = []
    for df_val, num in tqdm(zip(df_vals, val_nums), total=len(df_vals)):
        images = list(df_val.columns.values)
        for x in images:
            try:
                p_val = get_path_videos(num, x)
                if not (p_val == -1):  # if -1 then we didnt find video
                    val_paths.append(p_val)
                    val_y.append(LABELS.index(df_val[x]['label']))
                else:
                    val_not_found.append((num, x))
            except Exception as err:
                # print(err)
                pass

    return balancing(paths, y, val_paths, val_y)


def balancing(train_paths, train_labels, valid_paths, valid_labels):

    paths = train_paths
    y = train_labels
    val_paths = valid_paths
    val_y = valid_labels

    print('There are ' + str(y.count(1)) + ' fake train samples')
    print('There are ' + str(y.count(0)) + ' real train samples')
    print('There are ' + str(val_y.count(1)) + ' fake val samples')
    print('There are ' + str(val_y.count(0)) + ' real val samples')

    import random
    print("\nApplying Underbalancing Technique")
    print("Underbalancing - Training")
    real = []
    fake = []
    for m, n in zip(paths, y):
        if n == 0:
            real.append(m)
        else:
            fake.append(m)
    fake = random.sample(real, len(fake))
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

    print("\nApplying Underbalancing Technique")
    print("Underbalancing - Validation")
    real = []
    fake = []
    for m, n in zip(val_paths, val_y):
        if n == 0:
            real.append(m)
        else:
            fake.append(m)
    fake = random.sample(real, len(fake))
    val_paths, val_y = [], []
    for x in real:
        val_paths.append(x)
        val_y.append(0)
    for x in fake:
        val_paths.append(x)
        val_y.append(1)

    print('There are ' + str(y.count(1)) + ' fake train samples')
    print('There are ' + str(y.count(0)) + ' real train samples')
    print('There are ' + str(val_y.count(1)) + ' fake val samples')
    print('There are ' + str(val_y.count(0)) + ' real val samples')

    return paths, y, val_paths, val_y


def define_model_lstm():
    learning_model = Sequential()
    learning_model.add(
        LSTM(2048, input_shape=(12, 2048), dropout=0.5))  # input_shape = sequence length, feature vector length
    learning_model.add(Dense(512, activation='relu'))
    learning_model.add(Dropout(0.5))
    learning_model.add(Dense(2, activation='softmax'))
    learning_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-4))

    return learning_model


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


def get_feature_vector():
    list_folder_features_train = os.listdir('./data/deepfake_features/train/')
    list_features_train = []
    list_labels_train = []

    for vector in tqdm(list_folder_features_train):
        temp = vector.split('_')
        name = temp[0] + '_' + temp[1]
        label = temp[1]
        vector_dir = './data/deepfake_features/train/' + vector + '/' + name + '.npy'
        # print(np.load(vector_dir).shape)
        list_features_train.append(np.load(vector_dir))

        if label == 'REAL':
            list_labels_train.append(0)
        else:
            list_labels_train.append(1)

    # Validation Set
    list_folder_features_validation = os.listdir('./data/deepfake_features/validation/')
    list_features_validation = []
    list_labels_validation = []

    for vector in tqdm(list_folder_features_validation):
        temp = vector.split('_')
        name = temp[0] + '_' + temp[1]
        label = temp[1]
        vector_dir = './data/deepfake_features/validation/' + vector + '/' + name + '.npy'
        list_features_validation.append(np.load(vector_dir))

        if label == 'REAL':
            list_labels_validation.append(0)
        else:
            list_labels_validation.append(1)

    X, y, val_X, val_y = list_features_train, list_labels_train, list_features_validation, list_labels_validation

    X, y = shuffle(X, y)
    val_X, val_y = shuffle(val_X, val_y)

    return X, y, val_X, val_y


def train(model, X, y, val_X, val_y):
    def schedule(epoch):
        return lrs[epoch]

    lrs = [1e-3, 5e-4, 1e-4]
    LOAD_PRETRAIN = False

    import gc
    kfolds = 5
    losses = []
    models = []
    i = 0
    while len(models) < kfolds:
        model = define_model_lstm()
        if i == 0:
            model.summary()
        model.fit(X, y, epochs=2, callbacks=[LearningRateScheduler(schedule)])
        pred = model.predict(val_X)
        loss = log_loss(val_y, pred)
        losses.append(loss)
        print('fold ' + str(i) + ' model loss: ' + str(loss))
        if loss < 0.5:
            models.append(model)
        else:
            print('loss too bad, retrain!')
        K.clear_session()
        del model
        gc.collect()
        i += 1

    return models, losses


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


def larger_range(model_pred, time):
    return (((model_pred - 0.5) * time) + 0.5)


def check_answers(pred, real, num):
    for i, (x, y) in enumerate(zip(pred, real)):
        correct_incorrect = 'correct ✅ ' if round(float(x), 0) == round(float(y), 0) else 'incorrect❌'
        print(correct_incorrect + ' prediction: ' + str(x[0]) + ', answer: ' + str(y))
        if i > num:
            return


def correct_precentile(pred, real):
    correct = 0
    incorrect = 0
    for x, y in zip(pred, real):
        if round(float(x), 0) == round(float(y), 0):
            correct += 1
        else:
            incorrect += 1
    print('number correct: ' + str(correct) + ', number incorrect: ' + str(incorrect))
    print(str(round(correct / len(real) * 100, 1)) + '% correct' + ', ' + str(
        round(incorrect / len(real) * 100, 1)) + '% incorrect')



def main():

    # 1) get metadata of videos, split into train - validation sets, returns paths and labels
    #videos_path, videos_label, valid_videos_path, valid_videos_labels = preprocessing('E:\\dfdc_train_all\\')

    # 2) go through videos and save jpegs - Create feature vector with inception v3 in feature directory
    #vids_to_delete = video_to_frames(25, videos_path, videos_label, valid_videos_path, valid_videos_labels)


    # 3) get feature vectors for train and validation sets
    X, y, val_X, val_y = get_feature_vector()

    # 4) Balancing
    #X, y, val_X, val_y = balancing(X, y, val_X, val_y)
    # %%
    count = 0
    x_train, y_train = [], []
    for i in range(len(X)):
        temp = X[i]
        if X[i].shape[0] > 12:
            continue
        else:
            y_train.append(y[i])
            for j in range(len(temp)):
                temp_1 = temp[j]
                for k in range(len(temp_1)):
                    x_train.append(X[i][j][k])

    x_val, y_val = [], []
    for i in range(len(val_X)):
        temp = val_X[i]
        if val_X[i].shape[0] > 12:
            continue
        else:
            y_val.append(val_y[i])
            for j in range(len(temp)):
                temp_1 = temp[j]
                for k in range(len(temp_1)):
                    x_val.append(val_X[i][j][k])

    # 5) train model (LSTM) on feature vector
    lstm = define_model_lstm()
    list_models, list_losses = train(lstm, x_train, y_train, x_val, y_val)

    # 6) predict validation set
    best_model_pred = list_models[list_losses.index(min(list_losses))].predict([val_X])
    best_model = list_models[list_losses.index(min(list_losses))]
    model_pred = prediction_pipline(val_X, [best_model])

    # 7) Check performance of models
    random_pred = np.random.random(len(val_X))
    print('random loss: ' + str(log_loss(val_y, random_pred.clip(0.35, 0.65))))
    allone_pred = np.array([1 for _ in range(len(val_X))])
    print('1 loss: ' + str(log_loss(val_y, allone_pred)))
    allzero_pred = np.array([0 for _ in range(len(val_X))])
    print('0 loss: ' + str(log_loss(val_y, allzero_pred)))
    allpoint5_pred = np.array([0.5 for _ in range(len(val_X))])
    print('0.5 loss: ' + str(log_loss(val_y, allpoint5_pred)))

    print('Simple Averaging Loss: ' + str(log_loss(val_y, model_pred.clip(0.35, 0.65))))
    print(
        'Two Times Larger Range(Averaging) Loss: ' + str(log_loss(val_y, larger_range(model_pred, 2).clip(0.35, 0.65))))
    print('Best Single Model Loss: ' + str(log_loss(val_y, best_model_pred.clip(0.35, 0.65))))
    print('Two Times Larger Range(Single Model) Loss: ' + str(
        log_loss(val_y, larger_range(best_model_pred, 2).clip(0.35, 0.65))))

    if log_loss(val_y, model_pred.clip(0.35, 0.65)) < log_loss(val_y, larger_range(model_pred, 2).clip(0.35, 0.65)):
        two_times = False
        print('simple averaging is better')
    else:
        two_times = True
        print('two times larger range is better')
    two_times = False  # This is not a bug. I did this intentionally because the model can't get most of the private validation set right(based on LB)

    import scipy

    print(model_pred.clip(0.35, 0.65).mean())
    print(scipy.stats.median_absolute_deviation(model_pred.clip(0.35, 0.65))[0])

    check_answers(model_pred, val_y, 15)
    correct_precentile(model_pred, val_y)

    # 8) save models and weights
    model_json = best_model.to_json()
    with open("best_model.json", "w") as json_file:
        json_file.write(model_json)
        # serialize weights to HDF5
    best_model.save_weights("best_model_weights.h5")
    print("Saved best model to disk")
    json_file.close()

    # 8) save models and weights
    # serialize model to JSON
    from keras.models import model_from_json
    # i = 0
    # for model in tqdm(list_models):
    #
    #     model_json = model.to_json()
    #     with open("./models/model_%d.json" % i, "w") as json_file:
    #         json_file.write(model_json)
    #         # serialize weights to HDF5
    #     model.save_weights("./weights/model_%d.h5" % i)
    #     print("Saved model " + str(i) + " to disk")
    #     json_file.close()



if __name__ == "__main__":
    main()



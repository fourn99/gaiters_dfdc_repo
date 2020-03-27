
import pandas as pd
import keras
import os
import numpy as np
from sklearn.metrics import log_loss
from keras import Model, Sequential
from keras.layers import *
from keras.optimizers import *
from sklearn.model_selection import train_test_split
import cv2
from tqdm import tqdm
import glob
from mtcnn import MTCNN
import shutil



#%%

sorted(glob.glob('./data/deepfake_jpegs/meta*'))

#%%
df_train0 = pd.read_json('./data/deepfake_jpegs/metadata0.json')


df_trains = [df_train0]

# df_vals = [df_val1, df_val2, df_val3]
nums = list(range(len(df_trains) + 1))
LABELS = ['REAL', 'FAKE']
# val_nums = [47, 48, 49]

#%%

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

# go through all dataframes, add path of video frames to paths and label to y
paths = []
y = []
for df_train, num in tqdm(zip(df_trains, nums), total=len(df_trains)):
    images = list(df_train.columns.values)
    for x in images:
        try:
            p = get_path(num, x)
            if not(p == -1): # if -1 then we didnt capture frames for that video
                paths.append(p)
                y.append(LABELS.index(df_train[x]['label']))
        except Exception as err:
            # print(err)
            pass

# val_paths = []
# val_y = []
# for df_val, num in tqdm(zip(df_vals, val_nums), total=len(df_vals)):
#     images = list(df_val.columns.values)
#     for x in images:
#         try:
#             p_val = get_path(num, x)
#             if not (p_val == -1):  # if -1 then we didnt capture frames for that video
#                 val_paths.append(p_val)
#                 val_y.append(LABELS.index(df_val[x]['label']))
#         except Exception as err:
#             # print(err)
#             pass

# %%

print('There are ' + str(y.count(1)) + ' fake train samples')
print('There are ' + str(y.count(0)) + ' real train samples')
# print('There are ' + str(val_y.count(1)) + ' fake val samples')
# print('There are ' + str(val_y.count(0)) + ' real val samples')


# %%
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
# Rebvalance Validation
# real = []
# fake = []
# for m, n in zip(val_paths, val_y):
#     if n == 0:
#         real.append(m)
#     else:
#         fake.append(m)
# fake = random.sample(fake, len(real))
# val_paths, val_y = [], []
# for x in real:
#     val_paths.append(x)
#     val_y.append(0)
# for x in fake:
#     val_paths.append(x)
#     val_y.append(1)
#

# %%

print('There are ' + str(y.count(1)) + ' fake train samples')
print('There are ' + str(y.count(0)) + ' real train samples')
# print('There are ' + str(val_y.count(1)) + ' fake val samples')
# print('There are ' + str(val_y.count(0)) + ' real val samples')


#%%

from keras.preprocessing.image import load_img, img_to_array
def read_img(path):
    img = load_img(path, target_size=(299, 299))
    img = img_to_array(img)
    return img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    # return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


X = []  # input array for model
for img in tqdm(paths):
    for frames in os.listdir(img):
        # print(img + '/' + frames)
        X.append(read_img(img + '/' + frames))
# val_X = []
# for img in tqdm(val_paths):
#     for frames in os.listdir(img):
#         # print(img + '/' + frames)
#         X.append(read_img(img + '/' + frames))

# %%

import random


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
# val_X, val_y = shuffle(val_X, val_y)

#%%

from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Dropout
import numpy as np


def extract_features(sample_count, im_shape=(299,299,3)):
    features = np.zeros((sample_count, 8, 8, 2048))
    labels = np.zeros((sample_count, 2))
    base_mdl = InceptionV3(input_shape=im_shape, weights='imagenet', include_top=True)
    # extract feetures at last layer mixed10
    mdl = keras.Model(inputs=base_mdl.input,
                      outputs=base_mdl.get_layer('avg_pool').output)  # output size (None, 8, 8, 2048)



    # model_1 = Sequential()
    # model_1.add(Dense(256, activation='relu', input_dim = (8 * 8 * 2048)))
    # model_1.add(Dropout(0.5))
    # model_1.add(Dense(2, activation='softmax'))
    # # return model_1
    # model_2 = Sequential()
    # model_2.add(mdl)
    # model_2.add(Flatten())
    # model_2.add(model_1.layers[0])  # output format 256

    return mdl

model_incept = extract_features(300)
print(model_incept.summary())


#%%
# df_model = define_model()
# df_model.load_weights('./data/meso-pretrain/MesoInception_DF')
# f2f_model = define_model()
# f2f_model.load_weights('./data/meso-pretrain/MesoInception_F2F')

# %%
from keras.callbacks import LearningRateScheduler

lrs = [1e-3, 5e-4, 1e-4]

def schedule(epoch):
    return lrs[epoch]

# %%

LOAD_PRETRAIN = False

# %%

import gc

kfolds = 5
losses = []
if LOAD_PRETRAIN:
    # import keras.backend as K
    df_models = []
    f2f_models = []
    i = 0
    while len(df_models) < kfolds:
        model = define_model((150, 150, 3))
        if i == 0:
            model.summary()
        # model.load_weights('../input/meso-pretrain/MesoInception_DF')
        for new_layer, layer in zip(model.layers[1:-8], df_model.layers[1:-8]):
            new_layer.set_weights(layer.get_weights())
        model.fit([X], [y], epochs=2, callbacks=[LearningRateScheduler(schedule)])
        pred = model.predict([val_X])
        loss = log_loss(val_y, pred)
        losses.append(loss)
        print('fold ' + str(i) + ' model loss: ' + str(loss))
        df_models.append(model)
        K.clear_session()
        del model
        gc.collect()
        i += 1
    i = 0
    while len(f2f_models) < kfolds:
        model = define_model((150, 150, 3))
        # model.load_weights('../input/meso-pretrain/MesoInception_DF')
        for new_layer, layer in zip(model.layers[1:-8], f2f_model.layers[1:-8]):
            new_layer.set_weights(layer.get_weights())
        model.fit([X], [y], epochs=2, callbacks=[LearningRateScheduler(schedule)])
        pred = model.predict([val_X])
        loss = log_loss(val_y, pred)
        losses.append(loss)
        print('fold ' + str(i) + ' model loss: ' + str(loss))
        f2f_models.append(model)
        K.clear_session()
        del model
        gc.collect()
        i += 1
        models = f2f_models + df_models
else:
    models = []
    i = 0
    while len(models) < kfolds:
        model = define_model((150, 150, 3))
        if i == 0:
            model.summary()
        model.fit([X], [y], epochs=2, callbacks=[LearningRateScheduler(schedule)])
        pred = model.predict([val_X])
        loss = log_loss(val_y, pred)
        losses.append(loss)
        print('fold ' + str(i) + ' model loss: ' + str(loss))
        if loss < 0.68:
            models.append(model)
        else:
            print('loss too bad, retrain!')
        K.clear_session()
        del model
        gc.collect()
        i += 1

# %%
def prediction_pipline(X, two_times=False):
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


# %%

best_model_pred = models[losses.index(min(losses))].predict([val_X])
#%%
# create procedure to getweights of best model and save them

# %%
model_pred = prediction_pipline(val_X)

# %%

random_pred = np.random.random(len(val_X))
print('random loss: ' + str(log_loss(val_y, random_pred.clip(0.35, 0.65))))
allone_pred = np.array([1 for _ in range(len(val_X))])
print('1 loss: ' + str(log_loss(val_y, allone_pred)))
allzero_pred = np.array([0 for _ in range(len(val_X))])
print('0 loss: ' + str(log_loss(val_y, allzero_pred)))
allpoint5_pred = np.array([0.5 for _ in range(len(val_X))])
print('0.5 loss: ' + str(log_loss(val_y, allpoint5_pred)))


# %%

print('Simple Averaging Loss: ' + str(log_loss(val_y, model_pred.clip(0.35, 0.65))))
print('Two Times Larger Range(Averaging) Loss: ' + str(log_loss(val_y, larger_range(model_pred, 2).clip(0.35, 0.65))))
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
#%%
import scipy

print(model_pred.clip(0.35, 0.65).mean())
print(scipy.stats.median_absolute_deviation(model_pred.clip(0.35, 0.65))[0])


# %%

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


check_answers(model_pred, val_y, 15)
correct_precentile(model_pred, val_y)

# %%

del X, y, val_X, val_y

# %%
#todo addapt following procedure to read and save test videos

MAX_SKIP = 10
NUM_FRAME = 150
test_dir = 'C:\\Users\\admin\\PycharmProjects\\gaiters_dfdc_repo\\data\\test_videos\\'
filenames = os.listdir(test_dir)
prediction_filenames = filenames
test_video_files = [test_dir + x for x in filenames]
detector = MTCNN()


def detect_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final = []
    detected_faces_raw = detector.detect_faces(img)
    if detected_faces_raw == []:
        # print('no faces found')
        return []
    confidences = []
    for n in detected_faces_raw:
        x, y, w, h = n['box']
        final.append([x, y, w, h])
        confidences.append(n['confidence'])
    if max(confidences) < 0.9:
        return []
    max_conf_coord = final[confidences.index(max(confidences))]
    # return final
    return max_conf_coord


def crop(img, x, y, w, h):
    x -= 40
    y -= 40
    w += 80
    h += 80
    if x < 0:
        x = 0
    if y <= 0:
        y = 0
    return cv2.cvtColor(cv2.resize(img[y:y + h, x:x + w], (150, 150)), cv2.COLOR_BGR2RGB)


def detect_video(video):
    v_cap = cv2.VideoCapture(video)
    v_cap.set(1, NUM_FRAME)
    success, vframe = v_cap.read()
    vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
    bounding_box = detect_face(vframe)
    if bounding_box == []:
        count = 0
        current = NUM_FRAME
        while bounding_box == [] and count < MAX_SKIP:
            current += 1
            v_cap.set(1, current)
            success, vframe = v_cap.read()
            vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
            bounding_box = detect_face(vframe)
            count += 1
        if bounding_box == []:
            print('no faces found')
            prediction_filenames.remove(video.replace('C:\\Users\\admin\\PycharmProjects\\gaiters_dfdc_repo\\data\\test_videos\\', ''))
            return None
    x, y, w, h = bounding_box
    v_cap.release()
    return crop(vframe, x, y, w, h)


test_X = []
for video in tqdm(test_video_files):
    x = detect_video(video)
    if x is None:
        continue
    test_X.append(x)

# %%
# create procedure to fetch model and set weight to our pretrained weigths


#%%

df_test = pd.read_csv('C:\\Users\\admin\\PycharmProjects\\gaiters_dfdc_repo\\data\\sample_submission.csv')
df_test['label'] = 0.5
preds = prediction_pipline(test_X, two_times=two_times).clip(0.35, 0.65)
for pred, name in zip(preds, prediction_filenames):
    name = name.replace('C:\\Users\\admin\\PycharmProjects\\gaiters_dfdc_repo\\data\\test_videos', '')
    df_test.iloc[list(df_test['filename']).index(name), 1] = pred

# %%

print(preds.clip(0.35, 0.65).mean())
print(scipy.stats.median_absolute_deviation(preds.clip(0.35, 0.65))[0])
print(preds[:10])

# %%

df_test.head()

# %%

df_test.to_csv('submission.csv', index=False)

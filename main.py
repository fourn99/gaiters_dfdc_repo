# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import os
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
# %matplotlib inline
import cv2 as cv

DATA_FOLDER = 'C:\\Users\\Malou\\Desktop\\gaiters_dfdc_repo\\data'

TRAIN_SAMPLE_FOLDER = 'train_sample_videos'
TEST_FOLDER = 'test_videos'
print("Malou's push")
print(f"Train samples: {len(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))}")
# print(f"Test samples: {len(os.listdir(os.path.join(DATA_FOLDER, TEST_FOLDER)))}")


# FACE_DETECTION_FOLDER = '../input/haar-cascades-for-face-detection'
# print(f"Face detection resources: {os.listdir(FACE_DETECTION_FOLDER)}")

# what type of extensions ?
train_list = list(os.listdir(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER)))
ext_dict = []
for file in train_list:
    file_ext = file.split('.')[1]
    if file_ext not in ext_dict:
        ext_dict.append(file_ext)
print(f"Extensions: {ext_dict}")

# how many files with extensions are there ?
for file_ext in ext_dict:
    print(f"Files with extension `{file_ext}`: {len([file for file in train_list if file.endswith(file_ext)])}")

json_file = [file for file in train_list if file.endswith('json')][0]
print(f"JSON file: {json_file}")


def get_meta_from_json(path):
    df = pd.read_json(os.path.join(DATA_FOLDER, path, json_file))
    df = df.T
    return df


meta_train_df = get_meta_from_json(TRAIN_SAMPLE_FOLDER)
meta_train_df.head()


# missing data exploration
def missing_data(data):
    total = data.isnull().sum()
    percent = (data.isnull().sum() / data.isnull().count() * 100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    return np.transpose(tt)


missing_data(meta_train_df)


# -- Unique Values
def unique_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in data.columns:
        unique = data[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return np.transpose(tt)


unique_values(meta_train_df)


# -- Most Frequent Originals
def most_frequent_values(data):
    total = data.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in data.columns:
        itm = data[col].value_counts().index[0]
        val = data[col].value_counts().values[0]
        items.append(itm)
        vals.append(val)
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    return np.transpose(tt)


most_frequent_values(meta_train_df)


# -- Data Distribution
def plot_count(feature, title, df, size=1):
    """
    Plot count of classes / feature
    param: feature - the feature to analyze
    param: title - title to add to the graph
    param: df - dataframe from which we plot feature's classes distribution
    param: size - default 1.
    """
    f, ax = plt.subplots(1, 1, figsize=(4 * size, 4))
    total = float(len(df))
    g = sns.countplot(df[feature], order=df[feature].value_counts().index[:20], palette='Set3')
    g.set_title("Number and percentage of {}".format(title))
    if size > 2:
        plt.xticks(rotation=90, size=8)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width() / 2.,
                height + 3,
                '{:1.2f}%'.format(100 * height / total),
                ha="center")
    plt.show()


plot_count('split', 'split (train)', meta_train_df)
plot_count('label', 'label (train)', meta_train_df)

#
# meta = np.array(list(meta_train_df.index))
# storage = np.array([file for file in train_list if  file.endswith('mp4')])
# print(f"Metadata: {meta.shape[0]}, Folder: {storage.shape[0]}")
# print(f"Files in metadata and not in folder: {np.setdiff1d(meta,storage,assume_unique=False).shape[0]}")
# print(f"Files in folder and not in metadata: {np.setdiff1d(storage,meta,assume_unique=False).shape[0]}")
#
#
# fake_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='FAKE'].sample(3).index)
# print(fake_train_sample_video)
#
#
#
# def display_image_from_video(video_path):
#     '''
#     input: video_path - path for video
#     process:
#     1. perform a video capture from the video
#     2. read the image
#     3. display the image
#     '''
#     capture_image = cv.VideoCapture(video_path)
#     ret, frame = capture_image.read()
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111)
#     frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     ax.imshow(frame)
#
# for video_file in fake_train_sample_video:
#     display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
#
# real_train_sample_video = list(meta_train_df.loc[meta_train_df.label=='REAL'].sample(3).index)
# print(real_train_sample_video)
#
# for video_file in real_train_sample_video:
#     display_image_from_video(os.path.join(DATA_FOLDER, TRAIN_SAMPLE_FOLDER, video_file))
#
#
#
# def display_image_from_video_list(video_path_list, video_folder=TRAIN_SAMPLE_FOLDER):
#     '''
#     input: video_path_list - path for video
#     process:
#     0. for each video in the video path list
#         1. perform a video capture from the video
#         2. read the image
#         3. display the image
#     '''
#     plt.figure()
#     fig, ax = plt.subplots(2,3,figsize=(16,8))
#     # we only show images extracted from the first 6 videos
#     for i, video_file in enumerate(video_path_list[0:6]):
#         video_path = os.path.join(DATA_FOLDER, video_folder,video_file)
#         capture_image = cv.VideoCapture(video_path)
#         ret, frame = capture_image.read()
#         frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#         ax[i//3, i%3].imshow(frame)
#         ax[i//3, i%3].set_title(f"Video: {video_file}")
#         ax[i//3, i%3].axis('on')
#
# same_original_fake_train_sample_video = list(meta_train_df.loc[meta_train_df.original=='meawmsgiti.mp4'].index)
# display_image_from_video_list(same_original_fake_train_sample_video)

import shutil
import numpy as np
import cv2
import os
import glob
from time import process_time
import os
import glob
from time import process_time
from imutils import face_utils
import dlib
import cv2
import tqdm

def image_face_detector(confidence, image, path_model, path_prototxt):
    """
    :param confidence: Certainty of detection
    :param image_paths: all images to be processed
    :param path_model: the detecting model path (caffemodel)
    :param path_prototxt: the prototxt file path
    :return: face_detection_coordinates
    Desc: On image input, applies face detection and confidence level of detection
    """
    net = cv2.dnn.readNetFromCaffe(path_prototxt, path_model)

    face_detection_coordinates = (1, 3, 2)  # fixed value for now
    face_detection_coordinates = np.zeros(face_detection_coordinates)

    # load the input image and construct an input blob for the image
    # by resizing to a fixed 300x300 pixels and then normalizing it
    #image = cv2.imread(image_paths)
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_detection_coordinates[i] = (i, confidence), (startX, startY), (endX, endY)

            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output image
    '''
    cv2.imshow("Output", image)
    cv2.resizeWindow('output', 600, 600)
    cv2.waitKey(1)    
    '''

    return face_detection_coordinates


def detect_video(video_path, video_name, frames_to_capture, destination):
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    count = 0

    vid = cv2.VideoCapture(video_path + video_name)
    video_name = video_name.replace('.mp4', '')
    os.makedirs(destination + video_name + '_frames' + '\\', exist_ok=True)
    while True:
        ret, cap = vid.read()   # Capture frame-by-frame
        if cap is not None:
            # number of faces detected in frame
            cr = image_face_detector(50, cap, './res10_300x300_ssd_iter_140000.caffemodel', './deploy.prototxt.txt')
            for i in range(len(cr)):
                frame = cap[int(cr[i][1][1]) - 25: int(cr[i][2][1]) + 25, int(cr[i][1][0]) - 25:int(cr[i][2][0]) + 25]
                dets = detector(frame, 0)  # cropped image
                # draw points on face for each rectangle
                for k, d in enumerate(dets):
                    # Get the landmarks/parts for the face in box d.
                    shape = predictor(frame, d)
                    # draw square around face
                    # cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (255, 0, 0), thickness=1)
                    # coords contains 64 points of the face
                    coords = []
                    # loop over the 68 facial landmarks and convert them to a 2-tuple of (x, y)-coordinates
                    #cv2.rectangle(frame, start_point, end_point, color, thickness)

                    for i in range(0, 68):
                        c = (int(shape.part(i).x), int(shape.part(i).y))
                        coords.append(c)
                        # draw dot of coordinate
                        cv2.circle(frame, coords[i], 1, (0, 0, 255), thickness=-1)

                    # writes cropped image of frame, with 25 pixel border
                    # [d.top() - 25:d.bottom() + 25, d.left() - 25:d.right() + 25]

                    cv2.imwrite(destination + video_name + '_frames' + '\\' + video_name + "_cropped_frame_%d.jpg" % count, frame)

                # sets nect frame to the 30th next frame
                count += frames_to_capture
                vid.set(1, count)
                # Display the resulting frame
                # cv2.imshow('frame', frame)
        else:
            vid.release()
            break


# this video has nothing being capture... to INVESTIGATE
# detect_video(video_path='D:\\Deep_Fake\\dfdc_train_all\\dfdc_train_part_00\\dfdc_train_part_0\\',
#              video_name='abhggqdift.mp4',
#              frames_to_capture=30,
#              destination='D:\\Deep_Fake\\dfdc_train_all_jpegs\\dfdc_train_part_0\\')

all_train_dir = 'D:\\Deep_Fake\\dfdc_train_all\\'
vid_sub_dir = [all_train_dir + x for x in os.listdir(all_train_dir)]
test_video_files = []

for i in range(len(vid_sub_dir)):
    test_video_dir = vid_sub_dir[i] + '\\' + str(os.listdir(vid_sub_dir[i])[0]) + '\\'  #D:\Deep_Fake\dfdc_train_all\dfdc_train_part_00\dfdc_train_part_0\
    # test_video_files = [test_video_dir + x for x in os.listdir(test_video_dir)]
    test_video_files = os.listdir(test_video_dir)
    os.makedirs('D:\\Deep_Fake\\dfdc_train_all_jpegs\\' + str(os.listdir(vid_sub_dir[i])[0]), exist_ok=True)
    destination_dir = 'D:\\Deep_Fake\\dfdc_train_all_jpegs\\' + str(os.listdir(vid_sub_dir[i])[0]) + '\\'
    for video in tqdm.tqdm(test_video_files):

        try:
            if video == 'metadata.json':
                shutil.copyfile(test_video_dir + video, destination_dir + 'metadata' + str(i)+'.json')
                # print('From: ' + test_video_dir + video + '\nToo: ' + destination_dir + 'metadata' + str(i) + '.json')

            detect_video(video_path=test_video_dir, video_name=video, frames_to_capture=150, destination=destination_dir)
            # if img_file is None:
            #     count+=1
            #     continue
            # cv2.imwrite('./DeepFake'+d_num+'/'+video.replace('.mp4','').replace(test_dir,'')+'.jpg',img_file)
        except Exception as err:
          print(err)


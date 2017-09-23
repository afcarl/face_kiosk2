import argparse
import cv2
import dlib
import numpy as np
import csv
import os
import random
from sklearn.neighbors import NearestNeighbors
from collections import deque

# dlib-based feature extraction
JAWLINE_POINTS = list(range(0, 17))
RIGHT_EYEBROW_POINTS = list(range(17, 22))
LEFT_EYEBROW_POINTS = list(range(22, 27))
NOSE_POINTS = list(range(27, 36))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
MOUTH_OUTLINE_POINTS = list(range(48, 61))
MOUTH_INNER_POINTS = list(range(61, 68))

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5], axis=-1)
    B = np.linalg.norm(eye[2] - eye[4], axis=-1)

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3], axis=-1)

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def is_eyes_closed(landmarks):
    landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

    leftEAR = eye_aspect_ratio(landmarks[LEFT_EYE_POINTS])
    rightEAR = eye_aspect_ratio(landmarks[RIGHT_EYE_POINTS])

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0
    return ear < 0.20, ear  # ratio of open/closed eyes

parser = argparse.ArgumentParser()
parser.add_argument("--npy_file", default='lfw.npy')
parser.add_argument("--csv_file", default='lfw.csv')
parser.add_argument("--faces_dir", default='/storage/LFW/lfw')
parser.add_argument("--n_neighbors", type=int, default=5)
parser.add_argument("--algorithm", choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto')
parser.add_argument("--video_camera", default=0)
parser.add_argument("--shape_predictor_path", default='shape_predictor_68_face_landmarks.dat')
parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')
parser.add_argument("--upscale", default=0)
parser.add_argument("--jitter", default=10)
args = parser.parse_args()

vecs = np.load(args.npy_file)
nn = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm=args.algorithm)
nn.fit(vecs)

with open(args.csv_file) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['file', 'name', 'left', 'top', 'right', 'bottom'])
    data = list(reader)
#print(vecs.shape, len(data), data[0])

vs = cv2.VideoCapture(args.video_camera)
assert vs.isOpened()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

def camera_loop():
    while True:
        ret, img = vs.read()
        assert ret
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, args.upscale)

        if rects:
            rect = rects[0]
            landmarks = predictor(gray, rect)

            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 255), 2)
            for p in landmarks.parts():
                cv2.circle(img, (p.x, p.y), 1, color=(0, 255, 255), thickness=-1)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == 27:
            esc = True
            break

camera_loop()

vs.release()
cv2.destroyAllWindows()

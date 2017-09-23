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

def eyes_aspect_ratio(landmarks):
    leftEAR = eye_aspect_ratio(landmarks[LEFT_EYE_POINTS])
    rightEAR = eye_aspect_ratio(landmarks[RIGHT_EYE_POINTS])

    # average the eye aspect ratio together for both eyes
    ear = (leftEAR + rightEAR) / 2.0
    return ear

parser = argparse.ArgumentParser()
parser.add_argument("--npy_file", default='lfw.npy')
parser.add_argument("--csv_file", default='lfw.csv')
parser.add_argument("--faces_dir", default='/storage/LFW/lfw')
parser.add_argument("--n_neighbors", type=int, default=5)
parser.add_argument("--algorithm", choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto')
parser.add_argument("--video_camera", default=0)
parser.add_argument("--shape_predictor_path", default='shape_predictor_68_face_landmarks.dat')
parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')
parser.add_argument("--upscale", type=int, default=0)
parser.add_argument("--jitter", type=int, default=10)
parser.add_argument("--eyes_closed_threshold", type=float, default=0.20)
parser.add_argument("--fullscreen", action="store_true")
parser.add_argument("--font_face", type=int, default=cv2.FONT_HERSHEY_SIMPLEX)
parser.add_argument("--font_scale", type=float, default=1)
parser.add_argument("--font_thickness", type=int, default=2)
args = parser.parse_args()

vecs = np.load(args.npy_file)
nn = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm=args.algorithm)
nn.fit(vecs)

with open(args.csv_file) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['file', 'name', 'left', 'top', 'right', 'bottom'])
    data = list(reader)
#print(vecs.shape, len(data), data[0])

if args.fullscreen:
    cv2.namedWindow("img", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

vs = cv2.VideoCapture(args.video_camera)
assert vs.isOpened()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

esc = False

while True:
    faces_queue = deque(maxlen=20)
    closed_queue = deque(maxlen=10)

    while True:
        ret, img = vs.read()
        assert ret
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, args.upscale)

        # Draw a rectangle aroudn the faces (dlib)
        #for rect in faces:
        if rects:
            rect = rects[0]
            landmarks = predictor(gray, rect)
            nplandmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

            cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 255), 2)
            for p in landmarks.parts():
                cv2.circle(img, (p.x, p.y), 1, color=(0, 255, 255), thickness=-1)

            ear = eyes_aspect_ratio(nplandmarks)
            closed = ear < args.eyes_closed_threshold
            cv2.putText(img, ("Closed!" if closed else "Open!") + str(ear), (5, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255, 0, 0) if closed else (0, 255, 0),
                thickness=1
            )

            faces_queue.append((img, rect, landmarks, nplandmarks))
            closed_queue.append(closed)
        else:
            faces_queue.clear()
            closed_queue.clear()

        cv2.putText(img, "Count: " + str(np.sum(closed_queue)), (5, 45),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1
        )

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == 27:
            esc = True
            break

        if len(closed_queue) == closed_queue.maxlen and np.all(closed_queue):
            break

    if esc:
        break

    img, rect, landmarks, nplandmarks = faces_queue[0]
    cv2.imshow('img', img)
    cv2.waitKey(1)

    descriptor = facerec.compute_face_descriptor(img, landmarks, args.jitter)
    dists, idxs = nn.kneighbors(np.array([descriptor]))
    if len(idxs) > 0:
        person = data[idxs[0][0]]
        img2 = cv2.imread(os.path.join(args.faces_dir, person['file']))
        rect2 = dlib.rectangle(int(person['left']), int(person['top']), int(person['right']), int(person['bottom']))
        '''
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        landmarks2 = predictor(gray2, rect2)
        nplandmarks2 = np.array([[p.x, p.y] for p in landmarks2.parts()])

        left_eye = np.mean(nplandmarks[LEFT_EYE_POINTS], axis=0)
        right_eye = np.mean(nplandmarks[RIGHT_EYE_POINTS], axis=0)
        nose = nplandmarks[30]

        left_eye2 = np.mean(nplandmarks2[LEFT_EYE_POINTS], axis=0)
        right_eye2 = np.mean(nplandmarks2[RIGHT_EYE_POINTS], axis=0)
        nose2 = nplandmarks2[30]
        
        points = np.float32([left_eye, right_eye, nose])
        points2 = np.float32([left_eye2, right_eye2, nose2])
        M = cv2.getAffineTransform(points2, points)        
        img3 = cv2.warpAffine(img2, M, (img.shape[1], img.shape[0]))
        '''
        
        img3 = cv2.resize(img2, img.shape[1::-1])

        esc = False
        while True:
            ret, img = vs.read()
            assert ret
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, args.upscale)

            # Draw a rectangle aroudn the faces (dlib)
            if rects:
                rect = rects[0]
                landmarks = predictor(gray, rect)
                nplandmarks = np.array([[p.x, p.y] for p in landmarks.parts()])

                ear = eyes_aspect_ratio(nplandmarks)
                if ear > args.eyes_closed_threshold:
                    break

            if cv2.waitKey(1) & 0xFF == 27:
                esc = True
                break

        if esc:
            break

        ((text_width, text_height), retval) = cv2.getTextSize(person['name'], 
                args.font_face, args.font_scale, args.font_thickness)
        cv2.putText(img3, person['name'], (img3.shape[1] // 2 - text_width // 2, img3.shape[0] - text_height - 5),
                fontFace=args.font_face, fontScale=args.font_scale, color=(255, 255, 255), thickness=args.font_thickness
        )

        cv2.imshow('img', img3)

        while True:
            ret, img3 = vs.read()
            assert ret
            gray3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
            rects3 = detector(gray3, args.upscale)

            if rects3:
                rect3 = rects3[0]
                landmarks3 = predictor(gray3, rect3)
                descriptor3 = facerec.compute_face_descriptor(img3, landmarks3, args.jitter)
                if np.linalg.norm(np.array(descriptor) - descriptor3) > 0.6:
                    break
            else:
                break

            if cv2.waitKey(1) & 0xFF == 27:
                break

vs.release()
cv2.destroyAllWindows()

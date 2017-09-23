from __future__ import print_function
import argparse
import os
import cv2
import dlib
import numpy as np
import csv

parser = argparse.ArgumentParser()
parser.add_argument("dir")
parser.add_argument("npy_file")
parser.add_argument("csv_file")
parser.add_argument("--upscale", type=int, default=1)
parser.add_argument("--jitter", type=int, default=100)
parser.add_argument("--predictor_path", default='shape_predictor_68_face_landmarks.dat')
parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')    
args = parser.parse_args()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

face_vectors = []
face_data = []

for dirname in sorted(os.listdir(args.dir)):
    absdirname = os.path.join(args.dir, dirname)
    if os.path.isdir(absdirname):
        for filename in sorted(os.listdir(absdirname)):
            absfilename = os.path.join(absdirname, filename)
            img = cv2.imread(absfilename)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img[...,::-1] # BGR to RGB
            faces = detector(gray, args.upscale)
            #assert len(faces) == 1
            print(filename, ':', len(faces))
            for rect in faces:
                landmarks = predictor(gray, rect)
                descriptor = facerec.compute_face_descriptor(img, landmarks, args.jitter)
                face_vectors.append(descriptor)
                face_data.append((os.path.join(dirname, filename), dirname.replace('_', ' '), rect.left(), rect.top(), rect.right(), rect.bottom()))

np.save(args.npy_file, np.array(face_vectors))
print(np.array(face_vectors).shape)
with open(args.csv_file, 'wb') as f:
    writer = csv.writer(f)
    writer.writerows(face_data)

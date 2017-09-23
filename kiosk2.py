from __future__ import print_function
import argparse
import numpy as np
import csv
from sklearn.neighbors import NearestNeighbors
import dlib
import cv2
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--capture_device", type=int, default=0)
parser.add_argument("--delay", type=int, default=1)
parser.add_argument("--canvas_width", type=int, default=1024)
parser.add_argument("--canvas_height", type=int, default=768)
parser.add_argument("--face_size", type=int, default=160)
parser.add_argument("--face_count", type=int, default=5)
parser.add_argument("--font_face", type=int, default=cv2.FONT_HERSHEY_SIMPLEX)
parser.add_argument("--font_scale", type=float, default=0.5)
parser.add_argument("--font_thickness", type=int, default=1)
parser.add_argument("--show_name", type=int, default=1)
parser.add_argument("--show_distance", type=int, default=1)

parser.add_argument("--images_path", default="/storage/LFW/RFLFW")
parser.add_argument("--data_file", default="RFLFW2.csv")
parser.add_argument("--vecs_file", default="RFLFW2.npy")
parser.add_argument("--shape_predictor_path", default='shape_predictor_68_face_landmarks.dat')
parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')
parser.add_argument("--upscale", type=int, default=0)
parser.add_argument("--jitter", type=int, default=10)
parser.add_argument("--n_neighbors", type=int, default=5)
parser.add_argument("--algorithm", choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto')
parser.add_argument("--average_window", type=int, default=10)
parser.add_argument("--group_by", choices=["file", "name"], default="name")

args = parser.parse_args()

print("Initializing face detector...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args.shape_predictor_path)
facerec = dlib.face_recognition_model_v1(args.face_rec_model_path)

print("Loading feature vectors...")
vecs = np.load(args.vecs_file)
nn = NearestNeighbors(n_neighbors=args.n_neighbors, algorithm=args.algorithm)
nn.fit(vecs)

print("Loading image metadata...")
with open(args.data_file) as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=['file', 'name', 'left', 'top', 'right', 'bottom'])
    data = list(reader)

print("Initializing video capture...")
video = cv2.VideoCapture(args.capture_device)
frame_width = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

# initialize canvas and make background white
canvas = np.empty((args.canvas_height, args.canvas_width, 3), dtype=np.uint8)
canvas.fill(255)

# calculate text height
((text_width, text_height), retval) = cv2.getTextSize("Abrakadabra", args.font_face, args.font_scale, args.font_thickness)
assert retval, "Unable to determine text height"

frame_left = int(args.canvas_width / 2 - frame_width / 2)
assert frame_left > 0, "Video frame too big, increase canvas width"
frame_top = int((args.canvas_height - frame_height - args.face_size - text_height) / 3)
assert frame_top > 0, "Video frame too big, increase canvas height"

face_gap = int((args.canvas_width - args.face_count * args.face_size) / (args.face_count + 1))
assert face_gap > 0, "Lower face size"
face_top = int(2 * frame_top + frame_height)

text_top = int(face_top + args.face_size + text_height + 5)

results = []
n = 0
while True:
  # capture frame
  ret, frame = video.read()
  assert ret, "Frame could not be read from video capture device"

  # detect faces
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  rects = detector(gray, args.upscale)

  # skip following if no faces were found
  if len(rects) == 0:
    canvas[frame_top:frame_top+frame_height, frame_left:frame_left+frame_width, :] = frame
  else:
    # draw rectangle around the face
    frame2 = frame.copy()
    rect = rects[0]
    cv2.rectangle(frame2, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
    canvas[frame_top:frame_top+frame_height, frame_left:frame_left+frame_width, :] = frame2

    # extract features of the face
    landmarks = predictor(gray, rect)
    descriptor = np.array(facerec.compute_face_descriptor(frame, landmarks, args.jitter))

    # find nearest images
    dists, idxs = nn.kneighbors(descriptor[np.newaxis])
    results += [{'file': data[k]['file'], 'distance': dists[0, i], 'name': data[k]['name']} for i, k in enumerate(idxs[0])]

    # after every average_window steps average predictions
    n = (n + 1) % args.average_window
    if n == 0:
      # create dictionary with list of results in each group
      groups = dict()
      for res in results:
        groups.setdefault(res[args.group_by], []).append(res)

      # calculate average distance for each group
      distances = dict()
      for k, group in groups.iteritems():
          # calculate average distance
          distances[k] = float(sum(res['distance'] for res in group) / len(group))

      # order groups by average distance
      ordered_groups = sorted(distances, key=distances.get)

      # keep only face_count top groups (by distance)
      top_groups = ordered_groups[:args.face_count]

      # erase previous faces and texts
      canvas[face_top:].fill(255)

      # loop over top face groups
      for i, k in enumerate(top_groups):
        group = groups[k]
        res = group[0]

        # read the face image
        filename = os.path.join(args.images_path, res['file'])
        face = cv2.imread(filename)
        
        # draw the face
        face = cv2.resize(face, (args.face_size, args.face_size))
        face_left = face_gap + i * (face_gap + args.face_size)
        canvas[face_top:face_top+args.face_size,face_left:face_left+args.face_size, :] = face

        if args.show_name:
          # draw the text
          text = res['name']
          ((text_width, text_height), retval) = cv2.getTextSize(text, args.font_face, args.font_scale, args.font_thickness)
          assert retval, "Unable to determine text height"
          text_left = int(face_left + args.face_size / 2 - text_width / 2)
          cv2.putText(canvas, text, (text_left, text_top), args.font_face, args.font_scale, (10, 10, 10), args.font_thickness)

        if args.show_distance:
          text = str(distances[k]) + " " + str(len(group))
          ((text_width, text_height), retval) = cv2.getTextSize(text, args.font_face, args.font_scale, args.font_thickness)
          assert retval, "Unable to determine text height"
          text_left = int(face_left + args.face_size / 2 - text_width / 2)
          cv2.putText(canvas, text, (text_left, text_top + text_height + 10), args.font_face, args.font_scale, (10, 10, 10), args.font_thickness)

      results = []

  # display the canvas
  cv2.imshow('Video', canvas)

  if cv2.waitKey(1) & 0xFF == 27:
    break;

# when everything is done, release the capture
video.release()
cv2.destroyAllWindows()

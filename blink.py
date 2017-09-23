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
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

# Check if a point is inside a rectangle
def rect_contains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True

# Draw delaunay triangles
def draw_delaunay(img, subdiv, delaunay_color ) :
 
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
 
    for t in triangleList :
         
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
         
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
         
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)

# Draw voronoi diagram
def draw_voronoi(img, subdiv) :
 
    ( facets, centers) = subdiv.getVoronoiFacetList([])
 
    for i in range(len(facets)) :
        ifacet_arr = []
        for f in facets[i] :
            ifacet_arr.append(f)
         
        ifacet = np.array(ifacet_arr, np.int)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
 
        cv2.fillConvexPoly(img, ifacet, color, cv2.LINE_AA, 0);
        ifacets = np.array([ifacet])
        cv2.polylines(img, ifacets, True, (0, 0, 0), 1, cv2.LINE_AA, 0)
        cv2.circle(img, (centers[i][0], centers[i][1]), 3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
 
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

faces = deque(maxlen=10)

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

        subdiv = cv2.Subdiv2D((rect.left()-50, rect.top()-50, rect.right()+50, rect.bottom()+50))

        img_delaunay = img.copy()
        for p in landmarks.parts():
            subdiv.insert((p.x, p.y))
        draw_delaunay(img_delaunay, subdiv, (255, 255, 255))

        img_voronoi = np.zeros(img.shape, dtype = img.dtype)
        draw_voronoi(img_voronoi,subdiv)

        faces.append((img, rect, landmarks))

        cv2.rectangle(img, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (255, 0, 255), 2)
        for p in landmarks.parts():
            cv2.circle(img, (p.x, p.y), 1, color=(0, 255, 255), thickness=-1)

        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        leftEAR = eye_aspect_ratio(landmarks[LEFT_EYE_POINTS])
        rightEAR = eye_aspect_ratio(landmarks[RIGHT_EYE_POINTS])

        # average the eye aspect ratio together for both eyes
        ear = (leftEAR + rightEAR) / 2.0
        if ear > 0.18:  # ratio of open/closed eyes
            #print("Eyes are open, EAR: " + str(ear))
            cv2.putText(img, "Open! EAR: " + str(ear), (5, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 255, 0),
                thickness=1
            )
        else:
            cv2.putText(img, "Closed!" + str(ear), (5, 25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(255, 0, 0),
                thickness=1
            )
            break
    else:
        faces.clear()

    cv2.imshow('img', img)
    cv2.imshow('img_delaunay', img_delaunay)
    cv2.imshow('img_voronoi', img_voronoi)
    if cv2.waitKey(1) & 0xFF == 27:
        break

img, rect, landmarks = faces[0]
descriptor = facerec.compute_face_descriptor(img, landmarks, args.jitter)
descriptor = np.array(descriptor)
dists, idxs = nn.kneighbors(descriptor[np.newaxis])
if len(idxs) > 0:
    person = data[idxs[0][0]]
    img2 = cv2.imread(os.path.join(args.faces_dir, person['file']))
    rect2 = dlib.rectangle(int(person['left']), int(person['top']), int(person['right']), int(person['bottom']))
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    landmarks2 = predictor(gray2, rect2)

    subdiv = cv2.Subdiv2D((rect2.left()-50, rect2.top()-50, rect2.right()+50, rect2.bottom()+50))

    img_delaunay2 = img2.copy()
    for p in landmarks2.parts():
        subdiv.insert((p.x, p.y))
    draw_delaunay(img_delaunay2, subdiv, (255, 255, 255))

    img_voronoi2 = np.zeros(img2.shape, dtype = img.dtype)
    draw_voronoi(img_voronoi2,subdiv)

    cv2.imshow('img2', img2)
    cv2.imshow('img_delaunay2', img_delaunay2)
    cv2.imshow('img_voronoi2', img_voronoi2)
    cv2.waitKey(0)

vs.release()
cv2.destroyAllWindows()

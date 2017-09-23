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

TRIANGLES = [[38, 40, 37],
       [35, 30, 29],
       [38, 37, 20],
       [18, 37, 36],
       [33, 32, 30],
       [54, 64, 53],
       [30, 32, 31],
       [59, 48, 60],
       [40, 31, 41],
       [36, 37, 41],
       [21, 39, 38],
       [35, 34, 30],
       [51, 33, 52],
       [40, 29, 31],
       [57, 58, 66],
       [36, 17, 18],
       [35, 52, 34],
       [65, 66, 62],
       [58, 67, 66],
       [53, 63, 52],
       [61, 67, 49],
       [53, 65, 63],
       [56, 66, 65],
       [55, 10,  9],
       [64, 54, 55],
       [43, 42, 22],
       [46, 54, 35],
       [ 1,  0, 36],
       [19, 37, 18],
       [ 1, 36, 41],
       [ 0, 17, 36],
       [37, 19, 20],
       [21, 38, 20],
       [39, 40, 38],
       [28, 29, 39],
       [41, 31,  2],
       [59, 67, 58],
       [29, 30, 31],
       [34, 33, 30],
       [21, 27, 39],
       [28, 42, 29],
       [52, 33, 34],
       [62, 66, 67],
       [48,  4,  3],
       [41,  2,  1],
       [31,  3,  2],
       [37, 40, 41],
       [39, 29, 40],
       [57,  7, 58],
       [31, 48,  3],
       [ 5,  4, 48],
       [32, 49, 31],
       [60, 49, 59],
       [59,  5, 48],
       [ 7,  6, 58],
       [31, 49, 48],
       [49, 67, 59],
       [ 6, 59, 58],
       [ 6,  5, 59],
       [ 8,  7, 57],
       [48, 49, 60],
       [32, 33, 50],
       [49, 50, 61],
       [49, 32, 50],
       [51, 61, 50],
       [62, 67, 61],
       [33, 51, 50],
       [63, 65, 62],
       [51, 62, 61],
       [51, 52, 63],
       [51, 63, 62],
       [52, 35, 53],
       [47, 46, 35],
       [54, 10, 55],
       [56, 57, 66],
       [56,  8, 57],
       [53, 55, 65],
       [ 9,  8, 56],
       [56, 55,  9],
       [65, 55, 56],
       [10, 54, 11],
       [53, 64, 55],
       [53, 35, 54],
       [12, 54, 13],
       [12, 11, 54],
       [54, 14, 13],
       [35, 42, 47],
       [45, 16, 15],
       [22, 42, 27],
       [42, 35, 29],
       [27, 42, 28],
       [44, 25, 45],
       [44, 47, 43],
       [46, 14, 54],
       [45, 46, 44],
       [45, 14, 46],
       [39, 27, 28],
       [22, 23, 43],
       [21, 22, 27],
       [21, 20, 23],
       [43, 23, 24],
       [22, 21, 23],
       [44, 43, 24],
       [47, 42, 43],
       [25, 44, 24],
       [46, 47, 44],
       [16, 45, 26],
       [15, 14, 45],
       [45, 25, 26]]

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

# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst

# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask

def morphFaces(img1, img2, points1, points2, alpha):
    assert len(points1) == len(points2)
    
    points = []
    for p1, p2 in zip(points1, points2):
        p = ( 1 - alpha ) * p1 + alpha * p2
        points.append(p)

    # Compute weighted average point coordinates
    img = np.zeros(img1.shape, dtype=img1.dtype)
    for x, y, z in TRIANGLES:
        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [ points[x], points[y], points[z] ]
        morphTriangle(img1, img2, img, t1, t2, t, alpha)

    return img

parser = argparse.ArgumentParser()
parser.add_argument("--npy_file", default='lfw-deepfunneled.npy')
parser.add_argument("--csv_file", default='lfw-deepfunneled.csv')
parser.add_argument("--faces_dir", default='/storage/LFW/lfw-deepfunneled')
parser.add_argument("--n_neighbors", type=int, default=5)
parser.add_argument("--algorithm", choices=['auto', 'ball_tree', 'kd_tree', 'brute'], default='auto')
parser.add_argument("--video_camera", default=0)
parser.add_argument("--shape_predictor_path", default='shape_predictor_68_face_landmarks.dat')
parser.add_argument("--face_rec_model_path", default='dlib_face_recognition_resnet_model_v1.dat')
parser.add_argument("--upscale", type=int, default=0)
parser.add_argument("--jitter", type=int, default=10)
parser.add_argument("--eyes_closed_threshold", type=float, default=0.19)
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

            faces_queue.append((img.copy(), rect, landmarks, nplandmarks))

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
    #cv2.imshow('img', img)
    #cv2.waitKey(1)

    descriptor = facerec.compute_face_descriptor(img, landmarks, args.jitter)
    dists, idxs = nn.kneighbors(np.array([descriptor]))
    if len(idxs) > 0:
        person = data[idxs[0][0]]
        img2 = cv2.imread(os.path.join(args.faces_dir, person['file']))
        rect2 = dlib.rectangle(int(person['left']), int(person['top']), int(person['right']), int(person['bottom']))
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        landmarks2 = predictor(gray2, rect2)
        nplandmarks2 = np.array([[p.x, p.y] for p in landmarks2.parts()])

        left_eye = np.mean(nplandmarks[LEFT_EYE_POINTS], axis=0)
        right_eye = np.mean(nplandmarks[RIGHT_EYE_POINTS], axis=0)
        mouth = np.mean(nplandmarks[MOUTH_OUTLINE_POINTS], axis=0)

        left_eye2 = np.mean(nplandmarks2[LEFT_EYE_POINTS], axis=0)
        right_eye2 = np.mean(nplandmarks2[RIGHT_EYE_POINTS], axis=0)
        mouth2 = np.mean(nplandmarks2[MOUTH_OUTLINE_POINTS], axis=0)
        
        points = np.float32([left_eye, right_eye, mouth])
        points2 = np.float32([left_eye2, right_eye2, mouth2])
        M = cv2.getAffineTransform(points2, points)        
        img3 = cv2.warpAffine(img2, M, (img.shape[1], img.shape[0]), None, flags=cv2.INTER_LINEAR) #, borderMode=cv2.BORDER_REFLECT_101)
        #print(nplandmarks2.shape, M.shape)
        landmarks3 = cv2.transform(nplandmarks2[:,np.newaxis], M)
        landmarks3 = landmarks3[:, 0]
        #for p in landmarks3:
        #    cv2.circle(img3, tuple(p), 1, color=(0, 255, 255), thickness=-1)
        #cv2.imshow('img', img3)
        #cv2.waitKey(0)
        
        esc = False
        while True:
            ret, img = vs.read()
            assert ret
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, args.upscale)

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

        for alpha in np.arange(0, 1.1, 0.1):
            img4 = morphFaces(img, img3, nplandmarks, landmarks3, alpha)
            cv2.imshow('img', img4)
            cv2.waitKey(100)

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

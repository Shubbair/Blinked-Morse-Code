from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
from utils import *
 
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 
# define two constants, one for the eye aspect ratio to indicate
# blink and then a second constant for the number of consecutive
# frames the eye must be below the threshold
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 3
EYE_AR_CONSEC_FRAMES2 = 6
EYE_AR_CONSEC_FRAMES3 = 10

# initialize the frame counters and the total number of blinks
COUNTER = 0
#TOTAL = 0
TOTAL=[]
#Morse Message
string=""
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# grab the indexes of the facial landmarks for the left and
# right eye, respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream thread
print("[INFO] starting video stream thread...")
fileStream = True

if args["video"] != "":
    video_stream = cv2.VideoCapture(args["video"])
else:
    video_stream = cv2.VideoCapture(0)
    
if (video_stream.isOpened()== False): 
  print("Error opening video  file")

time.sleep(1.0)

# loop over frames from the video stream
while (video_stream.isOpened()):
	ret,frame = video_stream.read()
	frame = imutils.resize(frame, width=750)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# detect faces in the grayscale frame
	rects = detector(gray, 0)

	# loop over the face detections
	for rect in rects:
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		# extract the left and right eye coordinates, then use the
		# coordinates to compute the eye aspect ratio for both eyes
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# average the eye aspect ratio together for both eyes
		ear = (leftEAR + rightEAR) / 2.0

		# compute the convex hull for the left and right eye, then
		# visualize each of the eyes
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# check to see if the eye aspect ratio is below the blink
		# threshold, and if so, increment the blink frame counter
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		# otherwise, the eye aspect ratio is not below the blink
		# threshold
		else:
			# if the eyes were closed for a sufficient number of
			# then increment the total number of blinks
			if COUNTER >= EYE_AR_CONSEC_FRAMES and COUNTER<=EYE_AR_CONSEC_FRAMES2:
				TOTAL.append(".")
			elif COUNTER >=EYE_AR_CONSEC_FRAMES2 and COUNTER<=EYE_AR_CONSEC_FRAMES3:
				TOTAL.append("-")
			elif COUNTER>=EYE_AR_CONSEC_FRAMES3:
				s=str(Decode_Morse(''.join(TOTAL)))
				if s=="None":
					TOTAL=[]
				else:
					string+=s
					TOTAL=[]

			COUNTER = 0

		# draw the total number of blinks on the frame along with
		# the computed eye aspect ratio for the frame
		cv2.putText(frame, "Morse_Code: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (600, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
		cv2.putText(frame,"Messsage: {}".format(string), (20,100), 
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
 
	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
video_stream.stop()
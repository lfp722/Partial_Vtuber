import cv2
import mediapipe as mp


def find_coordinate(a, w, h):  # a = coordinate, w =  width, h = height
    ret = int((a.x*w)), int((a.y*h))
    return ret


def overlay(i, x, y, w, h, o_image):
    alpha = o_image[:, :, 3]
    mask = alpha/255
    for c in range(0, 3):
        i[y-h:y+h, x-w:x+w, c] = (o_image[:, :, 3] + mask) + (i[y-h:y+h, x-w:x+w,  c] * (1-mask))


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# open video file
cap = cv2.VideoCapture('video.mov') # put name of your video file


i_le = cv2.imread('left_eyes.png') # put name of your photo for left eyes
i_re = cv2.imread('right_eyes.png') # put name of your photo for right eyes
i_nose = cv2.imread('nose.png') # put name of your photo for nose

with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.8) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # If loading a video, use 'break' instead of 'continue'.
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image)

        # Draw the face det ection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
                # detection = 6 points of face
                rks = detection.location_data.relative_keypoints
                re = rks[0]  # right eyes
                le = rks[1]  # left eyes
                nose = rks[2]
                '''h = height, w = width, c = channel'''
                h, w, c = image.shape
                re = find_coordinate(re, w, h)
                le = find_coordinate(le, w, h)
                nose = find_coordinate(nose, w, h)

                '''
                overlay(image, *re, 50, 50, i_re)
                overlay(image, *le, 50, 50, i_le)
                overlay(image, *nose, 150, 50, i_nose)
                '''

                image[re[1]-50:re[1]+50, re[0]-50:re[0]+50] = i_re
                image[le[1] - 50:le[1] + 50, le[0] - 50:le[0] + 50] = i_le
                image[nose[1] - 50:nose[1] + 50, nose[0] - 150:nose[0] + 150] = i_nose

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow('MediaPipe Face Detection', cv2.resize(image, None, fx=0.5, fy=0.5))
        if cv2.waitKey(1) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

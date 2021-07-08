import dlib # https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/
# import imutils
import cv2 as cv
from imutils import face_utils
import math

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha_inv * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha * img[y1:y2, x1:x2, c])

        
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

glass_counter = 1
s_img = cv.imread(f'glass{glass_counter}.jpg')

cap = cv.VideoCapture(3)
while(cap.isOpened()):
    ret, frame = cap.read()
    
    frame = cv.flip(frame,1)

    # load the image and perform some operations on it

    face_Gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    boundary = detector(face_Gray, 1)
#     x_cor, y_cor = 0,0
#     x2, x1 = 0,0
#     y1,y2 = 0,0
#     slope = 0
#     z = 0
#     angle = 0
    for (index, rectangle) in enumerate(boundary):
        shape = predictor(face_Gray, rectangle)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rectangle)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        cv.putText(frame, "Face {}".format(index + 1), (x - 10, y - 10), cv.FONT_ITALIC, 0.7, (0, 0, 255), 2)
        
        for (x,y) in shape:
            cv.circle(frame, (x,y), 1, (0,0,255),-1)

#         slope = (shape[26][1]-shape[17][1])/(shape[26][0]-x1)
#         angle = math.degrees(math.atan(slope))

        x1 = shape[17][0]
        x2 = shape[26][0]
        y1 = shape[19][1]
        y2 = shape[28][1]
        
#         s_img = imutils.rotate(s_img, -angle)
        
        s_img = cv.resize(s_img, ((x2 - x1), (y2 - y1)))

#         s_img = cv.resize(s_img, (0,0),fx=0.1,fy=0.1)
#         overlay_image_alpha(frame, s_img[:, :, 0:3], (shape[17][0], shape[17][1]), s_img[:, :, 2] / 255.0)
        overlay_image_alpha(frame, s_img, (shape[17][0], shape[17][1]), s_img[:, :, 2] / 255.0)

    cv.imshow("detected face", frame)
    pressedKey = cv.waitKey(1) & 0xFF
    if pressedKey == ord('s'):
            glass_counter += 1
            s_img = cv.imread(f'glass{glass_counter}.jpg')
            if s_img is None:
                glass_counter = 1
                s_img = cv.imread(f'glass{glass_counter}.jpg')
    elif pressedKey == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
#27-18 x
# 29-20 y
# rotate = imutils.rotate(img, angle)
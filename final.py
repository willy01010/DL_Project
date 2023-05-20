import cv2
import mediapipe as mp
import numpy as np
import math
import time


track = []
combine = []
status = False
change = False
c=0
last_c_time = 0
#button_pressed = False
#button_name = "Capture"

sumi_img = cv2.imread("resize.png")

#def button_click(event, x, y, flags, param):
 #   global button_pressed
  #  if event == cv2.EVENT_LBUTTONDOWN:
   #     if 10 < x < 200 and 60 < y < 120:
    #        button_pressed = True

if __name__ == '__main__':

    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
            min_detection_confidence=0.9,
            min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()

            image_size = image.shape
            image_height = image_size[0]
            image_width = image_size[1]
            print('image_height:',image_height,',image_width:',image_width)
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            # 把影像丟入mp作處理
            results = hands.process(image)

            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # print(len(results))

            if results.multi_hand_landmarks:
                #print('**********************************************************')
                #print(len(results.multi_hand_landmarks))
                #print('**********************************************************')

                for hand_landmarks in results.multi_hand_landmarks:
                    MIDDLE_FINGER_TIP_X = int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width)
                    MIDDLE_FINGER_TIP_Y = int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height)

                    INDEX_FINGER_TIP_X = int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                    INDEX_FINGER_TIP_Y = int(
                        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)

                    INDEX_FINGER_TIP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x
                    INDEX_FINGER_TIP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                    # print(INDEX_FINGER_TIP_X_value,INDEX_FINGER_TIP_Y_value)

                    INDEX_FINGER_MCP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
                    INDEX_FINGER_MCP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
                    # print(INDEX_FINGER_MCP_X_value,INDEX_FINGER_MCP_Y_value)

                    p1 = np.array([INDEX_FINGER_TIP_X_value, INDEX_FINGER_TIP_Y_value])
                    p2 = np.array([INDEX_FINGER_MCP_X_value, INDEX_FINGER_MCP_Y_value])
                    p3 = p2 - p1
                    distance_INDEX = math.hypot(p3[0], p3[1])
                    # print(distance_text)

                    MIDDLE_FINGER_TIP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x
                    MIDDLE_FINGER_TIP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
                    # print(INDEX_FINGER_TIP_X_value,INDEX_FINGER_TIP_Y_value)

                    MIDDLE_FINGER_MCP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
                    MIDDLE_FINGER_MCP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
                    # print(INDEX_FINGER_MCP_X_value,INDEX_FINGER_MCP_Y_value)

                    p4 = np.array([MIDDLE_FINGER_TIP_X_value, MIDDLE_FINGER_TIP_Y_value])
                    p5 = np.array([MIDDLE_FINGER_MCP_X_value, MIDDLE_FINGER_MCP_Y_value])
                    p6 = p5 - p4
                    distance_MIDDLE = math.hypot(p6[0], p6[1])

                    RING_FINGER_TIP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x
                    RING_FINGER_TIP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
                    # print(INDEX_FINGER_TIP_X_value,INDEX_FINGER_TIP_Y_value)

                    RING_FINGER_MCP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x
                    RING_FINGER_MCP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y
                    # print(INDEX_FINGER_MCP_X_value,INDEX_FINGER_MCP_Y_value)

                    p7 = np.array([RING_FINGER_TIP_X_value, RING_FINGER_TIP_Y_value])
                    p8 = np.array([RING_FINGER_MCP_X_value, RING_FINGER_MCP_Y_value])
                    p9 = p8 - p7
                    distance_RING = math.hypot(p9[0], p9[1])

                    PINKY_TIP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x
                    PINKY_TIP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
                    # print(INDEX_FINGER_TIP_X_value,INDEX_FINGER_TIP_Y_value)

                    PINKY_MCP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x
                    PINKY_MCP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y
                    # print(INDEX_FINGER_MCP_X_value,INDEX_FINGER_MCP_Y_value)

                    p10 = np.array([PINKY_TIP_X_value, PINKY_TIP_Y_value])
                    p11 = np.array([PINKY_MCP_X_value, PINKY_MCP_Y_value])
                    p12 = p11 - p10
                    distance_PINKY = math.hypot(p12[0], p12[1])

                    THUMB_TIP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
                    THUMB_TIP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y
                    # print(INDEX_FINGER_TIP_X_value,INDEX_FINGER_TIP_Y_value)

                    THUMB_MCP_X_value = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x
                    THUMB_MCP_Y_value = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y
                    # print(INDEX_FINGER_MCP_X_value,INDEX_FINGER_MCP_Y_value)

                    p13 = np.array([THUMB_TIP_X_value, THUMB_TIP_Y_value])
                    p14 = np.array([THUMB_MCP_X_value, THUMB_MCP_Y_value])
                    p15 = p14 - p13
                    distance_THUMB = math.hypot(p15[0], p15[1])
                    # -------------------------------------

                    result = ''

                    if ((distance_THUMB < 0.1) and (distance_INDEX < 0.1) and
                            (distance_MIDDLE < 0.1) and (distance_RING < 0.1) and (distance_PINKY < 0.1)):
                        result = 'clear'
                        combine = []
                        track = []

                    if ((distance_THUMB < 0.1) and (distance_INDEX > 0.1) and
                            (distance_MIDDLE > 0.1) and (distance_RING < 0.1) and (distance_PINKY < 0.1)):
                        result = 'pause'
                        status = False

                    if ((distance_THUMB < 0.1) and (distance_INDEX > 0.1) and
                            (distance_MIDDLE < 0.1) and (distance_RING < 0.1) and (distance_PINKY < 0.1)):
                        result = 'drawing'
                        status = True

                    if cv2.waitKey(2) & 0xFF == ord('c'):
                        change = True
                        # c = c+1
                        result = 'change'

# ---------------------------------------
                    if status:
                        if change:
                            if time.time() - last_c_time > 1:
                                c = c + 1
                                combine.append(track)
                                track = []
                                last_c_time = time.time()
                                change = False
                        track.append([INDEX_FINGER_TIP_X_value, INDEX_FINGER_TIP_Y_value])

                    cv2.circle(image, (int(INDEX_FINGER_TIP_X_value * image_width), int(INDEX_FINGER_TIP_Y_value * image_height)), 10, (0, 0, 255), -1)

                    print("c:", c)
    
                    # if c < 1:
                    print("len(track)", len(track))
                    for i in range(1, len(track)):
                        cv2.line(image, (int(track[i-1][0] * image_width), int(track[i-1][1] * image_height)),
                                (int(track[i][0] * image_width), int(track[i][1] * image_height)), (255, 100, 0), 5)
                    # else:
                    for i in range(len(combine)):
                        print("len(track)", len(combine[i]))
                        for j in range(1, len(combine[i])):
                            print("combine[i][j-1][0]:", combine[i][j-1][0], "combine[i][j-1][1]:", combine[i][j-1][1])
                            print("combine[i][j][0]:", combine[i][j][0], "combine[i][j][1]:", combine[i][j][1])
                            cv2.line(image, (int(combine[i][j-1][0] * image_width), int(combine[i][j-1][1] * image_height)),
                                    (int(combine[i][j][0] * image_width), int(combine[i][j][1] * image_height)), (255, 100, 0), 5)

# ---------------------------------------

                    # if status:
                    #     if change:
                    #         if time.time() - last_c_time >1:
                    #             c = c+1
                    #             combine.append([track])
                    #             track = []
                    #             last_c_time = time.time()
                    #             change = False                           
                    #     track.append([INDEX_FINGER_TIP_X_value, INDEX_FINGER_TIP_Y_value])

                    # if status:
                    #     if change:
                    #         track.append([])
                    #         c = c+1
                    #     track[c].append([INDEX_FINGER_TIP_X_value, INDEX_FINGER_TIP_Y_value])

                    # cv2.circle(image, (int(INDEX_FINGER_TIP_X_value * image_width),
                    #                    int(INDEX_FINGER_TIP_Y_value * image_height)), 10, (0, 0, 255), -1)

                    
                    # print("c:",c)
                    # print("combine:",combine)
                    # if c <1:
                    #     print("len(track)",len(track))
                    #     for i in range(0, len(track)):
                    #         if i > 0:
                    #             cv2.line(image, (int(track[i-1][0] * image_width), int(track[i-1][1] * image_height)),
                    #                     (int(track[i][0] * image_width), int(track[i][1] * image_height)), (255, 100, 0), 5)
                    # else:
                    #     for i in range(0,len(combine)):
                    #         print("len(track)",len(track),"len(track[i]):",len(track[i]))
                    #         for j in range(0,len(combine[i])):
                    #             if j > 0:
                    #                 print("combine[i][j-1][0]:",combine[i][j-1][0],"combine[i][j-1][1]:",combine[i][j-1][1])
                    #                 print("combine[i][j][0]:",combine[i][j][0],"combine[i][j][1]:",combine[i][j][1])
                    #                 cv2.line(image, (int(combine[i][j-1][0] * image_width), int(combine[i][j-1][1] * image_height)),
                    #                         (int(combine[i][j][0] * image_width), int(combine[i][j][1] * image_height)), (255, 100, 0), 5)

                    # for i in range(0, len(track)):
                    #     print("len(track)",len(track),"len(track[i]):",len(track[i]))
                        # for j in range(0,len(track[i])):
                        #     if j > 0:
                    #             cv2.line(image, (int(track[i][j-1][0] * image_width), int(track[i][j-1][1] * image_height)),
                    #                     (int(track[i][j][0] * image_width), int(track[i][j][1] * image_height)), (255, 100, 0), 5)


                    cv2.putText(image, 'Status:' + str(result), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255),
                                5, cv2.LINE_AA)
                    # -------------------------------------

                    
            
            #cv2.putText(image, button_name, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), 5, cv2.LINE_AA)
            #cv2.rectangle(image, (10, 60), (200, 120), (0, 255, 0), -1)

                  
            # 確保圖像大小相同
            sumi_img = cv2.resize(sumi_img, (image.shape[1], image.shape[0]))

            #sumi_img = cv2.resize(sumi_img, (128, 128))
            
            # 設定疊加的權重 (alpha 值)
            alpha = 0.5

            # 疊加圖像
            blended_img = cv2.addWeighted(image, 1-alpha, sumi_img, alpha, 0)

            # 按下a即可擷取影像
            if cv2.waitKey(5) & 0xFF == ord('a'):
                cv2.imwrite('origen sktech at ' + str(int(time.time())) + '.jpg', image)

            if cv2.waitKey(5) & 0xFF == ord('b'):
                cv2.imwrite('image sktech at ' + str(int(time.time())) + '.jpg', blended_img)

            cv2.imshow('MediaPipe Hands', blended_img)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import math
import pyvjoy

# Calculate the angle between two points
def calculate_angle_2d(p1, p2):
    angle = np.arctan2(p2.y - p1.y, p2.x - p1.x)
    return angle * (180 / np.pi)

def main(input_file_name = None, output_file_name = None, mirror = False):
    if input_file_name is None:
        # open the camera
        cap = cv2.VideoCapture(0)
        #cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(input_file_name)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Create a VideoWriter object to save the processed video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_file_name, fourcc, fps, (width, height))

    # Initialize the holistic model of the mediapipe library
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Initialize the drawing utility
    mp_drawing = mp.solutions.drawing_utils
    j = pyvjoy.VJoyDevice(1)
    j.set_axis(pyvjoy.HID_USAGE_X, 0x4000)
    j.set_axis(pyvjoy.HID_USAGE_Y, 0x4000)
    j.set_axis(pyvjoy.HID_USAGE_SL0, 0)
    j.set_axis(pyvjoy.HID_USAGE_SL1, 0)

    # Define the colors for the left and right hands
    left_color = (255, 0, 0)  # blue
    right_color = (0, 0, 255)  # red
    fl = 3
    first_frame_flag = True
    init_left_knee_y = 0
    init_right_knee_y = 0
    frame_num = 0

    while cap.isOpened():
        # Read the camera frame
        ret, frame = cap.read()
        if mirror == True:
        # Mirror the camera image
            frame = cv2.flip(frame, 1)
        # Convert the color space
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the holistic model
        results = holistic.process(image_rgb)

        # Draw the skeleton
        annotated_image = frame.copy()

        if results.pose_landmarks:
            # Draw the other parts except the arms
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                   circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                     thickness=2))
            if mirror == True:
                # Since the image is mirrored, the left and right should be flipped when processing, for example, mp_holistic.PoseLandmark.LEFT_SHOULDER actually refers to the right side
                # Draw the left and right arms separately
                left_connections = [mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW,
                                    mp_holistic.PoseLandmark.RIGHT_WRIST]
                right_connections = [mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW,
                                    mp_holistic.PoseLandmark.LEFT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
                right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]
                right_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]
            else:
                left_connections = [mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW,
                                    mp_holistic.PoseLandmark.LEFT_WRIST]
                right_connections = [mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW,
                                    mp_holistic.PoseLandmark.RIGHT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
                right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ELBOW]
                right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]
                right_knee = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]

            for i in range(2):
                cv2.line(annotated_image,
                         (int(results.pose_landmarks.landmark[right_connections[i]].x * frame.shape[1]),
                          int(results.pose_landmarks.landmark[right_connections[i]].y * frame.shape[0])),
                         (int(results.pose_landmarks.landmark[right_connections[i + 1]].x * frame.shape[1]),
                          int(results.pose_landmarks.landmark[right_connections[i + 1]].y * frame.shape[0])),
                         right_color, 2)
                cv2.line(annotated_image,
                         (int(results.pose_landmarks.landmark[left_connections[i]].x * frame.shape[1]),
                          int(results.pose_landmarks.landmark[left_connections[i]].y * frame.shape[0])),
                         (int(results.pose_landmarks.landmark[left_connections[i + 1]].x * frame.shape[1]),
                          int(results.pose_landmarks.landmark[left_connections[i + 1]].y * frame.shape[0])),
                         left_color, 2)

            # Get the key points of the hands and draw the line segments
     
            cv2.line(annotated_image, (int(left_wrist.x * frame.shape[1]), int(left_wrist.y * frame.shape[0])),
                     (int(right_wrist.x * frame.shape[1]), int(right_wrist.y * frame.shape[0])), (0, 255, 0), 2)

            # section1: Calculate the angle of steering wheel rotation
            angle = calculate_angle_2d(left_wrist, right_wrist)
            cv2.putText(annotated_image, str(int(angle)),
                        (int((left_wrist.x + right_wrist.x) / 2 * frame.shape[1]), int((left_wrist.y + right_wrist.y) / 2 * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            hex_value = int((angle + 100) / 200 * 32767)

            j.set_axis(pyvjoy.HID_USAGE_X, hex_value)


            # section2: Check if it's in the shifting gear state
            # Two conditions: the length of the line connecting the two hands > 1.3 times the length of the line connecting the two shoulders and the angle between the right upper arm and the right forearm > 85°
            # Determine the distance
            shoulder_distance = math.dist((right_shoulder.x, right_shoulder.y), (left_shoulder.x, left_shoulder.y))
            wrist_distance = math.dist((right_wrist.x, right_wrist.y), (left_wrist.x, left_wrist.y))

            # Check the angle
            elbow_wrist_distance = math.dist((right_elbow.x, right_elbow.y), (right_wrist.x, right_wrist.y))
            shoulder_elbow_distance = math.dist((right_shoulder.x, right_shoulder.y), (right_elbow.x, right_elbow.y))
            shoulder_wrist_distance = math.dist((right_shoulder.x, right_shoulder.y), (right_wrist.x, right_wrist.y))
            angle = math.degrees(
                math.acos((elbow_wrist_distance ** 2 + shoulder_elbow_distance ** 2 - shoulder_wrist_distance ** 2) /
                          (2 * elbow_wrist_distance * shoulder_elbow_distance)))

            if wrist_distance > shoulder_distance * 1.5 and angle > 70:
                cv2.putText(annotated_image, "shift gear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)
                # Determine whether to shift up or down

                if right_wrist.y < right_shoulder.y:
                    cv2.putText(annotated_image, "Up", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    if fl > 0:
                        j.set_axis(pyvjoy.HID_USAGE_Y, 0x0000)
                        fl -= 1

                elif right_wrist.y > right_hip.y:
                    cv2.putText(annotated_image, "Down", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                    if fl > 0:
                        j.set_axis(pyvjoy.HID_USAGE_Y, 0x7FFF)
                        fl -= 1

            else:
                fl = 3
                j.set_axis(pyvjoy.HID_USAGE_Y, 0x4000)

             # section3: Determine whether the accelerator and brake pedals are pressed.          
            if first_frame_flag == True:
                init_left_knee_y += left_knee.y
                init_right_knee_y += right_knee.y
                frame_num += 1
                if frame_num > 30:
                    init_left_knee_y /= frame_num
                    init_right_knee_y /= frame_num
                    first_frame_flag = False
            else:
                left_y_distance = (left_knee.y - init_left_knee_y) * frame.shape[0]
                right_y_distance = (right_knee.y - init_right_knee_y) * frame.shape[0]
                cv2.putText(annotated_image, "{:.1f}".format(left_y_distance),
                            (int(left_knee.x * frame.shape[1]), int(left_knee.y * frame.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated_image, "{:.1f}".format(right_y_distance),
                            (int(right_knee.x * frame.shape[1]), int(right_knee.y * frame.shape[0])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

                # Check if the angles are between 85 and 95 degrees
                if left_y_distance < -10 and left_y_distance - right_y_distance < -10:
                    cv2.putText(annotated_image, "Brake", (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    left_dis = int((-left_y_distance - 10) / 30 * 32767)
                    j.set_axis(pyvjoy.HID_USAGE_SL0, left_dis)
                else:
                    j.set_axis(pyvjoy.HID_USAGE_SL0, 0x0000)
                if right_y_distance < -10 and right_y_distance - left_y_distance < -10:
                    cv2.putText(annotated_image, "Accelerator", (frame.shape[1] - 200, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    right_dis = int((-right_y_distance - 10) / 30 * 32767)
                    j.set_axis(pyvjoy.HID_USAGE_SL1, right_dis)
                else:
                    j.set_axis(pyvjoy.HID_USAGE_SL1, 0x0000)

        # Show the image
        if input_file_name is None:
            cv2.imshow('Annotated', annotated_image)
        else:
            out.write(annotated_image)
        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    if input_file_name is not None:
        out.release()
    j.reset()


# Execute the main function
if __name__ =='__main__':
    # main(input_file_name = '../data/knee.mp4', output_file_name = '../output/knee.mp4', mirror = False)
    main(mirror= True)



import matplotlib
matplotlib.use('Agg')
import cv2
import mediapipe as mp
import numpy as np
import math
import socket
import time

# Function to calculate frame rate
def calculate_frame_rate(start, end):
    time_diff = end - start
    if time_diff > 0:
        return 1.0 / time_diff
    return 0

# Calculate the angle between two points
def calculate_angle_2d(p1, p2):
    angle = np.arctan2(p2.y - p1.y, p2.x - p1.x)
    return angle * (180 / np.pi)

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def main(input_file_name = None, output_file_name = None, mirror = False):

    # # Create server socket
    server = socket.socket()
    host_ip = get_host_ip()
    port = 1999

    # Bind the address and port number
    server.bind((host_ip, port))
    server.listen(5)
    print(f"Server started at {host_ip}:{port} ...")

    # Wait for client connection
    conn, addr = server.accept()
    

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
    mp_pose = mp.solutions.pose
    pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    # Initialize the drawing utility
    mp_drawing = mp.solutions.drawing_utils

    # Define the colors for the left and right hands
    left_color = (255, 0, 0)  # blue
    right_color = (0, 0, 255)  # red
    first_frame_flag = True
    first_frame_flag_right_hand = True
    init_left_knee_y = 0
    init_right_knee_y = 0
    init_left_knee_x = 0
    init_right_knee_x = 0
    init_right_hand_x = 0
    init_right_hand_y = 0
    frame_num = 0
    frame_hand_num = 0
    start_time = 0
    end_time = 0
    fl = 3

    # Set the minimum update time for delay time and frame rate to prevent numbers from flashing too fast
    last_update_time_t = 0
    last_update_time_fps = 0
    display_t = 0
    display_fps = 0

    while cap.isOpened():
        # Read the camera frame
        
        st = time.time()
        
        ret, frame = cap.read()
        if mirror == True:
        # Mirror the camera image
            frame = cv2.flip(frame, 1)
        # Convert the color space
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run the pose model
        results = pose.process(frame)


        # Draw the skeleton
        annotated_image = frame.copy()
        if results.pose_landmarks:
            # Draw the other parts except the arms
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                   circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                     thickness=2))
            if mirror == True:
                # Since the image is mirrored, the left and right should be flipped when processing, for example, mp_holistic.PoseLandmark.LEFT_SHOULDER actually refers to the right side
                # Draw the left and right arms separately
                left_connections = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,
                                    mp_pose.PoseLandmark.RIGHT_WRIST]
                right_connections = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                                    mp_pose.PoseLandmark.LEFT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
                right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
            else:
                left_connections = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW,
                                    mp_pose.PoseLandmark.LEFT_WRIST]
                right_connections = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW,
                                    mp_pose.PoseLandmark.RIGHT_WRIST]
                left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                right_wrist = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                right_elbow = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]
                right_hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]
                left_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
                right_knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]

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
            if mirror == True:
                angle = calculate_angle_2d(left_wrist, right_wrist)
            else:
                angle = - calculate_angle_2d(right_wrist, left_wrist)

            cv2.putText(annotated_image, str(int(angle)),
                        (int((left_wrist.x + right_wrist.x) / 2 * frame.shape[1]), int((left_wrist.y + right_wrist.y) / 2 * frame.shape[0])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
            hex_value = int((angle + 91) / 182 * 32767)

            # If in gear shifting mode, lock the steering wheel
       
            po_angle = str(hex_value)

            # section2: Check if it's in the shifting gear state
            # Initialize right hand position
            po_gear = 'N'
            if first_frame_flag_right_hand == True:
                init_right_hand_x += right_wrist.x
                init_right_hand_y += right_wrist.y
                frame_hand_num += 1
                if frame_num > 30:
                    init_right_hand_x /= frame_num
                    init_right_hand_y /= frame_num
                    first_frame_flag_right_hand = False


            else:

            # Draw the initial position of the right hand
                cv2.circle(annotated_image,
                (int(init_right_hand_x * frame.shape[1]), int(init_right_hand_y * frame.shape[0])),
                radius=5, color=(0, 255, 0), thickness=-1)

                if mirror == True:
                    right_hand_x_distance = (right_wrist.x - init_right_hand_x) * frame.shape[1]
                else:
                    right_hand_x_distance = (init_right_hand_x - right_wrist.x) * frame.shape[1]

                right_hand_y_distance = ( init_right_hand_y - right_wrist.y) * frame.shape[0]

                # Determine the distance
                shoulder_distance = math.dist((right_shoulder.x, right_shoulder.y), (left_shoulder.x, left_shoulder.y))
                wrist_distance = math.dist((right_wrist.x, right_wrist.y), (left_wrist.x, left_wrist.y))



                    # Check if it is in gear shifting mode
                if wrist_distance > shoulder_distance * 1.3 and right_hand_x_distance > 80:
                    cv2.putText(annotated_image, "shift gear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                                cv2.LINE_AA)
                    po_gear = 'C'
                    # Determine whether to shift up or down
                    if right_hand_y_distance > 55:
                        cv2.putText(annotated_image, "Up", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                        po_gear = 'U'

                    elif right_hand_y_distance < -45:
                        cv2.putText(annotated_image, "Down", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 255, 0), 2, cv2.LINE_AA)
                        po_gear = 'D'

                    if po_gear == 'C':
                        po_gear = 'B'

                else:
                    po_gear = 'N'

            po_brake = 'F'
            po_acce = 'F'
             # section3: Determine whether the accelerator and brake pedals are pressed.          
            if first_frame_flag == True:
                init_left_knee_y += left_knee.y
                init_right_knee_y += right_knee.y

                init_left_knee_x += left_knee.x
                init_right_knee_x += right_knee.x

                frame_num += 1
                if frame_num > 30:
                    init_left_knee_y /= frame_num
                    init_right_knee_y /= frame_num
                    init_left_knee_x /= frame_num
                    init_right_knee_x /= frame_num
                    first_frame_flag = False


            else:

                # Draw the initial position of the left knee
                cv2.circle(annotated_image,
                        (int(init_left_knee_x * frame.shape[1]), int(init_left_knee_y * frame.shape[0])),
                        radius=5, color=(0, 0, 255), thickness=-1)

                # Draw the initial position of the right knee
                cv2.circle(annotated_image,
                        (int(init_right_knee_x * frame.shape[1]), int(init_right_knee_y * frame.shape[0])),
                        radius=5, color=(0, 0, 255), thickness=-1)

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
                    if mirror == True:
                        cv2.putText(annotated_image, "Brake", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(annotated_image, "Brake", (frame.shape[1] - 120, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    left_dis = int((-left_y_distance - 10) / 30 * 32767)
                    po_brake = str(left_dis)
                else:
                    po_brake = 'F'
                if right_y_distance < -10 and right_y_distance - left_y_distance < -10:
                    if mirror == True:
                        cv2.putText(annotated_image, "Accelerator", (frame.shape[1] - 200, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.putText(annotated_image, "Accelerator", (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                    right_dis = int((-right_y_distance - 10) / 30 * 32767)
                    po_acce = str(right_dis)
                else:
                    po_acce = 'F'

            message = po_angle +' '+ po_gear +' '+ po_brake +' '+ po_acce+' '
            print(message)
            # transfer message
            conn.send(str.encode(message))


        # Show the delay
        et = time.time()

        t_delay = et - st
        formatted_t = "{:.4f}".format(t_delay)
        t_delay = "Delay:{}s".format(formatted_t)

        # Update frame rate display every 0.5 seconds
        if time.time() - last_update_time_t >= 0.3:
            display_t = t_delay
            last_update_time_t = time.time()

        cv2.putText(annotated_image, display_t, (frame.shape[1] - 220, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Frame rate calculation
        end_time = time.time()
        fps = calculate_frame_rate(start_time, end_time)
        start_time = end_time

        # Update frame rate display every 0.5 seconds
        if time.time() - last_update_time_fps >= 0.4:
            display_fps = fps
            last_update_time_fps = time.time()

        cv2.putText(annotated_image, f"Frame rate: {display_fps:.2f} FPS", (0, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        # Show the image
        
        if input_file_name is None:
            cv2.imshow('Annotated', annotated_image)
        else:
            out.write(annotated_image)
        
        # Press 'q' to exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    server.close()
    if input_file_name is not None:
        out.release()



# Execute the main function
if __name__ =='__main__':
    # main(input_file_name = '../data/knee.mp4', output_file_name = '../output/knee.mp4', mirror = False)
    main(mirror= False)

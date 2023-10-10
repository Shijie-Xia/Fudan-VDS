import cv2
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# 初始化mediapipe的holistic模型
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 初始化绘图功能
mp_drawing = mp.solutions.drawing_utils


# 计算两点之间的角度
def calculate_angle(p1, p2):
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    return angle * (180 / np.pi)



def main():
    # 打开摄像头
    cap = cv2.VideoCapture(0)

    # 定义左右手的颜色
    left_color = (255, 0, 0)  # 蓝色
    right_color = (0, 0, 255)  # 红色

    while cap.isOpened():
        # 读取摄像头帧
        ret, frame = cap.read()
        # 镜像摄像头图像
        frame = cv2.flip(frame, 1)

        # 转换颜色空间
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 运行holistic模型
        results = holistic.process(image_rgb)

        # 绘制骨架
        annotated_image = frame.copy()
        if results.pose_landmarks:
            # 绘制除了手臂之外的其它部位
            mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2,
                                                                                   circle_radius=2),
                                      connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255),
                                                                                     thickness=2))
            # 由于图像镜像，下面处理时左右应当翻转,例如mp_holistic.PoseLandmark.LEFT_SHOULDER实际指右侧
            # 单独绘制左右手臂
            left_connections = [mp_holistic.PoseLandmark.RIGHT_SHOULDER, mp_holistic.PoseLandmark.RIGHT_ELBOW,
                                mp_holistic.PoseLandmark.RIGHT_WRIST]
            right_connections = [mp_holistic.PoseLandmark.LEFT_SHOULDER, mp_holistic.PoseLandmark.LEFT_ELBOW,
                                mp_holistic.PoseLandmark.LEFT_WRIST]
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

            # 获取手部关键点，并绘制线段

            left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].x * frame.shape[1], \
                          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST.value].y * frame.shape[0]
            right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].x * frame.shape[1], \
                          results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST.value].y * frame.shape[0]

            cv2.line(annotated_image, (int(left_wrist[0]), int(left_wrist[1])),
                     (int(right_wrist[0]), int(right_wrist[1])), (0, 255, 0), 2)

            # 计算角度
            angle = calculate_angle(left_wrist, right_wrist)
            cv2.putText(annotated_image, str(int(angle)),
                        (int((left_wrist[0] + right_wrist[0]) / 2), int((left_wrist[1] + right_wrist[1]) / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

            # 判断是否处于换挡状态
            #两个条件：两手连线的长度>两肩连线的长度*1.3且右手大臂与右手小臂之间的夹角>85°

            left_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            right_shoulder = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            left_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST]
            right_wrist = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST]
            right_elbow = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ELBOW]
            right_hip = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_HIP]


            #判断距离
            shoulder_distance = ((right_shoulder.x - left_shoulder.x) ** 2 + (
                    right_shoulder.y - left_shoulder.y) ** 2) ** 0.5
            wrist_distance = ((right_wrist.x - left_wrist.x) ** 2 + (right_wrist.y - left_wrist.y) ** 2) ** 0.5

            shifting_gear = wrist_distance > shoulder_distance * 1.3

            elbow_wrist_distance = ((right_elbow.x - right_wrist.x) ** 2 + (right_elbow.y - right_wrist.y) ** 2) ** 0.5
            shoulder_elbow_distance = ((right_shoulder.x - right_elbow.x) ** 2 + (
                    right_shoulder.y - right_elbow.y) ** 2) ** 0.5
            shoulder_wrist_distance = ((right_shoulder.x - right_wrist.x) ** 2 + (
                    right_shoulder.y - right_wrist.y) ** 2) ** 0.5

            # 判断角度
            angle = math.degrees(
                math.acos((elbow_wrist_distance ** 2 + shoulder_elbow_distance ** 2 - shoulder_wrist_distance ** 2) /
                          (2 * elbow_wrist_distance * shoulder_elbow_distance)))

            if shifting_gear and angle > 85:
                cv2.putText(annotated_image, "shift gear", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

                #判断升档还是降档

                if right_wrist.y < right_shoulder.y:
                    cv2.putText(annotated_image, "Up", (frame.shape[1] - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)
                elif right_wrist.y > right_hip.y:
                    cv2.putText(annotated_image, "Down", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), 2, cv2.LINE_AA)

        # 显示图像
        cv2.imshow('Annotated', annotated_image)

        # 按'q'键退出
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



# 执行主函数
main()



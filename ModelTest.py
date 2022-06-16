from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pickle
import time
import PoseDetector

#创建类
class ModelTest():
    #打开视频
    def OpenVideo(self,Path):
        cap=cv2.VideoCapture(Path)
        return cap

    #打开摄像头
    def OpenCamera(self):
        cap=cv2.VideoCapture(0)
        return cap



    #判断动作
    def ActionRecognition(self,cap):

        # 模型和标签名
        MODEL_NAME = "ActionModel.h5"
        LABEL_NAME = "ActionLabels.dat"

        # 加载标签
        with open(LABEL_NAME, "rb") as f:
            lb = pickle.load(f)

        # 加载神经网络
        model = load_model(MODEL_NAME)

        while (True):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 3)
            frame = cv2.resize(frame, (1024, 720))

            if ret == True:
                roi = frame
                roi1 = roi
                roi1 = cv2.resize(roi, (100, 100))
                roi1 = np.expand_dims(roi1, axis=0)
                roi1 = roi1 / 255.
                prediction = model.predict(roi1)
                detector = PoseDetector.poseDetector()
                detector.FindPose(roi)
                # 判断动作并打印在屏幕上
                Action_Probability = prediction[0][np.argmax(prediction[0])]
                Action = lb.inverse_transform(prediction)[0]
                cv2.putText(roi, Action, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(roi, ('%.2f' % (Action_Probability * 100)) + '%', (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 1)


            cv2.imshow('Action_Recognition', frame)

            key = cv2.waitKey(5) & 0xff
            # Esc键退出
            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                break

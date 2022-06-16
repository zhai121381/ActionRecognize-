import cv2
import mediapipe as mp
import time

#定义poseDetector类
class poseDetector():
    def __init__(self):
        #初始化mediapope
        self.mpDraw=mp.solutions.drawing_utils
        self.mpPose=mp.solutions.pose
        self.pose=self.mpPose.Pose()
#定义方法用于判断人体骨骼
    def FindPose(self,img,draw=True):

        #将传入的img图片通道变成RGB
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #用mediapipe处理图片识别出骨骼点
        self.results =self.pose.process(imgRGB)

        #判断是否找出了骨骼点
        if self.results.pose_landmarks:

            #如果找出了就判断是否要画出骨骼点
            if draw:
                #在图片中画出骨骼点并连接起来
                #self.mpDraw.DrawingSpec(color=(0,0,255),circle_radius=5)和
                #self.mpDraw.DrawingSpec(color=(0,255,0)分别定义骨骼点的颜色大小和连接线的颜色大小
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS,
                                           self.mpDraw.DrawingSpec(color=(0,0,255),circle_radius=5),
                                           self.mpDraw.DrawingSpec(color=(0,255,0)))
        return img#返回处理完的图片

    def getPosition(self,img,draw=True):
        #定义一个列表用于存储骨骼点的数组
        lmList=[]
        if self.results.pose_landmarks:
            #从获取的骨骼点数据中读取出各个骨骼点的id和位置
            for id,lm in enumerate(self.results.pose_landmarks.landmark):
                #获取图像的长宽和通道
                h,w,c=img.shape
                cx,cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img,(cx,cy),25,(0,0,255),cv2.FILLED)
        return lmList#返回获取的数据


#测试用主函数
def main():
    cap = cv2.VideoCapture("E:/DataFile/Code/video2.mp4")
    #cap=cv2.VideoCapture(0)
    pTime = 0
    detector=poseDetector()
    while True:
        success , img = cap.read()
        img=detector.FindPose(img)
        cv2.imshow('Image',img)
        lmList=detector.getPosition(img)
        print(lmList)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str("fps:" + f"{int(fps)}"), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.waitKey(10)

if __name__=="__main__":
    main()
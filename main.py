import os
import GetAcitionData
import ModelTest
import TrainModel
import PoseDetector
import cv2
import keyboard
import time

class MainTest():

#读取各种信息方法
    def ReadInfo(self):
        # 保存需要学习的动作的视频地址
        path = input("请输入需要读取的视频地址")

        # 保存所读取动作的名称
        ActionName = input("请输入动作的名称")

        # 保存需要存储的动作图片数量
        ActionNum = eval(input("请输入需要保存的图片数量"))

        #返回信息
        return path,ActionName,ActionNum

#保存处理过的动作图片信息
    def GetActionInfo(self,cap,ActionName,ActionNum,DataPath):

        #计数器
        count=0

        #创建SaveDate的对象
        SaveIm=GetAcitionData.SaveDate()

        while True:
            #当数据存储数量小于设定数量时继续
            if count<ActionNum:

                #定义img保存图片
                success,img=cap.read()

                #创建PoseDetector中poseDetector类的对象
                detector=PoseDetector.poseDetector()

                #使用对象中的FindPose方法画出骨骼
                img=detector.FindPose(img)

                #存储处理完的图像
                SaveIm.saveimg(img,count,ActionName,DataPath)

                #每进行一次存储计数器+1
                count=count+1



#测试用主函数
def main():
    #创建ModelTest对象
    OpenVideo=ModelTest.ModelTest()
    #创建TrainModel对象
    train=TrainModel.TrainNetWork()
    #创建MainTest对象
    Test=MainTest()

    # 保存获取的动作数据的地址
    ActionPath = "./ActionData/"
    while True:

        #keyboard函数判断按键为何
        #按下s键输入需要学习的视频地址和视频名称以及预备学习的数量
        if keyboard.is_pressed('s'):

            #按下s后程序睡眠一秒再运行防止多次启动
            time.sleep(1)

            #引用MainTest对象中的ReadInfo方法对视频地址，视频名称，截取次数进行设定
            path,ActionName,ActionNum=Test.ReadInfo()

            #传入地址并打开视频
            cap = OpenVideo.OpenVideo(path)

            #创建当前动作名称文件夹
            try:
                os.makedirs('./ActionData/' + ActionName)
            except OSError as e:
                print(ActionName + '文件夹已创建')

            #重新设置保存动作数据文件路径
            ActionPath=ActionPath+ActionName+"/"

            #对动作文件进行截取并进行处理后保存
            Test.GetActionInfo(cap,ActionName,ActionNum,ActionPath)

            print("fine")

            #释放视频文件
            cap.release()
            continue

        #按下t键输入需要学习的动作个数和迭代次数，并开始学习
        elif keyboard.is_pressed('t'):
            time.sleep(1)
            print("即将开始训练")
            #引用TrainModel中的train方法对保存的数据文件进行学习
            train.Train()
            continue

        #按下f键打开摄像头并判断动作
        elif keyboard.is_pressed('f'):
            #打开摄像头
            cap=OpenVideo.OpenCamera()

            #引用ModelTest中的ActionRecognition方法判断动作
            OpenVideo.ActionRecognition(cap)
            continue










    cap=OpenVideo.OpenCamera()

    #train.Train()
    OpenVideo.ActionRecognition(cap)


if __name__=="__main__":
    main()
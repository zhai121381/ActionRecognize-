import time
import os
import cv2
import PoseDetector

#定义类实现保存图片的方法
class SaveDate():
    #保存动作图片方法
    def saveimg(self,img,ActionNum,ActionName,DataPath):
        #定义存储图片的名字
        AcName=ActionName+str(ActionNum)

        #将照片存储至某个路径
        cv2.imwrite(DataPath+AcName+".png",img)

        #在控制台输出刚刚存储的照片的名字
        print("Picture---"+AcName)


# ActionRecognize-
根据CNN学习识别人体姿态和动作
本项目中共有四个模块
GetActionData.py
main.py
ModelTest.py
PoseDetector.py
TrainModel.py

1.PoseDetector.py，在该文件中定义了一个poseDetector类，在该类中主要实现了两个方法，Findpose和getposition
findpose方法中，使用了mediapipe库中自动寻找人体姿态的骨骼点的内置方法pose.process()，在获得骨骼点数据以后将结果保存在result中
同时通过mediapipe的内置方法Draw将骨骼点的数据全部标注在图片中并连接起来

2.GetActionData.py，在该文件中定义了一个用于保存图片的方法，使用该方法会调用Opencv的保存图像方法将图像写入本地，用于接下来的训练

3.TrainModel.py，在该文件中定义了一个进行卷积神经网络训练的方法，当调用该方法的时候会先从GetActionData.py方法所保存下来的图片中
读取出所有需要学习的动作图片的标签，随后将标签和相对应的图片保存至列表中，在保存完以后会对标签和图像进行处理，最后调用卷积神经网络对
图片进行学习并生成ActionModel.h5文件和ActionLabels.dat文件，这两个文件中所存储的内容就是训练完的数据和标签

4.ModelTest.py，该文件中共定义了三个方法，一个是调用opencv中的videocapture方法打开视频，一个是通过该方法打开电脑自带的摄像头
另一个是ActionRecognition方法，在该方法中会加载神经网络并对当前输入的视频数据进行读取，然后通过opencv创建一个窗口将判断出的数据
打印在窗口上

5.main.py，该文件将以上所有的方法和功能进行了汇总，运行main文件需要先按下s键，随后会提示输入需要学习的动作视频的地址，这里需要最起码
学习三个动作
随后是按下t键，按下这个键以后会调用TrainModel中的方法对存储的数据进行训练
训练完成之后可以按下f键，此时将会打开摄像头并读取摄像头的数据，接着会调用ModelTest中的方法判断当前摄像头读取出的动作
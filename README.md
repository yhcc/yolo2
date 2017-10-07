# yolo2  
该项目是https://pjreddie.com/darknet/yolo 的keras(backend:tensorflow)实现，原项目是一个可实时进行track object的项目。作者原有实现是C语言的。本项目结合作者该篇文章以及其之前的若干篇文章写出了基于keras的代码。

# 效果图  
> **car、dog和truck**   
> ![dog](https://github.com/yhcc/yolo2/blob/master/images/out/dog.jpg)  
> **giraffe和zebra**   
> ![giraffe](https://github.com/yhcc/yolo2/blob/master/images/out/giraffe.jpg)  
更多效果图可在images/out下看到，所生成的图片均为通过本项目生成。一个在线的视频demo在这里https://youtu.be/T7XOFJdu_y8 。

> ### 1. 准备神经网络model  
> 1) 下载作者提供的weights文件，在http://pjreddie.com/media/files/yolo.weights 下载。
> 2) 逐行运行model.ipynb文件，该脚本是将weights文件生成为一个keras的model文件。运行结束之后，model为model_data/model.h5  
> ### 2. 运行model  
> 1) 通过使用python detect_image.py model_data/model.h5 -t images/ -o images/out, 更多参数选择请参考--help函数.第一个参数是model的路径，-t是使用哪个文件夹下的图片来做为input，-o是把生成的标注了的图片放到什么位置。    
> ### 3. train model  
> 1) 如果希望对model进行train，使用train_model.ipynb, cell 3的第一个参数是使用哪个model进行train，第二参数是train的时候使用的优化方法。cell 4中是进行train的函数，第二个参数是train的数据来源。  
> ### 4. 视频识别
> 1) 通过运行python detect_video.py model_data/model.h5 -t images/xxx.avi -o demo.mp4即可识别视频中的物体。输出效果类似之前的视频demo  

> ## train的这部分的算法说明
> (1) 因为按照道理，应该直接读取model.h5，然后使用model.fit()这样来train,　但是这样有个问题，model的输出是(batch, 13, 13, 425)的形状, 而提供的y的形状是类似于data.txt这种，这两者之间根本没有办法算loss(因为keras要求，y的形状必须和model.output的形状一样）。  
> (2) 于是就需要找解决方案，找到的方案是，使用data_util.py中的data_generator()来产生y, 读取data.txt一行，然后产生出y，y的形状比较奇特，实际上一个 list,　里面的元素有\[方框的信息，object_mask, object_value, 全１序列\](这里是用来计算loss构造的).  
> (3) 此外把model传入train_model.py中的model_to_train()，在这里构造最后可以用来train的函数，大概思想是使用loss_util.py中loss_calculator()算出model.output和y(复杂的list那个)的loss(把这个loss当作构造的model的输出，这个loss是一个常数了，意味着我可以构造一个新的y(把这个新的y选为全是１)来和它匹配，以使得keras可以用它作为训练), 然后让构造的model减小的loss＝(新的y*loss), 又因为新的y全是１，所以实际上就是减小了loss, 通过这方式来使得原来model的weight得到更新。这里比较绕，比较tricky。  
> (4) train_model.py中的train_model()就只是执行了fit()的过程。结果会保存一份model的weight  


#yolo2
该项目是基于https://pjreddie.com/darknet/yolo/的keras(后台为tensorflow)实现，原项目主要是一个实时进行track object类似的项目。作者提供了实现的文章以及C语言的代码。本项目结合作者该篇文章以及其之前的若干篇文章进行了写出了基于keras的代码。

效果图可在images／out下看到，所生成的image均为通过本项目生成，一个在线的demo在这里https://youtu.be/T7XOFJdu_y8。

使用说明：
1.下载yolo.weight用于生成keras的model
wget http://pjreddie.com/media/files/yolo.weights

2. 使用yolo.weight生成model..h5(放在了model_data/model.h5下面)
python model.py darknet -w yolo.weights -s model_data/model.h5

３.使用这个model.h5做一个预测。
python detect_image.py model_data/model.h5
	预测图片来源是images/。下面有五张照片
	原图和输出的照片在image文件夹下，其中images/out_raw/是使用官方的代码运行出来的。除了输出的图片，在命令行还会输出图片框的位置，下面的红色部分
Found 2 boxes for giraffe.jpg
zebra 0.80 (270, 163) (451, 488)
giraffe 0.88 (130, 0) (494, 477)
Found 0 boxes for scream.jpg
Found 1 boxes for eagle.jpg
bird 0.90 (74, 37) (688, 498)
Found 3 boxes for dog.jpg
truck 0.78 (404, 68) (741, 184)
dog 0.79 (93, 178) (356, 572)
bicycle 0.83 (62, 81) (635, 481)
Found 5 boxes for horses.jpg
horse 0.41 (42, 150) (520, 375)
horse 0.69 (210, 176) (453, 394)
horse 0.72 (0, 171) (176, 410)
horse 0.82 (411, 191) (627, 368)
horse 0.90 (0, 159) (384, 446)
Found 3 boxes for person.jpg
dog 0.82 (41, 251) (221, 364)
horse 0.86 (375, 103) (631, 375)
person 0.87 (178, 52) (288, 422)
	数据结构：哪张图，发现了什么，多大概率，框的坐标(x_left_top,y_left_top, x_right_down, y_right_down). 
	为了证明我们写的train函数是有用的，直接把这个信息转变成用来train的数据。转换结果放在了data/data.txt(另一个customized_data.txt就不用管了，是之前准备只用这五张图片来train一个只识别几类物品的model,结果因为只有五张图片，实在效果不好，就放弃了). 每条数据的格式是
	path, x1, y1, w1, h1, class1, x2, y2, w2, h2
每条长度不定，因为每个图框的数量是不一定的。

4.使用上面的data.txt对model.h5进行finetune, 然后再来预测这五张image, 如果概率更高，而且框圈得越准确，说明train是有用的。
(1)打开jupyter notebook, 然后运行train_model.ipynb.
(2)按照每条执行下来即可，具体参数代表的意思在每个函数里面有部分解释。
(3)然后使用下面的命令再预测一次，因为train_model.ipynb出来的new_model.h5其实只是一个weight, 所以先load　model模型，然后将weight替换为new_model.h5,　这部分在detect_image.py已经实现了。
python detect_image.py model_data/model.h5  -w model_data/new_model.h5

模型不好train的原因:
(1)模型太复杂, 如果从新train一个模型的话，数据量太大，即便是fine-tune也时间不够
(2)存在一些hyperparameter，不好找最佳值，比如loss_util.py中的alpha1,alpha2...等


这部分图片存在images/out_finetune/上
经过下面的对比，发现圈出的框变小了。对test集有了一点overfitting，证明我们的train还是起了作用。另外confidence有一点下降，是因为我们的train的参数选择了x,y,w,h的loss weight更大(loss_util.py中的alpha1, alpha2对比alpha3)



5. 整个train的这部分比较tricky。这是train的原理
(1). 因为按照道理，应该直接读取model.h5，然后使用model.fit()这样来train,　但是这样有个问题，model的输出是(batch, 13, 13, 425)的形状, 而提供的y的形状是类似于data.txt这种，这两者之间根本没有办法算loss(因为keras要求，y的形状必须和model.output的形状一样）。
(2)于是就需要找解决方案，找到的方案是，使用data_util.py中的data_generator()来产生y, 读取data.txt一行，然后产生出y，y的形状比较奇特，实际上一个list,　里面的元素有[方框的信息，object_mask, object_value, 全１序列](这里是用来计算loss构造的).　
(3)此外把model传入train_model.py中的model_to_train()，在这里构造最后可以用来train的函数，大概思想是使用loss_util.py中loss_calculator()算出model.output和y(复杂的list那个)的loss(把这个loss当作构造的model的输出，这个loss是一个常数了，意味着我可以构造一个新的y(把这个新的y选为全是１)来和它匹配，以使得keras可以用它作为训练), 然后让构造的model减小的loss＝(新的y*loss), 又因为新的y全是１，所以实际上就是减小了loss, 通过这方式来使得原来model的weight得到更新。这里比较绕，比较tricky。
(4)train_model.py中的train_model()就只是执行了fit()的过程。结果会保存一份model的weight

6. detect_video.py，这个实现了直接识别视频，甚至可以实时的识别，具体的demo可以看之前的README中有。


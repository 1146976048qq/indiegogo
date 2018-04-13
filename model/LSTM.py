import numpy as np
import datetime as dt

start_time = dt.datetime.now()
print(start_time)

wordsList = np.load('wordsList.npy')
print('Loaded the word list !\n')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')
print('Load the word vectors !\n')

print("词汇列表的长度 ：", len(wordsList))
print("词向量的纬度 ：", wordVectors.shape)

# 得到单词 basketball 的向量表示
basketballIndex = wordsList.index('basketball')
wordVectors[basketballIndex]

# 输入一个句子，构造其向量表示
import tensorflow as tf

maxSeqLength = 10  # Maxinum of sentence
numDimentions = 300  # Dimensions for each word vector
firstSentence = np.zeros((maxSeqLength), dtype='int32')
firstSentence[0] = wordsList.index("i")
firstSentence[1] = wordsList.index("thought")
firstSentence[2] = wordsList.index("the")
firstSentence[3] = wordsList.index("movie")
firstSentence[4] = wordsList.index("was")
firstSentence[5] = wordsList.index("incredible")
firstSentence[6] = wordsList.index("and")
firstSentence[7] = wordsList.index("inspiring")
# firstSentence[8,9,10] = wordsList.index(0)
print("句子的形式：", firstSentence.shape)
print("Shows the row index for each word: ", firstSentence)

# The 10 x 50 output should contain the 50 dimensional word vectors
# for each of the 10 words in the sequence
with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors, firstSentence).eval().shape)

# 可视化数据规模，类型
import numpy as np
from os import listdir
from os.path import isfile, join

load_po_ne_file_time = dt.datetime.now()
# 用于训练模型的 数据集
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]
numWords = []
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Positive files finished')

for pf in negativeFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
print('Negative files finished')

numFiles = len(numWords)
print("总文件数: ", numFiles)
print("文件中的总单词数量: ", sum(numWords))
print("文件中的平均单词数量: ", sum(numWords) / len(numWords))

print("load_po_ne_file_time: ", dt.datetime.now()-load_po_ne_file_time)

# # 使用 Matplot 将数据进行可视化
# import matplotlib.pyplot as plt
#
# # %matplotlib inline
# plt.hist(numWords, 50)
# plt.xlabel('Sequece Length')
# plt.ylabel('Frequency')
# plt.axis([0, 1200, 0, 8000])
# plt.show()

# 从直方图可以得出合适的句子最大长度值（设定为250）
maxSeqLength = 250

# 将单个文件中的文本转换成索引矩阵，举例如下（某条评论）
fname = positiveFiles[3]
with open(fname) as f:
    for lines in f:
        print(lines)
        exit

# 接下来将该评论转换为索引矩阵
# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re

strip_special_chars = re.compile("[^A-Za-z0-9]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


firstFile = np.zeros((maxSeqLength), dtype='int32')
with open(fname) as f:
    indexCounter = 0
    line = f.readline()
    cleanedLine = cleanSentences(line)
    split = cleanedLine.split()
    for word in split:
        try:
            firstFile[indexCounter] = wordsList.index(word)
        except ValueError:
            firstFile[indexCounter] = 399999  # vector for unknown words
        indexCounter = indexCounter + 1
firstFile

# 用想通的方法处理25000条评论，导入电影训练集，得到一个25000*250的矩阵
# 将25000条训练数据集导入模型训练
# ids = np.zeros((numFiles, maxSeqLength), dtype='int32') # 返回给定形状和类型的用0填充的数组 25000*250
# fileCounter = 0
# for pf in positiveFiles:
#     with open(pf, "r", encoding='utf-8') as f:
#         indexCounter = 0
#         line = f.readline()
#         cleanedLine = cleanSentences(line)
#         split = cleanedLine.split()
#         for word in split:
#             try:
#                 ids[fileCounter][indexCounter] = wordsList.index(word)
#             except ValueError:
#                 ids[fileCounter][indexCounter] = 399999  # vector for unknown words
#             indexCounter = indexCounter + 1
#             if indexCounter >= maxSeqLength:
#                 break
#         fileCounter = fileCounter + 1
#
# for nf in negativeFiles:
#     with open(nf, "r", encoding='utf-8') as f:
#         indexCounter = 0
#         line = f.readline()
#         cleanedLine = cleanSentences(line)
#         split = cleanedLine.split()
#         for word in split:
#             try:
#                 ids[fileCounter][indexCounter] = wordsList.index(word)
#             except ValueError:
#                 ids[fileCounter][indexCounter] = 399999  # vector for unknown words
#                 indexCounter = indexCounter + 1
#                 if indexCounter >= maxSeqLength:
#                     break
#         fileCounter = fileCounter + 1
# np.save('idsMatrix', ids)  # 保存处理好的索引文件

# ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
# fileCounter = 0
# for pf in positiveFiles:
#    with open(pf, "r", encoding="utf-8") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
#
# for nf in negativeFiles:
#    with open(nf, "r", encoding="utf-8") as f:
#        indexCounter = 0
#        line=f.readline()
#        cleanedLine = cleanSentences(line)
#        split = cleanedLine.split()
#        for word in split:
#            try:
#                ids[fileCounter][indexCounter] = wordsList.index(word)
#            except ValueError:
#                ids[fileCounter][indexCounter] = 399999 #Vector for unkown words
#            indexCounter = indexCounter + 1
#            if indexCounter >= maxSeqLength:
#                break
#        fileCounter = fileCounter + 1
# #Pass into embedding function and see if it evaluates.
#
# np.save('idsMatrix', ids)

ids = np.load('idsMatrix.npy')

# Couple of helper functions
# That will be useful when training the network

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1, 10499)
            labels.append([1, 0])
        else:
            num = randint(14499, 24999)
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels


def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(10499, 14499)
        if (num <= 12499):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = ids[num-1:num]
    return arr, labels

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000


# RNN 模型
import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimentions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

# lstm 设置
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

print("超参数-------------")
# 超参数设置
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
# bias = tf.Variable(tf.random_normal(shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)  # 最终输出值

# 定义正确的预测函数和正确率评估参数
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

# define a standard cross entropy loss with a softmax layer   交叉熵定义损失函数
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# visualize the process of loss and accuracy
import datetime

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

print(" - - - 开始训练模型 - - - : ", datetime.datetime.now())
# Train the model
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):
    # next batch of reviews
    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    # write summary to Tensorboard
    if(i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    # save the network every 10000 training iterations
    if(i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("save to %s" % save_path, dt.datetime.now())
writer.close()
print(" - - - 模型训练完毕 - - - : ", datetime.datetime.now())

# # load trained model
# sess = tf.InteractiveSession()
# saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

## Model Test
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch();
    print("Accuracy for this batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100, "%")

end_time = dt.datetime.now()
print(end_time)
time = (start_time-end_time)
print("LSTM waste time : ", time)
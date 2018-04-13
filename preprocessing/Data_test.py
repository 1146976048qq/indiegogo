from imp import reload

import pandas as pd
import csv
pd.set_option('display.width', 2000)
#encoding=utf8
import sys
reload(sys)

import csv

# indie_backer_finally = pd.read_csv('C:/Users/KKZHANG/PycharmProjects/indiegogo/indie_backer_finally.txt', header=None, encoding='utf-8', error_bad_lines=False)
# print(indie_backer_finally.head())
# indie_backer_finally.to_csv('C:/Users/KKZHANG/PycharmProjects/indiegogo/out_backer.csv')

# indiegogo_data_comment = pd.read_csv('indiegogo_oerk_finally_2.txt', encoding='utf-8')
# print(indiegogo_data_comment.head())
# print(indiegogo_data_project.info())
# print("OKOK" * 10)



# indiegogo_data_project = pd.read_csv('C:/Users/KKZHANG/Desktop/Indiegogo_Data/my/indie_cam_spider_finally_new.csv', header=None, sep=',')

# project_info = csv.DictReader(open('C:/Users/KKZHANG/Desktop/Indiegogo_Data/my/indie_cam_spider_finally_new.csv', 'r', encoding='utf-8'))
# project_data = []
# for lines in project_info:
#     project_data.append(lines)
# row_length = len(project_data)

# data = pd.read_table('C:/Users/KKZHANG/Desktop/indie_cam_spider_finally_new.txt', sep=',')




# f = open('indie_perk_finally.csv', 'r', errors='ignore')
# outF = open('indiegogo_oerk_finally_2.txt', 'w')
# for raw in f:
#     print(str(raw), file=outF)
#
#
# f.close()
# outF.close()



# from textblob import TextBlob
#
# text = '''
# The titular threat of The Blob has always struck me as the ultimate movie
# monster: an insatiably hungry, amoeba-like mass able to penetrate
# virtually any safeguard, capable of--as a doomed doctor chillingly
# describes it--"assimilating flesh on contact.
# Snide comparisons to gelatin be damned, it's a concept with the most
# devastating of potential consequences, not unlike the grey goo scenario
# proposed by technological theorists fearful of
# artificial intelligence run rampant.
# '''
#
# blob = TextBlob(text)
# blob.tags           # [('The', 'DT'), ('titular', 'JJ'),
#                     #  ('threat', 'NN'), ('of', 'IN'), ...]
#
# blob.noun_phrases   # WordList(['titular threat', 'blob',
#                     #            'ultimate movie monster',
#                     #            'amoeba-like mass', ...])
#
# for sentence in blob.sentences:
#     print(sentence.sentiment.polarity)
# # 0.060
# # -0.341
#
# blob.translate(to="es")  # 'La amenaza titular de The Blob...'

















#
# # read data
# indiegogo_data_comment = pd.read_csv('C:/Users/KKZHANG/Desktop/Indiegogo_Data/my/indie_comment_finally.csv', header=None)
# #
# print(indiegogo_data_comment.shape)
# print(indiegogo_data_comment.head())
#
# comment = indiegogo_data_comment.iloc[:, 12]
# project_id = indiegogo_data_comment.iloc[:, 1]
# reply = indiegogo_data_comment.iloc[:, 15]
# account = indiegogo_data_comment.iloc[:, 6]
#
# print(comment.head(3))
# print("\n", "评论数量 : ", comment.shape, "\n")
#
# print("去重后的project_id 数据", project_id.drop_duplicates().shape)
# print("查看项目是否有多条评论\n", project_id.duplicated().head(3))
#
# # print("reply 数据", reply.head(3), "\n")
# print("回复评论数：", reply.shape, "\n")
#
# print("账户数量：", account.shape)
# print("去重账户数量：", account.drop_duplicates().shape)
#
# # comment.to_csv('comment.csv')
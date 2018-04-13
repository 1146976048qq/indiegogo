# -*- coding: utf-8 -*-
# 导入所需的python库

import pandas as pd
pd.set_option('display.width', 2000)

# 读取数据集 indie_comment_finally
comment = pd.read_csv('C:/Users/KKZHANG/PycharmProjects/indiegogo/indie_comment_finally.txt', header=None, encoding='utf-8', error_bad_lines=False)
# print(comment.head(3))

# 设置字段（列）名称
col_names = ['comment_id', 'project_id', 'owne_id', 'account_name', 'account_id', 'comment']
comment.columns = col_names

# 将 comment_id 设置为数据集的index，并删除原 comment_id 所在列
comment.index = comment['comment_id']
comment.drop('comment_id', axis=1, inplace=True)
# 查看处理后的数据集，输出前10行
print('-----'*30, '\n', comment.head(10))
print(comment.shape)

# #将 comment 列保存为单文件
# comment['comment'].to_csv('C:/Users/KKZHANG/Desktop/data_preprocess/all_comment.csv', header=None)

df_comment = pd.DataFrame(comment)
df_comment = df_comment.dropna()
# num_pro = df_comment.drop_duplicates('project_id').count()
# print("Comment 下的项目数 量: ", num_pro)

# # 每个项目对应的评论数
# num = df_comment[u'project_id'].value_counts()
# print("项目对应的评论数：\n", num)

# # 删选评论数超过50的项目
# print(df_comment.columns)
# fifty_project = df_comment[[df_comment.count('project_id')]>50]
# print(fifty_project.head(5))




# 删除 owned_id,account_name, account_id 列，便于LSTM处理
pure_comment = df_comment.drop(['owne_id', 'account_name', 'account_id'], axis=1)
print(pure_comment.shape, pure_comment['comment'].head())
df_pure_comment = pd.DataFrame(pure_comment)

# 删除相同的评论
df_pure_comment.drop_duplicates(['comment'])
df_pure_comment.to_csv('C:/Users/KKZHANG/Desktop/data_preprocess/pure_comment.csv', header=None)
print(df_pure_comment.shape)



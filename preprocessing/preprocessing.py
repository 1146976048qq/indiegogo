# -*- coding: utf-8 -*-
# 导入所需的python库

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
pd.set_option('display.width', 2000)

# 特征工程
# 下面分别对 indie_backer_finally, indie_backer_finally_old, indie_cam_spider_finally_new
# indie_cam_spider_finally_old, indie_comment_finally,indie_comment_reply,indie_owner_finally,indie_owner_finally_old
# indie_perk_finally,indie_team_finally,indiegogo_product 进行预处理

print("\n" * 2,'*****************indie_backer_finally_old********************')
# indie_backer_finally
# 读取数据集
indie_backer_finally = pd.read_csv('C:/Users/KKZHANG/Desktop/Indiegogo_Data/my/indie_backer_finally.csv', header=None)

# 设置字段（列）名称
col_names = ['id', 'backer_name', 'backer_id', 'backer_profile_url', 'time_ago', 'pledger_image_url', 'display_amount', 'display_amount_iso_code',
            'pledge_details_url', 'order_id', 'project_id', 'full_json']
indie_backer_finally.columns = col_names
# print(indie_backer_finally.head(5), '\n', '-----'*30, '\n', indie_backer_finally.count())

# 将 id（id）设置为数据集的index，并删除原 id 所在列
indie_backer_finally.index = indie_backer_finally['id']
indie_backer_finally.drop('id', axis=1, inplace=True)
# 查看处理后的数据集，输出前3行
print('-----'*30, '\n', indie_backer_finally.head(3))
print(indie_backer_finally.shape)



print('\n'*5, '*****************indie_backer_finally_old********************')
# indie_backer_finally_old
# 读取数据集
indie_backer_finally_old = pd.read_csv('C:/Users/KKZHANG/Desktop/Indiegogo_Data/my/indie_backer_finally_old.csv', header=None)
indie_backer_finally_old.columns = col_names
# print(indie_backer_finally_old.head(5), '\n', '-----'*30, '\n', indie_backer_finally_old.count())

# 将 id（id）设置为数据集的index，并删除原 id 所在列
indie_backer_finally_old.index = indie_backer_finally_old['id']
indie_backer_finally_old.drop('id', axis=1, inplace=True)
# 查看处理后的数据集，输出前3行
print('-----'*30, '\n', indie_backer_finally_old.head(3))
print(indie_backer_finally_old.shape)



print('\n'*5, '*****************indie_cam_spider_finally_new********************')
# indie_cam_spider_finally_new
# 读取数据集
indie_cam_spider_finally_new = pd.read_csv('C:/Users/KKZHANG/Desktop/Indiegogo_Data/my/indie_cam_spider_finally_new.csv', header=None)


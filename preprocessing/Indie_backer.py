# -*- coding: utf-8 -*-
# 导入所需的python库

import pandas as pd
pd.set_option('display.width', 2000)

# 读取数据集   indie_backer_finally
backer = pd.read_csv('C:/Users/KKZHANG/PycharmProjects/indiegogo/indie_backer_finally.txt', header=None, encoding='utf-8', error_bad_lines=False)
# print(backer.head(3))

# 设置字段（列）名称
col_names = ['id', 'backer_name', 'backer_id', 'display_amount', 'order_id', 'project_id']
backer.columns = col_names

# 将 id（id）设置为数据集的index，并删除原 id 所在列
backer.index = backer['id']
backer.drop('id', axis=1, inplace=True)
# 查看处理后的数据集，输出前10行
print('-----'*30, '\n', backer.head(10))
print(backer.shape)


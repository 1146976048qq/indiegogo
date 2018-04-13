# -*- coding: utf-8 -*-
# 导入所需的python库

import pandas as pd
pd.set_option('display.width', 2000)


# 读取数据集 comment
owner = pd.read_csv('C:/Users/KKZHANG/PycharmProjects/indiegogo/indie_owner_finally.txt', header=None, encoding='utf-8', error_bad_lines=False)
print(owner.head(3))

# 设置字段（列）名称 ----- 10个feature,223148
col_names = ['owner_id', 'owner_name', 'owner_description', 'owner_campaigns_count', 'owner_contributions_count', 'friends_count', 'owner_comments_count', 'facebook_verified'
             , 'email_verified', 'linkedin_verified', 'admin_verified']
owner.columns = col_names

# 将 id（owner_id）设置为数据集的index，并删除原 owner_id 所在列
owner.index = owner['owner_id']
owner.drop('owner_id', axis=1, inplace=True)
# 查看处理后的数据集，输出前10行
print('-----'*30, '\n', owner.head(10))
print(owner.shape)


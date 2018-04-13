# -*- coding: utf-8 -*-
# 导入所需的python库

import pandas as pd
pd.set_option('display.width', 2000)


# 读取数据集 project
project = pd.read_csv('C:/Users/KKZHANG/PycharmProjects/indiegogo/indie_cam_spider_finally_new.txt', header=None, encoding='utf-8', error_bad_lines=False)
print(project.head(3))

# 设置字段列名称 ------ 16个feature,121128
col_names = ['id', 'project_id', 'explore', 'category', 'project_type', 'title', 'tagline', 'campaign_goal_amount', 'campaign_raised_amount', 'funds_raised_amount',
             'funds_raised_percent', 'campaign_days_left', 'campaign_country', 'campaign_name', 'campaign_description', 'campaign_type', 'owner_id']

project.columns = col_names

project.index = project['project_id']
project.drop('project_id', axis=1, inplace=True)
project.drop('id', axis=1, inplace=True)
# 查看处理后的数据集，输出前10行
print('-----'*30, '\n', project.head(10))
print(project.shape)



df_project = pd.DataFrame(project)

# 统计成功的项目数量
succeed_project = df_project[df_project['funds_raised_percent']>=1]
print("成功项目的数量：", succeed_project[u'funds_raised_percent'].count())
# 统计失败的项目数量
failed_project = df_project[df_project[u'funds_raised_percent']<1]
print("失败项目的数量：", failed_project[u'funds_raised_percent'].count())
# print(df.shape)




import csv
import pandas as pd
pd.set_option('display.width', 2000)


# 将不能读的 indie_cam_spider_finally_new.txt 文件重新转一下格式编码
f = open('C:/Users/KKZHANG/PycharmProjects/indiegogo/indie_cam_spider_finally_new.txt', 'r', errors='ignore')
out = open('C:/Users/KKZHANG/PycharmProjects/indiegogo/project.txt', 'w')
for row in f:
    print(str(row), file=out)
f.close()
out.close()
import matplotlib.pyplot as plt
import os
import matplotlib
import seaborn as sns

dir_path = r"I:\Dataset\Train_Set\Torch\train_all"
# dir_path = r"I:\Dataset\pytorch\train_all"
matplotlib.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus']=False     # 正常显示负号
x = []
y = []
cnt = 0
sum = 0
for itm in os.listdir(dir_path):
    new_dir = os.path.join(dir_path,itm)
    num = len(os.listdir(new_dir))
    x.append(int(itm))
    y.append(num)
    print(num,"|" , end="")
    cnt += 1
    sum += num
print("-"*5)
print(float(sum)/cnt)
sns.barplot(x=x,y=y)

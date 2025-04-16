import matplotlib.pyplot as plt
import seaborn as sns



def plot_bar(data,x_label,y_label,title,rotation=None,bottom_margin=None):
    # plt.figure(figsize=(10,6))
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    total = sum(y)
    plt.bar(x,y)
    
    # 在每个柱形上方添加数值标签
    for i in range(len(x)):
        percentage = y[i] / total * 100
        plt.text(x[i], y[i], f"{percentage:.2f}%", 
                ha='center', va='bottom')
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if rotation:
        plt.xticks(ticks=x,rotation=rotation)
    if bottom_margin:
        plt.subplots_adjust(bottom=bottom_margin)
    plt.show()
    # plt.savefig(f"{title}.png")

def plot_bie(data,title):
    plt.figure(figsize=(10,6))
    value = [i[1] for i in data]
    label = [i[0] for i in data]
    plt.pie(value,labels=label,autopct='%1.1f%%')
    plt.title(title)
    plt.axis('equal')
    plt.show()
    # plt.savefig(f"{title}.png")

def plot_density(data,x_label,y_label,title):
    # 创建图形
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, fill=True)
    
    # 设置标签和标题
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # 显示图形
    plt.show()
    # plt.savefig(f"{title}.png")
import matplotlib.pyplot as plt
import seaborn as sns


plt.figure(figsize=(15/2.6,9/2.6))

def plot_bar(data,x_label,y_label,title,rotation=None,bottom_margin=None):
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    total = sum(y)
    plt.bar(x,y,color='#e1ccb1')
    
    # 在每个柱形上方添加数值标签 - 使用Times New Roman显示数字
    for i in range(len(x)):
        percentage = y[i] / total * 100
        if x[i] <=8:
            plt.text(x[i], y[i], f"{percentage:.2f}%", 
                ha='right', va='bottom', fontsize=9,fontname='Times New Roman',rotation=-70)
        else:
            plt.text(x[i], y[i], f"{percentage:.2f}%", 
                ha='left', va='bottom', fontsize=9,fontname='Times New Roman',rotation=45)
    
    plt.xlabel(x_label,fontsize=9,fontname='STSONG')
    plt.ylabel(y_label,fontsize=9,fontname='STSONG')
    # plt.title(title,fontsize=9,fontname='STSONG')
    plt.xticks(fontsize=9,fontname='Times New Roman')
    plt.yticks(fontsize=9,fontname='Times New Roman')
    # plt.grid(True)
    if rotation:
        # plt.xticks(ticks=x,rotation=rotation,ha='right')
        plt.xticks(ticks=x,rotation=rotation)
    if bottom_margin:
        plt.subplots_adjust(bottom=bottom_margin)
    plt.show()
    # plt.savefig(f"{title}.png")

def plot_bie(data,title):
    # plt.figure(figsize=(10,6))
    value = [i[1] for i in data]
    label = [i[0] for i in data]
    plt.pie(value,labels=label,autopct='%1.1f%%')
    # plt.title(title)
    plt.axis('equal')
    plt.show()
    # plt.savefig(f"{title}.png")

def plot_density(data,x_label,y_label,title):
    # 创建图形
    # plt.figure(figsize=(10, 6))
    sns.kdeplot(data=data, fill=True,color='#e1ccb1')
    
    # 设置标签和标题
    plt.xlabel(x_label,fontsize=9,fontname='STSONG')
    plt.ylabel(y_label,fontsize=9,fontname='STSONG')
    # plt.title(title,fontsize=9,fontname='STSONG')
    plt.xticks(fontsize=9,fontname='Times New Roman')
    plt.yticks(fontsize=9,fontname='Times New Roman')

    # 显示图形
    plt.show()
    # plt.savefig(f"{title}.png")
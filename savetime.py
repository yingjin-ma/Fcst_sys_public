import os

sourceFilePath = os.getcwd()
sourceFileName = sourceFilePath + "/P38_MPNN_631gss1_augV"# 定义要分割的文件


def cutFile ():
    print("正在读取文件...")
    sourceFileData = open(sourceFileName, 'r', encoding='utf-8')
    ListOfLine = sourceFileData.read().splitlines()  # 将读取的文件内容按行分割，然后存到一个列表中
    n = len(ListOfLine)
    print("文件共有" + str(n) + "行")
    dest = "../Fcst_sys_public/P38_MPNN_631gss1_augV"  # 定义分割后新生成的文件
    destData = open(dest, "w", encoding='utf-8')
    for line in ListOfLine:
        destData.write(line.split()[2] + '\n')
    print("分割完成")


cutFile()
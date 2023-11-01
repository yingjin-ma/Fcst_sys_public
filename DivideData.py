# coding:utf-8
# 将大文本文件分割成多个小文本文件
import os

sourceFilePath = os.getcwd() + "/database/rawdata/P38data.updated1"
sourceFileName = sourceFilePath + "/fcst_raw_P38_631gss_data2.dat"# 定义要分割的文件


def cutFile (ratio):
    print("正在读取文件...")
    sourceFileData = open(sourceFileName, 'r', encoding='utf-8')
    ListOfLine = sourceFileData.read().splitlines()  # 将读取的文件内容按行分割，然后存到一个列表中
    n = len(ListOfLine)
    print("文件共有" + str(n) + "行")
    train = ListOfLine[ : int(n * ratio)]
    valid = ListOfLine[int(n * ratio) : ]
    desttrain = sourceFilePath + "/train2.dat"  # 定义分割后新生成的文件
    desttrainData = open(desttrain, "w", encoding='utf-8')
    for line in train:
        desttrainData.write(line + '\n')
    desttvalid = sourceFilePath + "/valid2.dat"  # 定义分割后新生成的文件
    desttvalidData = open(desttvalid, "w", encoding='utf-8')
    for line in valid:
        desttvalidData.write(line + '\n')
    print("分割完成")


cutFile(0.75)
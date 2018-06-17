---
layout: pages
title: 《R语言与数据挖掘》 第一章 R语言的安装与使用
date: 2018-03-17 11:06:35
tags: 
    - R语言
    - 读书笔记
categories:
    - 读书笔记
---

# ubuntu下R语言环境搭建

R语言的安装

```bash
sudo apt-get install r-base
```

Rstudio安装
在[官网](https://www.rstudio.com/products/rstudio/download/)下载后，进入到下载页面，用dpkg命令安装

```bash
sudo dpkg -i rstudio-xenial-1.1.423-amd64.deb
```

# R 常用操作

## help

在命令窗口输入help(内容)，右下角就会显示需要查询的内容的具体描述

## Ctrl+L

清除命令窗口中的所有显示内容

## rm(list=ls())

清除R工作空间中的内存变量

一般利用rm(list=ls())和gc()来清除内存变量并释放内存空间

## install.packages、library

install.packages()用来下载和安装包
library()用于载入所下载的包

## getwd、setwd

获取或设置当前工作目录的位置

## save、load

save将工作空间中的指定对象保存到指定的文件中

load从磁盘文件中取一个工作空间到当前会话中

# R数据分析包

+ 分类与预测
	+ nnet()需要加载BP神经网络包nnet包
	+ randomForest()需要加载随机森林randomForest包
	+ svm()需要加载e1071包
	+ tree()需要加载CRAT决策树tree包
+ 聚类分析
	+ hclust()函数、kmeans()函数在stats包中
+ 关联分析
	+ apriori()需要加载arules包
+ 时间序列
	+ arima()需要加载forecast、tseries包
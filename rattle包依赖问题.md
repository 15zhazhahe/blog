---
title: 安装rattle包时遇到的依赖问题
date: 2018-04-03 19:57:02
tags: ['R语言']
categories: ['R语言']
---

安装rattle时,遇到依赖问题

```R
ERROR: dependencies ‘RGtk2’, ‘cairoDevice’, ‘XML’ are not available for package ‘rattle’
* removing ‘/home/richie/R/x86_64-pc-linux-gnu-library/3.2/rattle’
Warning in install.packages :
  installation of package ‘rattle’ had non-zero exit status
```

<!--more-->

安装cairoDevice,遇到依赖问题

```R
ERROR: gtk+2.0 not found by pkg-config.
ERROR: configuration failed for package ‘cairoDevice’
* removing ‘/home/richie/R/x86_64-pc-linux-gnu-library/3.2/cairoDevice’
Warning in install.packages :
  installation of package ‘cairoDevice’ had non-zero exit status
```

先在命令行执行如下命令,安装gtk包

```bash
sudo apt-get install libgtk2.0-dev
```
然后再次执行

```R
install.packages("cairoDevice")
```

安装XML时,也遇到了依赖问题

```R
ERROR: configuration failed for package ‘XML’
* removing ‘/home/richie/R/x86_64-pc-linux-gnu-library/3.2/XML’
Warning in install.packages :
  installation of package ‘XML’ had non-zero exit status
```

需要先在命令行执行如下命令

```bash
sudo apt-get install libxml2-dev
```

然后重新安装

```R
install.packages("XML")
```

安装rgtk2出现问题

```R
Warning in install.packages :
  package ‘rgkt2’ is not available (for R version 3.2.3)
```

R版本太低,重新装个R

安装最新版本的R,先添加一个源

```bash
vim /etc/apt/sources.list
```

然后在末尾增加一行

```bash
deb https://cloud.r-project.org//bin/linux/ubuntu xenial/
```

添加CRAN存储密钥,更新软件源
CRAN中存储的Ubuntu包需要通过密钥E084DAB9进行签名验证

```bash
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9
```

然后更新软件源

```bash
sudo apt-get update
```

然后安装R基本版与开发版

```bash
sudo apt-get install r-base
sudo apt-get install r-base-dev
```

最后在安装rattle就成功了

```R
install.packages("rattle")
```

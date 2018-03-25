---
layout: pages
title: 《C++ Primer Plus 第六版》 第2章 开始学习C++
date: 2018-03-10 22:02:57
tags: 
    - C++
    - 读书笔记
categories:
    - 读书笔记
---

> 对基础语法的大致框架做了简要的介绍

<!-- more -->

# 进入C++

```c++
/*
多行注释
所有语句用 ; 分号结尾
大部分情况下,C++程序必须含有一个名为main(大小写敏感)的函数 作为程序的入口
*/
                                    // 单行注释
#include <iostream>                 // 预处理器编译指令#include
int main()                          // 函数头
{                                   // 函数体 {}
    using namespace std;            // 遍历指令
    cout << "Come up and C++ me some time.";
    cout << endl;
    cout<< "You won't regret it!" << endl;
    return 0;                       // 结束函数的 return 语句
}
```

## C++ 预处理器

C++ 有一个预处理器(这里只讨论处理名词以 # 开头的编译指令),该程序在进行主编译前对源文件进行处理

include预处理指令的作用是在指令处展开被包含的文件。包含可以是多重的，也就是说一个被包含的文件中还可以包含其他文件。预处理器至多支持15层嵌套包含。

预处理过程不检查在转换单元中是否已经包含了某个文件并阻止对它的多次包含,所以这样就很容易发生相同的一个文件被包含多次的情况，但可以使用条件式编译的预处理指令避免多次包含相同的文件

## 头文件名

C++ 对老式 C 的头文件保留了扩展名，而 C++ 头文件则没有扩展名。有些 C 头文件被转换为 C++头文件，只是这些文件去掉了 .h，并加上了前缀 c (表明来自 C 语言)。例如，C++版本的 math.h 为 cmath

## 名称空间

使用不是.h结尾的头文件时，应用下面的名称空间编译指令来使这个头文件中的定义对程序可用

```c++
using namespace std;    // using 编译指令
```

名称空间支持是 C++ 的一项特性， 它能使得在编写大型程序时更为方便，且有助于组织程序。

# C++语句

C++ 程序是一组函数，每个函数有事一组语句。

## 声明语句

要将信息项存储在计算机中，必须指出信息的存储位置和所需的内存空间，而在C++中就是采用声明语句来指出**存储类型**并提供**位置标签**的，

## 赋值语句

赋值语句将值赋给存储单元。

```c++
carrots = 25；   // = 表示赋值运算符
```

# 类简介

类是用户定义的一种数据类型。类的定义包括**表示的信息**和**数据执行的操作**。类之于对象，就像类型之于变量。话句话说，类定义描述的就是数据格式及其用法，而对象则是根据数据格式规范创建的实体。

类描述指定了可对类对象执行的所有操作。对这些特定对象执行这些操作可以采用两种方法：调用类方法、重载运算符

# 函数

```c++
#include <iostream>
using namespace std;
/*
函数格式
返回类型 函数名(形式参数列表)
{
    函数体
}
*/
void function(int);   // 函数原型
int main(void)
{
    ...
    return 0;
}
void function(int n)
{
    ...
}
```

对于 main 函数来说，可以将操作系统看做调用程序。因此，main() 的返回值并不是返回给程序的其他部分，而是返回给操作系统。
例如，UNIX 外壳脚本和 Windows 命令行批处理文件都被设计成运行程序，并测试他们的返回值（退出值），通常约定退出值为 0 表示运行程序运行成功，为非零表示存在问题

# 复习题

## 1. C++ 程序的模块叫什么？

函数

## 2. 下面的预处理器编译指令是做什么用的？

```c++
#include <iostream>
```

在最终编译之前，使用iostream中的内容替换该编译指令

## 3. 下面的语句是做什么用的

```c++
using namespace std;
```

使程序可以使用std名称空间中的定义

## 4. 什么语句可以用来打印短语 "Hello, world"，然后开始新的一行？

```c++
cout <<"Hello,world"<<endl;
printf("Hello,world\n");
```

## 5. 什么语句可以用来创建名为 cheeses 的整数变量？

```c++
int cheeses;
```

## 6. 什么语句可以用来将值 32 赋给变量 cheeses？

```c++
cheeses = 32
```

## 7. 什么语句可以来将从键盘输入的值读入变量 cheeses 中？

```c++
cin >> cheeses;
scanf("%d",&cheeses);
```

## 8. 什么语句可以用来打印 "We have X varieties of cheese."， 其中X为变量cheeses的当前值

```c++

cout<<"We have "<<cheeses<<" varieties of cheese."<<endl;
printf("We have %d varieties of cheese.", cheeses);
```

## 9. 下面的函数原型指出了关于函数的哪些信息？

```c++
// 调用该函数时，应提供一个类型为double的参数，该函数会返回一个类型为int的值
int froop(double t);
// 调用该函数时，需要提供一个类型为int的参数，该函数没有返回值
void rattle(int n);
// 调用该函数时，不用提供参数，该函数会返回一个类型为int的值
int prune(void);
```

## 10. 定义函数时，在什么情况下不必使用关键之 return ？

当函数的返回类型为 void 时

## 11. 假设您编写的 main() 函数包含如下代码，而编译器指出cout是一个未知标识符。导致这种问题的原因很可能是什么？指出3种修复这种问题的办法

```c++
cout << "Please enter your PIN: ";
```

没有引用iostream头文件（引用即可）

没有指明命名空间（使用using namespace std 或者使用 std::cout）

# 编程练习

编程练习代码在 [github](https://github.com/15zhazhahe/reading-notes/tree/master/CPP-Primer-Plus/Chapter-02) 上
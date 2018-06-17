# C/C++ 内存管理

占坑

http://blog.csdn.net/wdzxl198/article/details/9050587

# C/C++ 内存分配

## malloc/free 函数

```C++
//声明
void *malloc(unsigned int size);
void *free(void *memblock);

//使用
int *p = (int*)malloc(sizeof(int) * length);
free(p);
```

在内存的动态分配区域中分配一个长度为 size 的连续空间，如果分配成功，则返回所分配内存空间的首地址，否则返回NULL，申请的内存不会进行初始化。

因为指针p之前的类型以及它所指的内存的容量事先知道，所以语句 free(p) 能正确的释放内存。如果p是NULL指针，那么对p无论操作多少次都不会出现问题，如果p不是NULL指针，那么free对p连续操作两次就会导致程序运行错误。

## calloc 函数

```C++
void *calloc(unsigned int num,
			 ungisned int size)
```

按照所给的数据个数和数据类型所占字节数，分配一个大小为 num * size 的连续空间

**与malloc不同的是：**calloc 申请内存空间后，会自动初始化内存空间为0，但是 malloc 不会进行初始化，其内存空间存储的是一些随机数据

## realloc函数

```c++
void *realloc(void *ptr, 
			  unsigned int size)
```

动态分配一个长度为 size 的内存空间，并把内存空间的首地址赋值给 ptr，把ptr内存空间调整为size。

申请的内存空间不会进行初始化

## new/delete 运算符

```c++
//使用
int *p = new int[length];
delete []objects; //释放数组的内存
delete objects; //释放对象内存
```

new 是动态分配内存的运算符，自动计算需要分配的空间，在分配类的内存空间时，会同时调用类的构造函数，对内存空间进行初始化，即完成类的初始化工作。

动态分配内置类型是否自动初始化取决于变量定义的位置，在函数体外（全局变量？）定义的变量都初始化为0，在函数体内定义的内置类型变量都不进行初始化。


## malloc/free 和 new/delete 的区别

+ **相同点：** 都可用于申请动态内存和释放内存
+ **不同点**
	+ new 是操作符，malloc是函数
	+ new 是保留字，malloc需要头文件库函数支持
	+ new 出来的指针是直接带类型信息的。malloc返回的是 void 指针，malloc不会识别要申请的内存是什么类型，只关心内存的总字节数。
	+ new 不单单是分配内存，而且还会调用类的构造函数，同理delete会调用类的析构函数，而malloc则只分配内存，不会进行初始化类成员工作，同理 free不会调用析构函数
	+ 内存泄露对于 malloc 或者 new 都可以检查出来的，区别再于new可以指明是那个文件的哪一行，而malloc没有这些信息


https://blog.csdn.net/hackbuteer1/article/details/6789164

https://www.cnblogs.com/jiayouwyhit/p/3753909.html

https://blog.csdn.net/wdzxl198/article/details/9050587

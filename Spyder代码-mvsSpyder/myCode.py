#!/usr/bin/env python
# coding: utf-8

### 类型定义及输出

#Python语句前加#表示注释，不参与运算 
X=12.34              #定义X为实数
print(X)             #无格式输出
print('X=%5.4f'%X)   #有格式输出

### 加载包及运算 

import math          #加载数学运算包
Y=1+2-3*4/5+6**2+math.exp(7)+math.log(8); print(Y)

from math import *  #加载数学运算函数
Z=1+2-3*4/5+6**2+exp(7)+log(8); print('Z=%8.4f'%Z)

### 基本绘图

x=[1,2,3,4,5];
y=[1,4,9,16,25];  #y=x^2
import matplotlib.pyplot as plt #加载基本绘图包 
plt.plot(x,y,'o:');

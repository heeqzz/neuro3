# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 09:53:17 2025

@author: 1
"""
import random
import torch
x=torch.zeros(1,3,dtype=torch.int)
print(x)
x = torch.randint(1,10,(1,3))
print(x)
 
x =x.to(dtype=torch.float32)
x.requires_grad=True 
print(x)

y=x**2 #вариант1-нечетный, поэтому возводим в 2 степень 
#print(y)
b=random.randint(1,10)
#print("b =",b)
z=y*b
#print(z)
w=torch.exp(z)
#print(w)
out=w.mean()
out.backward()  # Вычисляем градиент
print(x.grad)

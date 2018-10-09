# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 19:32:40 2018
@author: hamed
"""

def swap(A, i, j):
    assert i>-1 and j>-1 and i!=j
    assert i<len(A) and j<len(A)
    temp=A[i]
    A[i]=A[j]
    A[j]=temp
    return A

def area(A, i,j):
    return min(A[i], A[j])* abs(i-j)
    
def max_area(A):
    n=len(A)
    if n==1:
        return 0
    elif n==2:
        return area(A,0,1)
    i = 0
    j = n-1
    max_sofar=area(A,i,j)

    while i<n-1 and j>0:
        right=area(A,i+1,j)
        left=area(A,i,j-1)
        
        if right>=left:
            if right>max_sofar:
                max_sofar=right
                i+=1
        else:
            if left>max_sofar:
                max_sofar=left
                j-=1
    return max_sofar
            
                
            
            
        
    
        
        
       
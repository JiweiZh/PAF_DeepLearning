import random as rd
import numpy as np
#import matplotlib.pyplot as plt

def graylevels_to_bandw(im,dec=0):
    return np.round(im+dec)

def new_resolution(im,n):
    if n==0:
        return im
    return np.round(im*(2**n-1))/(2**n-1)

def change_pix():
    inc=np.random.random()
    return inc

def change_pix2(n):
    inc=np.random.random()/2+0.25
    sign=np.round(np.random.random())*2-1
    return (n+sign*inc)%1

def invert_pix(im,i):
    im[i]=1-im[i]

"""
lam=param de la loi de Poisson qui donne le nombre de pixels erronnes
On tire au sort un nombre number_of_errors de pixels a changer selon une loi de Poisson de parametre lam
On tire au sort number_of_errors numeros representant les pixels a changer : ils sont stockes dans le vecteur pix_to_change
on change ces pixels
"""

def error1(tab,lam):
    number_of_errors = min(784,np.random.poisson(lam))
    pix_to_change = np.random.choice(784,number_of_errors,replace=False)
    res = np.copy(tab)
    for i in pix_to_change:
        res[i]=change_pix()
    return res

def error2(tab,number_of_errors):
    pix_to_change = np.random.choice(784,min(784,number_of_errors),replace=False)
    res = np.copy(tab)
    for i in pix_to_change:
        res[i]=change_pix()
    return res

#def binary_error2(tab,p):
    #p=proba d'erreur pour chaque bit

def binary_error1(tab):
    number_of_errors = np.random.poisson(lam)
    pix_to_change = np.random.choice(784,min(784,number_of_errors),replace=False)
    res = np.copy(tab)
    for i in pix_to_change:
        invert_pix(res,i)
    return res

def lines_to_delete(l):
    lines_to_delete = []
    for i in range(28):
        for j in range(l):
            lines_to_delete.append(28*i+j)
            lines_to_delete.append(28*j+i)
            lines_to_delete.append(783-28*j-i)
            lines_to_delete.append(28*i+27-j)
    return lines_to_delete

def cut_all_images(list_of_images,lines_to_delete):
    return np.delete(list_of_images,lines_to_delete,0)

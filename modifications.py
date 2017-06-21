import random as rd
import numpy as np
#import matplotlib.pyplot as plt

def graylevels_to_bandw(im,dec=0):
    return np.round(im+dec)

def change_pix(n):
    inc=np.random.random()/2+0.25
    return (n+inc)%1

def invert_pix(im,i):
    im[i]=1-im[i]

def error1(tab,lam):
    #lam=param de la loi de Poisson qui donne le nombre de pixels erronnés
    #On tire au sort un nombre number_of_errors de pixels à changer selon une loi de Poisson de parametre lam
    #On tire au sort number_of_errors numéros représentant les pixels à changer : ils sont stockés dans le vecteur pix_to_change
    #on change ces pixels
    number_of_errors = np.random.poisson(lam)
    pix_to_change = np.random.choice(784,min(784,number_of_errors),replace=False)
    res = np.copy(tab)
    for i in pix_to_change:
        res[i]=change_pix(res[i])
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

# Numerical Simulations for Exploring Van der Pol Oscillator

# Chris Fritz 1/15/2020

##import statements
import numpy as np
import matplotlib.pyplot as plt

## van der pol eqs
# dx/dt = y
# dy/dt = mu * (1-x**2) * y  - x

def y_dot(x, y, mu):
    return mu * (1-x*x) * y - x

def x_dot(y):
    return y

## consider simulation OO framework: fast simulatable object that you provide difeqs for

## parameters
mu = 1     # damping
x0 = .5    # init x
y0 = .5    # init y

## run simulation


## plot data

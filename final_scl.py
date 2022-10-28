
#taylor method
from sympy import *
x, y = symbols('x y')

def fact(n):
    factorial = 1
    for i in range(1, n + 1):
        factorial *= i
    return factorial

def taylor_series(y_diff, x0, y0, h):
    sum = 0
    for i in range(1, 5):
        sum += (h ** i) * y_diff.subs({x : x0, y : y0}) / fact(i)
        y_diff = diff(y_diff, x)
    return sum

x0 = 0
y0 = 1
h = 0.1
Y = 0.2
y_dash = x ** 2 * y - 1

for _ in range(5):
    y0 = y0 + taylor_series(y_dash, x0, y0, h)
    print(f"f({x0 + h:2.1f}) = {y0}")
    x0 += h

#adambashforth

import numpy as np

def RK4_step(f,y,t,dt, N=1):
    dt /= N;
    for k in range(N):
        k1=f(y,t)*dt; k2=f(y+k1/2,t+dt/2)*dt; k3=f(y+k2/2,t+dt/2)*dt; k4=f(y+k3,t+dt)*dt;
        y, t = y+(k1+2*(k2+k3)+k4)/6, t+dt        
    return y

def Adams_Moulton_4th(function, y_matrix, time):
    y = np.zeros((np.size(time), np.size(y_matrix)))
    y[0] = y_matrix
### bootstrap steps with 4th order one-step method
    dt = time[1] - time[0]
    y[1] = RK4_step(function,y[0], time[0], dt, N=4)
    y[2] = RK4_step(function,y[1], time[1], dt, N=4)
    y[3] = RK4_step(function,y[2], time[2], dt, N=4)

    f_m2 = function(y[0], time[0])
    f_m1 = function(y[1], time[1])
    f_0 = function(y[2], time[2])
    f_1 = function(y[3], time[3])
    for i in range(3,len(time) - 1):
    ### first shift the "virtual" function value array so that
    ### [f_m3, f_m2, f_m1, f_0] corresponds to [ f[i-3], f[i-2], f[i-1], f[i] ]
        f_m3, f_m2, f_m1, f_0 = f_m2, f_m1, f_0, f_1
    ### predictor formula 4th order [ 55/24, -59/24, 37/24, -3/8 ]
        y[i+1] = y[i] + (dt/24) * (55*f_0 - 59*f_m1 + 37*f_m2 - 9*f_m3)
        f_1 = function(y[i+1], time[i+1])
    ### Corrector formula 4th order [ 3/8, 19/24, -5/24, 1/24 ]
        y[i+1] = y[i] + (dt/24) * (9*f_1 + 19*f_0 - 5*f_m1 + f_m2)
        f_1 = function(y[i+1], time[i+1])
    return y
    
    
#euler
import numpy as np

# Define parameters
f = lambda t, s: np.exp(-t) # ODE
h = 0.1 # Step size
t = np.arange(0, 1 + h, h) # Numerical grid
s0 = -1 # Initial Condition

# Explicit Euler Method
s = np.zeros(len(t))
s[0] = s0

for i in range(0, len(t) - 1):
    s[i + 1] = s[i] + h*f(t[i], s[i])
    
print(s)

#modifiedeuler
import numpy as np

# Define parameters
f = lambda t, s: np.exp(-t) # ODE
h = 0.1 # Step size
t = np.arange(0, 1 + h, h) # Numerical grid
s0 = -1 # Initial Condition

s = np.zeros(len(t))
sp= np.zeros(len(t))
s[0] = s0
sp[0] = s0

for i in range(0, len(t) - 1):
    s[i + 1] = s[i] + h*f(t[i], s[i])
    sp[i+1]= s[i] + h*0.5*(f(t[i],s[i]) + f(t[i+1], s[i+1]))

print(sp)

#predictorcorrector
def f(x, y):
    v = y - 2 * x * x + 1;
    return v;
 
# predicts the next value for a given (x, y)
# and step size h using Euler method
def predict(x, y, h):
     
    # value of next y(predicted) is returned
    y1p = y + h * f(x, y);
    return y1p;
 
# corrects the predicted value
# using Modified Euler method
def correct(x, y, x1, y1, h):
     
    # (x, y) are of previous step
    # and x1 is the increased x for next step
    # and y1 is predicted y for next step
    e = 0.00001;
    y1c = y1;
 
    while (abs(y1c - y1) > e + 1):
        y1 = y1c;
        y1c = y + 0.5 * h * (f(x, y) + f(x1, y1));
 
    # every iteration is correcting the value
    # of y using average slope
    return y1c;
 
def printFinalValues(x, xn, y, h):
    while (x < xn):
        x1 = x + h;
        y1p = predict(x, y, h);
        y1c = correct(x, y, x1, y1p, h);
        x = x1;
        y = y1c;
 
    # at every iteration first the value
    # of for next step is first predicted
    # and then corrected.
    print("The final value of y at x =",
                     int(x), "is :", y);
 
# Driver Code
if __name__ == '__main__':
     
    # here x and y are the initial
    # given condition, so x=0 and y=0.5
    x = 0; y = 0.5;
 
    # final value of x for which y is needed
    xn = 1;
 
    # step size
    h = 0.2;
 
    printFinalValues(x, xn, y, h);
    
 #rugnekutta
 import numpy as np

def dydt(t, y): 
    return np.sin(t)**2*y

def f(t): return 2*np.exp(0.5*(t-np.sin(t)*np.cos(t)))


def RK3(t, y, h):
    # compute approximations
    k_1 = dydt(t, y)
    k_2 = dydt(t+h/2, y+(h/2)*k_1)
    k_3 = dydt(t+h/2, y+h*(-k_1 + 2*k_2))

    # calculate new y estimate
    y = y + h * (1/6) * (k_1 + 4 * k_2 + k_3)
    return y


def RK4(t, y, h):
    # compute approximations
    k_1 = dydt(t, y)
    k_2 = dydt(t+h/2, y+(h/2)*k_1)
    k_3 = dydt(t+h/2, y+(h/2)*k_2)
    k_4 = dydt(t+h, y+h*k_3)

    # calculate new y estimate
    y = y + h * (1/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    return y


# Initialization
ti = 0
tf = 5
h = 0.5
n = int((tf-ti)/h)
t = 0
y = 2
y_rk3 = 2

print("t \t\t yRK3 \t\t yRK4 \t\t f(t)")
print(f"{t:.1f} \t\t {y_rk3:4f} \t\t {y:4f} \t\t {f(t):.4f}")

t_plot = []
y_RK3 = []
y_RK4 = []
y_analytical = []


##
for i in range(1, n+1):
    t_plot.append(t)
    y_RK4.append(y)
    y_RK3.append(y_rk3)
    y_analytical.append(f(t))

    # calculate new y estimate
    y = RK4(t, y, h)
    y_rk3 = RK3(t, y_rk3, h)

    t += h
    print(f"{t:.1f} \t\t {y_rk3:4f} \t\t {y:4f} \t\t {f(t):.4f}")

# 设计EKF（卡尔曼滤波器）系统用并`C++`实现

该项目在项目[1](https://github.com/chendaxiashizhu/Kalman)提供的算法与项目[2](https://github.com/karanchawla/GPS_IMU_Kalman_Filter)提供的C++代码上修改完成。

## 算法模型

飞机模型如下：
$$
x_{k}=\left[\begin{array}{c}x \\ y \\ \dot{x} \\ \dot{y} \\ \ddot{x} \\ \ddot{y}\end{array}\right]
$$
运动方程为：
$$
x_{k+1}=A \cdot x_{k}+B \cdot u
$$
详细方程为：
$$
\begin{aligned} x_{k+1}=& x_{k}+\dot{x}_{k} \cdot \Delta t+\ddot{x}_{k} \cdot \frac{1}{2} \Delta t^{2} \\ y_{k+1}=& y_{k}+\dot{y}_{k} \cdot \Delta t+\ddot{y}_{k} \cdot \frac{1}{2} \Delta t^{2} \\ \dot{x}_{k+1}=& \dot{x}_{k}+\ddot{x} \cdot \Delta t \\ \dot{y}_{k+1}=& \dot{y}_{k}+\ddot{y} \cdot \Delta t \\ \ddot{x}_{k+1}=& \ddot{x}_{k} \\ \ddot{y}_{k+1}=& \ddot{y}_{k} \end{aligned}
$$
这里是没有输入量u：
$$
x_{k+1}=\left[\begin{array}{cccccc}1 & 0 & \Delta t & 0 & \frac{1}{2} \Delta t^{2} & 0 \\ 0 & 1 & 0 & \Delta t & 0 & \frac{1}{2} \Delta t^{2} \\ 0 & 0 & 1 & 0 & \Delta t & 0 \\ 0 & 0 & 0 & 1 & 0 & \Delta t \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1\end{array}\right]\left[\begin{array}{c}x \\ y \\ \dot{x} \\ \dot{y} \\ \ddot{x} \\ \ddot{y}\end{array}\right]
$$
测量方程：
$$
y=\left[\begin{array}{cccccc}1 & 0 & 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 & 0 & 1\end{array}\right] \cdot x
$$
确认过程噪声Q矩阵：
$$
Q=G \cdot G^{T} \cdot \sigma_{a}^{2}
$$
![DraggedImage.png](https://imgconvert.csdnimg.cn/aHR0cDovL3E3bXEweDFjeS5ia3QuY2xvdWRkbi5jb20vdHAyMDIwMzI0MjEzODI5X0RyYWdnZWRJbWFnZS5wbmc?x-oss-process=image/format,png)
确认测量噪声R矩阵：

```python
ra = 10.0**2   # Noise of Acceleration Measurement
rp = 100.0**2  # Noise of Position Measurement
R = np.matrix([[rp, 0.0, 0.0, 0.0],
               [0.0, rp, 0.0, 0.0],
               [0.0, 0.0, ra, 0.0],
               [0.0, 0.0, 0.0, ra]])
print(R, R.shape)
```

## 算法仿真

```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy.stats import norm
x = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
print(x, x.shape)
n=x.size # States

P = np.diag([100.0, 100.0, 10.0, 10.0, 1.0, 1.0])
dt = 0.1 # Time Step between Filter Steps

A = np.matrix([[1.0, 0.0, dt, 0.0, 1/2.0*dt**2, 0.0],
              [0.0, 1.0, 0.0, dt, 0.0, 1/2.0*dt**2],
              [0.0, 0.0, 1.0, 0.0, dt, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, dt],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
H = np.matrix([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

ra = 10.0**2   # Noise of Acceleration Measurement
rp = 100.0**2  # Noise of Position Measurement
R = np.matrix([[rp, 0.0, 0.0, 0.0],
               [0.0, rp, 0.0, 0.0],
               [0.0, 0.0, ra, 0.0],
               [0.0, 0.0, 0.0, ra]])
from sympy import Symbol, Matrix
from sympy.interactive import printing
printing.init_printing()
dts = Symbol('\Delta t')
Qs = Matrix([[0.5*dts**2],[0.5*dts**2],[dts],[dts],[1.0],[1.0]])

sa = 0.001
G = np.matrix([[1/2.0*dt**2],
               [1/2.0*dt**2],
               [dt],
               [dt],
               [1.0],
               [1.0]])
Q = G*G.T*sa**2
I = np.eye(n)
m = 500 # Measurements

sp= 1.0 # Sigma for position
px= 0.0 # x Position
py= 0.0 # y Position

# mpx = np.array(px+sp*np.random.randn(m))
# mpy = np.array(py+sp*np.random.randn(m))

# Generate GPS Trigger
GPS=np.ndarray(m,dtype='bool')
GPS[0]=True
# Less new position updates
for i in range(1,m):
    if i%10==0:
        GPS[i]=True
    else:
        GPS[i]=False
        
# Acceleration
sa= 0.1 # Sigma for acceleration
ax= 0.0 # in X
ay= 0.0 # in Y
import numpy as np
data = np.loadtxt("data1.txt")
mpx = data[:,0]
mpy = data[:,1]
mx = data[:,2]
my = data[:,3]
measurements = np.vstack((mpx,mpy,mx,my))
# Preallocation for Plotting
xt = []
yt = []
dxt= []
dyt= []
ddxt=[]
ddyt=[]
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Pddx=[]
Pddy=[]
Kx = []
Ky = []
Kdx= []
Kdy= []
Kddx=[]
Kddy=[]


def savestates(x, Z, P, K):
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    ddxt.append(float(x[4]))
    ddyt.append(float(x[5]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Pddx.append(float(P[4,4]))
    Pddy.append(float(P[5,5]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))
    Kddx.append(float(K[4,0]))
    Kddy.append(float(K[5,0]))
for filterstep in range(m):
    
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = A*x
    
    # Project the error covariance ahead
    P = A*P*A.T + Q    
    
    
    # Measurement Update (Correction)
    # ===============================
    # if there is a GPS Measurement
#     就这样简单的处理
    if GPS[filterstep]:
        # Compute the Kalman Gain
        S = H*P*H.T + R
        K = (P*H.T) * np.linalg.pinv(S)
    
        
        # Update the estimate via z
        Z = measurements[:,filterstep].reshape(H.shape[0],1)
        y = Z - (H*x)                            # Innovation or Residual
        x = x + (K*y)
        
        # Update the error covariance
        P = (I - (K*H))*P
    
    # Save states for Plotting
    savestates(x, Z, P, K)
def plot_x():
    
    fig = plt.figure(figsize=(16,16))

    plt.subplot(311)
    plt.step(range(len(measurements[0])),ddxt, label='$\ddot x$')
    plt.step(range(len(measurements[0])),ddyt, label='$\ddot y$')

    plt.title('Estimate (Elements from State Vector $x$)')
    plt.legend(loc='best',prop={'size':22})
    plt.ylabel(r'Acceleration $m/s^2$')
    plt.ylim([-.1,.1])

    plt.subplot(312)
    plt.step(range(len(measurements[0])),dxt, label='$\dot x$')
    plt.step(range(len(measurements[0])),dyt, label='$\dot y$')

    plt.ylabel('')
    plt.legend(loc='best',prop={'size':22})
    plt.ylabel(r'Velocity $m/s$')
    plt.ylim([-1,1])

    plt.subplot(313)
    plt.step(range(len(measurements[0])),xt, label='$x$')
    plt.step(range(len(measurements[0])),yt, label='$y$')

    plt.xlabel('Filter Step')
    plt.ylabel('')
    plt.legend(loc='best',prop={'size':22})
    plt.ylabel(r'Position $m$')
    plt.ylim([-1,1])

    plt.savefig('Kalman-Filter-CA-StateEstimated.png', dpi=72, transparent=True, bbox_inches='tight')
plot_x()
```

最终的结果如下：

![DraggedImage-1.png](https://imgconvert.csdnimg.cn/aHR0cDovL3E3bXEweDFjeS5ia3QuY2xvdWRkbi5jb20vdHAyMDIwMzI0MjEzODI5X0RyYWdnZWRJbWFnZS0xLnBuZw?x-oss-process=image/format,png#pic_center =400x)
![DraggedImage-2.png](https://imgconvert.csdnimg.cn/aHR0cDovL3E3bXEweDFjeS5ia3QuY2xvdWRkbi5jb20vdHAyMDIwMzI0MjEzODI5X0RyYWdnZWRJbWFnZS0yLnBuZw?x-oss-process=image)

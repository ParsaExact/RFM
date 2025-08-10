# -*- coding: utf-8 -*-

import time
import sympy as sym
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
torch.set_default_dtype(torch.float64)

# function to fix random seed
def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True

# parameter initialization
rand_mag = 1.0
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -rand_mag, b = rand_mag)
        nn.init.uniform_(m.bias, a = -rand_mag, b = rand_mag)

# RFM network
class RFM(nn.Module):
    def __init__(self, input_dim, J_n, x_max, x_min, t_max, t_min):
        super(RFM, self).__init__()
        self.input_dim = input_dim
        self.output_dim = J_n
        self.r_n_1 = torch.tensor([2.0/(x_max - x_min),2.0/(t_max - t_min)])
        self.x_n = torch.tensor([(x_max + x_min)/2,(t_max + t_min)/2])
        self.layer = nn.Sequential(nn.Linear(self.input_dim, self.output_dim, bias=True),nn.Tanh())
    
    def forward(self,x):
        x = self.r_n_1 * (x - self.x_n)
        x = self.layer(x)
        return x

L = 10
D = 10

x_l = 0
x_r = L
y_d = -D/2
y_u = D/2

E = 3.0e7
mu = 0.3
P = 1.0e3
I = D**3/12

a = E / (1 - mu**2)
b = (1 - mu) / 2
c = (1 + mu) / 2
epsilon = 1.0e-24

def u(x,y):
    u_ = -P*y/(6*E*I) * ((6*L - 3*x)*x + (2+mu)*(y**2 - D**2/4))
    return u_


def v(x,y):
    v_ = P/(6*E*I) * (3*mu*y**2*(L - x) + (4 + 5*mu)*D**2*x/4 + (3*L-x)*x**2)
    return v_

def bx(x0,y0):
    x, y = sym.symbols("x y")
    u_ = -P*y/(6*E*I) * ((6*L-3*x)*x + (2+mu)*(y**2 - D**2/4))
    v_ = P/(6*E*I) * (3*mu*y**2*(L-x) + (4+5*mu)*D**2*x/4 + (3*L-x)*x**2)

    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    uxx = sym.diff(ux, x)
    uxy = sym.diff(ux, y)
    uyy = sym.diff(uy, y)

    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    vxx = sym.diff(vx, x)
    vxy = sym.diff(vx, y)
    vyy = sym.diff(vy, y)
    
    # 体力
    bx_ = -a * (uxx + b * uyy + c * vxy)
    bx = sym.lambdify((x, y), bx_, "numpy")
    return bx(x0,y0)


def by(x0,y0):
    x, y = sym.symbols("x y")
    u_ = -P*y/(6*E*I) * ((6*L-3*x)*x + (2+mu)*(y**2 - D**2/4))
    v_ = P/(6*E*I) * (3*mu*y**2*(L-x) + (4+5*mu)*D**2*x/4 + (3*L-x)*x**2)

    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    uxx = sym.diff(ux, x)
    uxy = sym.diff(ux, y)
    uyy = sym.diff(uy, y)

    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    vxx = sym.diff(vx, x)
    vxy = sym.diff(vx, y)
    vyy = sym.diff(vy, y)
    # 体力
    by_ = -a * (vyy + b * vxx + c * uxy)
    by = sym.lambdify((x, y), by_, "numpy")
    return by(x0,y0)


def px(x0,y0,nx,ny):
    x, y = sym.symbols("x y")
    u_ = -P*y/(6*E*I) * ((6*L-3*x)*x + (2+mu)*(y**2 - D**2/4))
    v_ = P/(6*E*I) * (3*mu*y**2*(L-x) + (4+5*mu)*D**2*x/4 + (3*L-x)*x**2)
    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    px = a * (nx * (ux + mu * vy) + ny * b * (uy + vx))
    px = sym.lambdify((x, y), px, "numpy")
    return px(x0,y0)


def py(x0,y0,nx,ny):
    x, y = sym.symbols("x y")
    u_ = -P*y/(6*E*I) * ((6*L-3*x)*x + (2+mu)*(y**2 - D**2/4))
    v_ = P/(6*E*I) * (3*mu*y**2*(L-x) + (4+5*mu)*D**2*x/4 + (3*L-x)*x**2)
    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    py = a * (ny * (vy + mu * ux) + nx * b * (vx + uy))
    py = sym.lambdify((x, y), py, "numpy")
    return py(x0,y0)


def stress_sigma_x(x0,y0):
    x, y = sym.symbols("x y")
    u_ = -P*y/(6*E*I) * ((6*L-3*x)*x + (2+mu)*(y**2 - D**2/4))
    v_ = P/(6*E*I) * (3*mu*y**2*(L-x) + (4+5*mu)*D**2*x/4 + (3*L-x)*x**2)
    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    sigma_x = a*(ux + mu * vy)
    sigma_x = sym.lambdify((x, y), sigma_x, "numpy")
    return sigma_x(x0,y0)


def stress_sigma_y(x0,y0):
    x, y = sym.symbols("x y")
    u_ = -P*y/(6*E*I) * ((6*L-3*x)*x + (2+mu)*(y**2 - D**2/4))
    v_ = P/(6*E*I) * (3*mu*y**2*(L-x) + (4+5*mu)*D**2*x/4 + (3*L-x)*x**2)
    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    sigma_y = a*(vy + mu * ux)
    sigma_y = sym.lambdify((x, y), sigma_y, "numpy")
    return sigma_y(x0,y0)


def stress_tau_xy(x0,y0):
    x, y = sym.symbols("x y")
    u_ = -P*y/(6*E*I) * ((6*L-3*x)*x + (2+mu)*(y**2 - D**2/4))
    v_ = P/(6*E*I) * (3*mu*y**2*(L-x) + (4+5*mu)*D**2*x/4 + (3*L-x)*x**2)
    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    tau_xy = a * b * (uy + vx)
    tau_xy = sym.lambdify((x, y), tau_xy, "numpy")
    return tau_xy(x0,y0)


vanal_u = np.vectorize(u)
vanal_v = np.vectorize(v)
vanal_bx = np.vectorize(bx)
vanal_by = np.vectorize(by)
vanal_px = np.vectorize(px)
vanal_py = np.vectorize(py)
vanal_sigma_x = np.vectorize(stress_sigma_x)
vanal_sigma_y = np.vectorize(stress_sigma_y)
vanal_tau_xy = np.vectorize(stress_tau_xy)

def Pre_Definition(Nx,Ny,J_n,Qx,Qy):
    models = []
    points = []
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = (x_r - x_l)/Nx * k + x_l
        x_max = (x_r - x_l)/Nx * (k+1) + x_l
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Ny):
            t_min = (y_u - y_d)/Ny * n + y_d
            t_max = (y_u - y_d)/Ny * (n+1) + y_d
            model_u = RFM(input_dim = 2, J_n = J_n, x_min = x_min, x_max = x_max, t_min = t_min, t_max = t_max)
            model_v = RFM(input_dim = 2, J_n = J_n, x_min = x_min, x_max = x_max, t_min = t_min, t_max = t_max)
            model_u = model_u.apply(weights_init)
            model_v = model_v.apply(weights_init)
            model_u = model_u.double()
            model_v = model_v.double()
            for param in model_u.parameters():
                param.requires_grad = False
            for param in model_v.parameters():
                param.requires_grad = False
            model_for_x.append([model_u,model_v])
            t_devide = np.linspace(t_min, t_max, Qy + 1)
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(Qx+1,Qy+1,2)
            point_for_x.append(torch.tensor(grid,requires_grad=True))
        models.append(model_for_x)
        points.append(point_for_x)
    return(models,points)


def Matrix_Assembly(models,points,Nx,Ny,J_n,Qx,Qy,inputs = None):
    # matrix define (Aw=b)
    A_u_bx = None
    A_v_bx = None
    A_u_by = None
    A_v_by = None
    f_bx = None
    f_by = None
    
    
    B_line_1_u = np.zeros([Ny*Qy,Nx*Ny*J_n])
    B_line_1_v = np.zeros([Ny*Qy,Nx*Ny*J_n])
    p_line_1_u = np.zeros([Ny*Qy,1])
    p_line_1_v = np.zeros([Ny*Qy,1])
    
    B_line_2_u_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_2_v_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_2_x = np.zeros([Nx*Qx,1])
    B_line_2_u_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_2_v_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_2_y = np.zeros([Nx*Qx,1])
    
    B_line_4_u_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_4_v_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_4_x = np.zeros([Nx*Qx,1])
    B_line_4_u_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_4_v_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_4_y = np.zeros([Nx*Qx,1])
    
    B_line_3_u_x = np.zeros([Ny*Qy,Nx*Ny*J_n])
    B_line_3_v_x = np.zeros([Ny*Qy,Nx*Ny*J_n])
    p_line_3_x = np.zeros([Ny*Qy,1])
    B_line_3_u_y = np.zeros([Ny*Qy,Nx*Ny*J_n])
    B_line_3_v_y = np.zeros([Ny*Qy,Nx*Ny*J_n])
    p_line_3_y = np.zeros([Ny*Qy,1])
    
    
    A_t_u_c = np.zeros([Nx*Qx*(Ny-1),Nx*Ny*J_n]) # x_axis continuity
    A_t_v_c = np.zeros([Nx*Qx*(Ny-1),Nx*Ny*J_n]) # x_axis continuity
    A_x_u_c = np.zeros([Ny*Qy*(Nx-1),Nx*Ny*J_n]) # t_axis continuity
    A_x_v_c = np.zeros([Ny*Qy*(Nx-1),Nx*Ny*J_n]) # t_axis continuity
    
    A_t_ux_c = np.zeros([Nx*Qx*(Ny-1),Nx*Ny*J_n]) # x_axis continuity_C1
    A_t_uy_c = np.zeros([Nx*Qx*(Ny-1),Nx*Ny*J_n]) # x_axis continuity_C1
    A_t_vx_c = np.zeros([Nx*Qx*(Ny-1),Nx*Ny*J_n]) # x_axis continuity_C1
    A_t_vy_c = np.zeros([Nx*Qx*(Ny-1),Nx*Ny*J_n]) # x_axis continuity_C1
    
    A_x_ux_c = np.zeros([Ny*Qy*(Nx-1),Nx*Ny*J_n]) # t_axis continuity_C1
    A_x_uy_c = np.zeros([Ny*Qy*(Nx-1),Nx*Ny*J_n]) # t_axis continuity_C1
    A_x_vx_c = np.zeros([Ny*Qy*(Nx-1),Nx*Ny*J_n]) # t_axis continuity_C1
    A_x_vy_c = np.zeros([Ny*Qy*(Nx-1),Nx*Ny*J_n]) # t_axis continuity_C1
    
    k = 0
    n = 0
    input_points = []
    start = time.time()
    for k in range(Nx):
        for n in range(Ny):
            print("block: ",k,n)
            in_ = points[k][n].detach().numpy()
            input_points.extend(in_[:Qx,:Qy,:].reshape((-1,2)))
            u = models[k][n][0](points[k][n])
            v = models[k][n][1](points[k][n])
            u_values = u.detach().numpy()
            v_values = v.detach().numpy()
            
            J_n_begin = k*Ny*J_n + n*J_n
            
            u_grads = []
            v_grads = []
            u_grad_xx = []
            u_grad_xy = []
            u_grad_yy = []
            v_grad_xx = []
            v_grad_xy = []
            v_grad_yy = []
            
            for i in range(J_n):
                u_xy = torch.autograd.grad(outputs=u[:,:,i], inputs=points[k][n],
                                      grad_outputs=torch.ones_like(u[:,:,i]),
                                      create_graph = True, retain_graph = True)[0]
                v_xy = torch.autograd.grad(outputs=v[:,:,i], inputs=points[k][n],
                                      grad_outputs=torch.ones_like(v[:,:,i]),
                                      create_graph = True, retain_graph = True)[0]
                u_grads.append(u_xy.squeeze().detach().numpy())
                v_grads.append(v_xy.squeeze().detach().numpy())
                
                u_x_xy = torch.autograd.grad(outputs=u_xy[:,:,0], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(u[:,:,i]),
                                  create_graph = True, retain_graph = True)[0]
                u_y_xy = torch.autograd.grad(outputs=u_xy[:,:,1], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(u[:,:,i]),
                                  create_graph = True, retain_graph = True)[0]
                u_grad_xx.append(u_x_xy[:,:,0].squeeze().detach().numpy())
                u_grad_xy.append(u_x_xy[:,:,1].squeeze().detach().numpy())
                u_grad_yy.append(u_y_xy[:,:,1].squeeze().detach().numpy())
                
                v_x_xy = torch.autograd.grad(outputs=v_xy[:,:,0], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(v[:,:,i]),
                                  create_graph = True, retain_graph = True)[0]
                v_y_xy = torch.autograd.grad(outputs=v_xy[:,:,1], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(v[:,:,i]),
                                  create_graph = True, retain_graph = True)[0]
                v_grad_xx.append(v_x_xy[:,:,0].squeeze().detach().numpy())
                v_grad_xy.append(v_x_xy[:,:,1].squeeze().detach().numpy())
                v_grad_yy.append(v_y_xy[:,:,1].squeeze().detach().numpy())
                
            u_grads = np.array(u_grads).swapaxes(0,3) # (2,Qx,Qy,J_n)
            v_grads = np.array(v_grads).swapaxes(0,3) # (2,Qx,Qy,J_n)
            
            u_grad_xx = np.array(u_grad_xx)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n) # (Qx,Qy,J_n)
            u_grad_xy = np.array(u_grad_xy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n) # (Qx,Qy,J_n)
            u_grad_yy = np.array(u_grad_yy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n) # (Qx,Qy,J_n)
            v_grad_xx = np.array(v_grad_xx)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n) # (Qx,Qy,J_n)
            v_grad_xy = np.array(v_grad_xy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n) # (Qx,Qy,J_n)
            v_grad_yy = np.array(v_grad_yy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n) # (Qx,Qy,J_n)
            
            u_bx = np.zeros((Qx*Qy,Nx*Ny*J_n))
            v_bx = np.zeros((Qx*Qy,Nx*Ny*J_n))
            u_by = np.zeros((Qx*Qy,Nx*Ny*J_n))
            v_by = np.zeros((Qx*Qy,Nx*Ny*J_n))
            
            u_bx[:, J_n_begin : J_n_begin + J_n] = -a * (u_grad_xx + b * u_grad_yy)
            v_bx[:, J_n_begin : J_n_begin + J_n] = -a * c * v_grad_xy
            v_by[:, J_n_begin : J_n_begin + J_n] = -a * (v_grad_yy + b * v_grad_xx)
            u_by[:, J_n_begin : J_n_begin + J_n] = -a * c * u_grad_xy
            
            if A_u_bx is None:
                A_u_bx = u_bx
                A_v_bx = v_bx
                A_u_by = u_by
                A_v_by = v_by
                f_bx = np.zeros((Qx*Qy,1))
                f_by = np.zeros((Qx*Qy,1))
            else:
                A_u_bx = np.concatenate((A_u_bx,u_bx),axis = 0)
                A_v_bx = np.concatenate((A_v_bx,v_bx),axis = 0)
                A_u_by = np.concatenate((A_u_by,u_by),axis = 0)
                A_v_by = np.concatenate((A_v_by,v_by),axis = 0)
                f_bx = np.concatenate((f_bx,np.zeros((Qx*Qy,1))),axis = 0)
                f_by = np.concatenate((f_by,np.zeros((Qx*Qy,1))),axis = 0)
            
            # line 1 : x = 0
            if k == 0:
                B_line_1_u[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = u_values[0,:Qy,:]
                B_line_1_v[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = v_values[0,:Qy,:]
                p_line_1_u[n*Qy : n*Qy+Qy,:] = vanal_u(in_[0,:Qy,0],in_[0,:Qy,1]).reshape((Qy,1))
                p_line_1_v[n*Qy : n*Qy+Qy,:] = vanal_v(in_[0,:Qy,0],in_[0,:Qy,1]).reshape((Qy,1))
            
            # line 3 : x = L
            if k == Nx - 1:
                nx3 = 1
                ny3 = 0
                B_line_3_u_x[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (nx3 * u_grads[0,-1,:Qy,:] + ny3 * b * u_grads[1,-1,:Qy,:])
                B_line_3_v_x[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (nx3 * mu * v_grads[1,-1,:Qy,:] + ny3 * b * v_grads[0,-1,:Qy,:])
                B_line_3_u_y[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (ny3 * mu * u_grads[0,-1,:Qy,:] + nx3 * b * u_grads[1,-1,:Qy,:])
                B_line_3_v_y[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (ny3 * v_grads[1,-1,:Qy,:] + nx3 * b * v_grads[0,-1,:Qy,:])
                p_line_3_x[n*Qy : n*Qy+Qy,:] = vanal_px(in_[-1,:Qy,0],in_[-1,:Qy,1],nx3,ny3).reshape((Qy,1))
                p_line_3_y[n*Qy : n*Qy+Qy,:] = vanal_py(in_[-1,:Qy,0],in_[-1,:Qy,1],nx3,ny3).reshape((Qy,1))
            
            # line 4 : y = -D/2
            if n == 0:
                nx4 = 0
                ny4 = -1
                B_line_4_u_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx4 * u_grads[0,:Qx,0,:] + ny4 * b * u_grads[1,:Qx,0,:] )
                B_line_4_v_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx4 * mu * v_grads[1,:Qx,0,:] + ny4 * b * v_grads[0,:Qx,0,:])
                B_line_4_u_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny4 * mu * u_grads[0,:Qx,0,:] + nx4 * b * u_grads[1,:Qx,0,:])
                B_line_4_v_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny4 * v_grads[1,:Qx,0,:] + nx4 * b * v_grads[0,:Qx,0,:])
                p_line_4_x[k*Qx : k*Qx+Qx,:] = vanal_px(in_[:Qx,0,0],in_[:Qx,0,1],nx4,ny4).reshape((Qx,1))
                p_line_4_y[k*Qx : k*Qx+Qx,:] = vanal_py(in_[:Qx,0,0],in_[:Qx,0,1],nx4,ny4).reshape((Qx,1))
            
            # line 2 : y = D/2
            if n == Ny-1:
                nx2 = 0
                ny2 = 1
                B_line_2_u_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx2 * u_grads[0,:Qx,-1,:] + ny2 * b * u_grads[1,:Qx,-1,:])
                B_line_2_v_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx2 * mu * v_grads[1,:Qx,-1,:] + ny2 * b * v_grads[0,:Qx,-1,:])
                B_line_2_u_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny2 * mu * u_grads[0,:Qx,-1,:] + nx2 * b * u_grads[1,:Qx,-1,:])
                B_line_2_v_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny2 * v_grads[1,:Qx,-1,:] + nx2 * b * v_grads[0,:Qx,-1,:])
                p_line_2_x[k*Qx : k*Qx+Qx,:] = vanal_px(in_[:Qx,-1,0],in_[:Qx,-1,1],nx2,ny2).reshape((Qx,1))
                p_line_2_y[k*Qx : k*Qx+Qx,:] = vanal_py(in_[:Qx,-1,0],in_[:Qx,-1,1],nx2,ny2).reshape((Qx,1))
            
            # t_axis continuity
            if Ny > 1:
                t_axis_begin = k*(Ny-1)*Qx + n*Qx 
                if n == 0:
                    A_t_u_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = u_values[:Qx,-1,:]
                    A_t_v_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = v_values[:Qx,-1,:]
                    A_t_ux_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = u_grads[0,:Qx,-1,:]
                    A_t_uy_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = u_grads[1,:Qx,-1,:]
                    A_t_vx_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = v_grads[0,:Qx,-1,:]
                    A_t_vy_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = v_grads[1,:Qx,-1,:]
                elif n == Ny-1:
                    A_t_u_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -u_values[:Qx,0,:]
                    A_t_v_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -v_values[:Qx,0,:]
                    A_t_ux_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[0,:Qx,0,:]
                    A_t_uy_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[1,:Qx,0,:]
                    A_t_vx_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[0,:Qx,0,:]
                    A_t_vy_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[1,:Qx,0,:]
                else:
                    A_t_u_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = u_values[:Qx,-1,:]
                    A_t_v_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = v_values[:Qx,-1,:]
                    A_t_ux_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = u_grads[0,:Qx,-1,:]
                    A_t_uy_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = u_grads[1,:Qx,-1,:]
                    A_t_vx_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = v_grads[0,:Qx,-1,:]
                    A_t_vy_c[t_axis_begin : t_axis_begin + Qx, J_n_begin : J_n_begin + J_n] = v_grads[1,:Qx,-1,:]
                    A_t_u_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -u_values[:Qx,0,:]
                    A_t_v_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -v_values[:Qx,0,:]
                    A_t_ux_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[0,:Qx,0,:]
                    A_t_uy_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[1,:Qx,0,:]
                    A_t_vx_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[0,:Qx,0,:]
                    A_t_vy_c[t_axis_begin - Qx : t_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[1,:Qx,0,:]
            
            # x_axis continuity
            if Nx > 1:
                x_axis_begin = n*(Nx-1)*Qy + k*Qy
                if k == 0:
                    A_x_u_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = u_values[-1,:Qy,:]
                    A_x_v_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = v_values[-1,:Qy,:]
                    A_x_ux_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = u_grads[0,-1,:Qy,:]
                    A_x_uy_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = u_grads[1,-1,:Qy,:]
                    A_x_vx_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = v_grads[0,-1,:Qy,:]
                    A_x_vy_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = v_grads[1,-1,:Qy,:]
                elif k == Nx-1:
                    A_x_u_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -u_values[0,:Qy,:]
                    A_x_v_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -v_values[0,:Qy,:]
                    A_x_ux_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[0,0,:Qy,:]
                    A_x_uy_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[1,0,:Qy,:]
                    A_x_vx_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[0,0,:Qy,:]
                    A_x_vy_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[1,0,:Qy,:]
                else:
                    A_x_u_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = u_values[-1,:Qy,:]
                    A_x_v_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = v_values[-1,:Qy,:]
                    A_x_ux_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = u_grads[0,-1,:Qy,:]
                    A_x_uy_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = u_grads[1,-1,:Qy,:]
                    A_x_vx_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = v_grads[0,-1,:Qy,:]
                    A_x_vy_c[x_axis_begin : x_axis_begin + Qy, J_n_begin : J_n_begin + J_n] = v_grads[1,-1,:Qy,:]
                    A_x_u_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -u_values[0,:Qy,:]
                    A_x_v_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -v_values[0,:Qy,:]
                    A_x_ux_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[0,0,:Qy,:]
                    A_x_uy_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -u_grads[1,0,:Qy,:]
                    A_x_vx_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[0,0,:Qy,:]
                    A_x_vy_c[x_axis_begin - Qy : x_axis_begin, J_n_begin : J_n_begin + J_n] = -v_grads[1,0,:Qy,:]
    
    end = time.time()
    print("time",end-start)
    A_bx = np.concatenate((A_u_bx,A_v_bx),axis = 1)
    A_by = np.concatenate((A_u_by,A_v_by),axis = 1)
    
    B_line_1_u = np.concatenate((B_line_1_u,np.zeros(B_line_1_u.shape)),axis = 1)
    B_line_1_v = np.concatenate((np.zeros(B_line_1_v.shape),B_line_1_v),axis = 1)
    
    B_line_2_x = np.concatenate((B_line_2_u_x,B_line_2_v_x),axis = 1)
    B_line_2_y = np.concatenate((B_line_2_u_y,B_line_2_v_y),axis = 1)
    
    B_line_3_x = np.concatenate((B_line_3_u_x,B_line_3_v_x),axis = 1)
    B_line_3_y = np.concatenate((B_line_3_u_y,B_line_3_v_y),axis = 1)
    
    B_line_4_x = np.concatenate((B_line_4_u_x,B_line_4_v_x),axis = 1)
    B_line_4_y = np.concatenate((B_line_4_u_y,B_line_4_v_y),axis = 1)
    
    A_t_u_c = np.concatenate((A_t_u_c,np.zeros(A_t_u_c.shape)),axis = 1)
    A_t_v_c = np.concatenate((np.zeros(A_t_v_c.shape),A_t_v_c),axis = 1)
    A_x_u_c = np.concatenate((A_x_u_c,np.zeros(A_x_u_c.shape)),axis = 1)
    A_x_v_c = np.concatenate((np.zeros(A_x_v_c.shape),A_x_v_c),axis = 1)
    
    A_t_ux_c = np.concatenate((A_t_ux_c,np.zeros(A_t_ux_c.shape)),axis = 1)
    A_t_uy_c = np.concatenate((A_t_uy_c,np.zeros(A_t_uy_c.shape)),axis = 1)
    A_t_vx_c = np.concatenate((np.zeros(A_t_vx_c.shape),A_t_vx_c),axis = 1)
    A_t_vy_c = np.concatenate((np.zeros(A_t_vy_c.shape),A_t_vy_c),axis = 1)
    
    A_x_ux_c = np.concatenate((A_x_ux_c,np.zeros(A_x_ux_c.shape)),axis = 1)
    A_x_uy_c = np.concatenate((A_x_uy_c,np.zeros(A_x_uy_c.shape)),axis = 1)
    A_x_vx_c = np.concatenate((np.zeros(A_x_vx_c.shape),A_x_vx_c),axis = 1)
    A_x_vy_c = np.concatenate((np.zeros(A_x_vy_c.shape),A_x_vy_c),axis = 1)
    A_continuity = np.concatenate((A_t_u_c,A_t_v_c,A_x_u_c,A_x_v_c,A_t_ux_c,A_t_uy_c,A_t_vx_c,\
                                   A_t_vy_c,A_x_ux_c,A_x_uy_c,A_x_vx_c,A_x_vy_c),axis = 0)
    A = np.concatenate((A_bx,A_by,B_line_1_u,B_line_1_v,B_line_2_x,B_line_2_y,B_line_3_x,B_line_3_y,B_line_4_x \
                        ,B_line_4_y,A_continuity),axis=0)
    f = np.concatenate((f_bx.reshape((-1,1)),f_by.reshape((-1,1)),p_line_1_u,p_line_1_v,p_line_2_x,p_line_2_y,p_line_3_x,p_line_3_y,p_line_4_x \
                        ,p_line_4_y,np.zeros((A_continuity.shape[0],1))),axis=0)
    return(A,f)



def Test(models,Nx,Ny,J_n,Qx,Qy,w,plot = False):
    true_values_u = []
    true_values_v = []
    true_values_sigma_x = []
    true_values_sigma_y = []
    true_values_tau_xy = []
    
    numerical_values_u = []
    numerical_values_v = []
    numerical_values_sigma_x = []
    numerical_values_sigma_y = []
    numerical_values_tau_xy = []
    
    test_Qx = 2*Qx
    test_Qy = 2*Qy
    
    for k in range(Nx):
        true_value_u_x = []
        true_value_v_x = []
        true_value_sigma_x_x = []
        true_value_sigma_y_x = []
        true_value_tau_xy_x = []
        
        numerical_value_u_x = []
        numerical_value_v_x = []
        numerical_value_sigma_x_x = []
        numerical_value_sigma_y_x = []
        numerical_value_tau_xy_x = []
        for n in range(Ny):
            print("test ",k,n)
            # forward and grad
            x_min = (x_r - x_l)/Nx * k + x_l
            x_max = (x_r - x_l)/Nx * (k+1) + x_l
            t_min = (y_u - y_d)/Ny * n + y_d
            t_max = (y_u - y_d)/Ny * (n+1) + y_d
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_devide = np.linspace(t_min, t_max, test_Qy + 1)[:test_Qy]
            grid = np.array(list(itertools.product(x_devide,t_devide)))
            #grid = grid[inner(grid),:]
            test_point = torch.tensor(grid,requires_grad=True)
            out_u = models[k][n][0](test_point)
            value_u = out_u.detach().numpy()
            out_v = models[k][n][1](test_point)
            value_v = out_v.detach().numpy()
            u_grads = []
            v_grads = []
            for i in range(J_n):
                u_xy = torch.autograd.grad(outputs=out_u[:,i], inputs=test_point,
                                      grad_outputs=torch.ones_like(out_u[:,i]),
                                      create_graph = False, retain_graph = True)[0]
                v_xy = torch.autograd.grad(outputs=out_v[:,i], inputs=test_point,
                                      grad_outputs=torch.ones_like(out_v[:,i]),
                                      create_graph = False, retain_graph = True)[0]
                u_grads.append(u_xy.squeeze().detach().numpy())
                v_grads.append(v_xy.squeeze().detach().numpy())
            
            u_grads = np.array(u_grads).swapaxes(0,2) # (2,Qx*Qy,J_n)
            v_grads = np.array(v_grads).swapaxes(0,2) # (2,Qx*Qy,J_n)
            
            true_value_sigma_x = vanal_sigma_x(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            true_value_sigma_y = vanal_sigma_y(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            true_value_tau_xy = vanal_tau_xy(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            true_value_u = vanal_u(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            true_value_v = vanal_v(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            
            numerical_value_u = np.dot(value_u, w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)
            numerical_value_v = np.dot(value_v, w[Nx*Ny*J_n + k*Ny*J_n + n*J_n : Nx*Ny*J_n + k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)
            #numerical_value_sigma_x = np.dot(u_grads[0,:,:], w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)
            #numerical_value_sigma_y = np.dot(v_grads[1,:,:], w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)
            #numerical_value_tau_xy = np.dot(u_grads[1,:,:], w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)\
            #                        + np.dot(v_grads[0,:,:], w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)
            numerical_value_sigma_x = a*(np.dot(u_grads[0,:,:], w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)\
                                    + mu*np.dot(v_grads[1,:,:], w[Nx*Ny*J_n + k*Ny*J_n + n*J_n : Nx*Ny*J_n + k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy))
            numerical_value_sigma_y = a*(np.dot(v_grads[1,:,:], w[Nx*Ny*J_n + k*Ny*J_n + n*J_n : Nx*Ny*J_n + k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)\
                                    + mu*np.dot(u_grads[0,:,:], w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy))
            numerical_value_tau_xy = a*b*(np.dot(u_grads[1,:,:], w[k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy)\
                                    + np.dot(v_grads[0,:,:], w[Nx*Ny*J_n + k*Ny*J_n + n*J_n : Nx*Ny*J_n + k*Ny*J_n + n*J_n + J_n,:]).reshape(test_Qx,test_Qy))
            true_value_u_x.append(true_value_u)
            true_value_v_x.append(true_value_v)
            true_value_sigma_x_x.append(true_value_sigma_x)
            true_value_sigma_y_x.append(true_value_sigma_y)
            true_value_tau_xy_x.append(true_value_tau_xy)
            numerical_value_u_x.append(numerical_value_u)
            numerical_value_v_x.append(numerical_value_v)
            numerical_value_sigma_x_x.append(numerical_value_sigma_x)
            numerical_value_sigma_y_x.append(numerical_value_sigma_y)
            numerical_value_tau_xy_x.append(numerical_value_tau_xy)
        
        true_value_u_x = np.concatenate(true_value_u_x, axis=1)
        true_value_v_x = np.concatenate(true_value_v_x, axis=1)
        true_value_sigma_x_x = np.concatenate(true_value_sigma_x_x, axis=1)
        true_value_sigma_y_x = np.concatenate(true_value_sigma_y_x, axis=1)
        true_value_tau_xy_x = np.concatenate(true_value_tau_xy_x, axis=1)
        true_values_u.append(true_value_u_x)
        true_values_v.append(true_value_v_x)
        true_values_sigma_x.append(true_value_sigma_x_x)
        true_values_sigma_y.append(true_value_sigma_y_x)
        true_values_tau_xy.append(true_value_tau_xy_x)
        
        numerical_value_u_x = np.concatenate(numerical_value_u_x, axis=1)
        numerical_value_v_x = np.concatenate(numerical_value_v_x, axis=1)
        numerical_value_sigma_x_x = np.concatenate(numerical_value_sigma_x_x, axis=1)
        numerical_value_sigma_y_x = np.concatenate(numerical_value_sigma_y_x, axis=1)
        numerical_value_tau_xy_x = np.concatenate(numerical_value_tau_xy_x, axis=1)
        numerical_values_u.append(numerical_value_u_x)
        numerical_values_v.append(numerical_value_v_x)
        numerical_values_sigma_x.append(numerical_value_sigma_x_x)
        numerical_values_sigma_y.append(numerical_value_sigma_y_x)
        numerical_values_tau_xy.append(numerical_value_tau_xy_x)
    
    true_values_u = np.concatenate(true_values_u, axis=0)
    true_values_v = np.concatenate(true_values_v, axis=0)
    true_values_sigma_x = np.concatenate(true_values_sigma_x, axis=0)
    true_values_sigma_y = np.concatenate(true_values_sigma_y, axis=0)
    true_values_tau_xy = np.concatenate(true_values_tau_xy, axis=0)
    numerical_values_u = np.concatenate(numerical_values_u, axis=0)
    numerical_values_v = np.concatenate(numerical_values_v, axis=0)
    numerical_values_sigma_x = np.concatenate(numerical_values_sigma_x, axis=0)
    numerical_values_sigma_y = np.concatenate(numerical_values_sigma_y, axis=0)
    numerical_values_tau_xy = np.concatenate(numerical_values_tau_xy, axis=0)
    
    epsilon_u = np.abs(true_values_u - numerical_values_u)
    epsilon_v = np.abs(true_values_v - numerical_values_v)
    epsilon_sigma_x = np.abs(true_values_sigma_x - numerical_values_sigma_x)
    epsilon_sigma_y = np.abs(true_values_sigma_y - numerical_values_sigma_y)
    epsilon_tau_xy = np.abs(true_values_tau_xy - numerical_values_tau_xy)
    
    e_u = epsilon_u.reshape((-1,1))
    e_v = epsilon_v.reshape((-1,1))
    e_sigma_x = epsilon_sigma_x.reshape((-1,1))
    e_sigma_y = epsilon_sigma_y.reshape((-1,1))
    e_tau_xy = epsilon_tau_xy.reshape((-1,1))
    value_u = true_values_u.reshape((-1,1))
    value_v = true_values_v.reshape((-1,1))
    value_sigma_x = true_values_sigma_x.reshape((-1,1))
    value_sigma_y = true_values_sigma_y.reshape((-1,1))
    value_tau_xy = true_values_tau_xy.reshape((-1,1))
    r = [math.sqrt(sum(e_u*e_u)/len(e_u))/math.sqrt(sum(value_u*value_u)/len(value_u)),\
         math.sqrt(sum(e_v*e_v)/len(e_v))/math.sqrt(sum(value_v*value_v)/len(value_v)),\
         math.sqrt(sum(e_sigma_x*e_sigma_x)/len(e_sigma_x))/math.sqrt(sum(value_sigma_x*value_sigma_x)/len(value_sigma_x)),\
         math.sqrt(sum(e_sigma_y*e_sigma_y)/len(e_sigma_y))/math.sqrt(sum(value_sigma_y*value_sigma_y)/len(value_sigma_y)),\
         math.sqrt(sum(e_tau_xy*e_tau_xy)/len(e_tau_xy))/math.sqrt(sum(value_tau_xy*value_tau_xy)/len(value_tau_xy))]

    print('********************* u ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_u.max(),'L_2=',math.sqrt(sum(e_u*e_u)/len(e_u)),"L_2 relative error = ",r[0])
    print('********************* v ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_v.max(),'L_2=',math.sqrt(sum(e_v*e_v)/len(e_v)),"R=",r[1])
    print('********************* sigma_x ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_sigma_x.max(),'L_2=',math.sqrt(sum(e_sigma_x*e_sigma_x)/len(e_sigma_x)),"R=",r[2])
    print('********************* sigma_y ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_sigma_y.max(),'L_2=',math.sqrt(sum(e_sigma_y*e_sigma_y)/len(e_sigma_y)),"R=",r[3])
    print('********************* tau_xy ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_tau_xy.max(),'L_2=',math.sqrt(sum(e_tau_xy*e_tau_xy)/len(e_tau_xy)),"R=",r[4])
    print('********************* ERROR *********************')
    if plot == False:
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios':[2, 2]}, sharex=True)
        sns.heatmap(epsilon_u.astype(float).T, cmap="YlGnBu", ax=axes[0]).invert_yaxis()
        sns.heatmap(epsilon_v.astype(float).T, cmap="YlGnBu", ax=axes[1]).invert_yaxis()
        plt.savefig('./Timoshenko_beam_result_N=%s_J_n=%s_Q=%s_error=%s_and%s.pdf'%(Nx,J_n,Qx,r[0],r[1]), dpi=300)
        #plt.show()
        '''
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios':[2, 2]}, sharex=True)
        sns.heatmap(true_values_u.astype(float).T, cmap="YlGnBu", ax=axes[0]).invert_yaxis()
        sns.heatmap(true_values_v.astype(float).T, cmap="YlGnBu", ax=axes[1]).invert_yaxis()
        plt.savefig('./Timoshenko_beam_result_N=%s_J_n=%s_Q=%s_true_value.pdf'%(Nx,J_n,Qx), dpi=300)
        
        fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios':[2, 2]}, sharex=True)
        sns.heatmap(numerical_values_u.astype(float).T, cmap="YlGnBu", ax=axes[0]).invert_yaxis()
        sns.heatmap(numerical_values_v.astype(float).T, cmap="YlGnBu", ax=axes[1]).invert_yaxis()
        plt.savefig('./Timoshenko_beam_result_N=%s_J_n=%s_Q=%s_value.pdf'%(Nx,J_n,Qx), dpi=300)
        #plt.show()
        '''
        fig, axes = plt.subplots(3, 1, gridspec_kw={'height_ratios':[2, 2, 2]}, sharex=True)
        sns.heatmap(epsilon_sigma_x.astype(float).T, cmap="YlGnBu", ax=axes[0]).invert_yaxis()
        sns.heatmap(epsilon_sigma_y.astype(float).T, cmap="YlGnBu", ax=axes[1]).invert_yaxis()
        sns.heatmap(epsilon_tau_xy.astype(float).T, cmap="YlGnBu", ax=axes[2]).invert_yaxis()
        #sns.heatmap(epsilon_sigma_y.astype(float).T, cmap="YlGnBu", ax=axes[1]).invert_yaxis()
        plt.savefig('./Timoshenko_beam_result_N=%s_J_n=%s_Q=%s_stress_error=%s_and%s_and%s.pdf'%(Nx,J_n,Qx,r[2],r[3],r[4]), dpi=300)
        #plt.show()
        '''
        fig, axes = plt.subplots(3, 1, gridspec_kw={'height_ratios':[2, 2, 2]}, sharex=True)
        sns.heatmap(numerical_values_sigma_x.astype(float).T, cmap="YlGnBu", ax=axes[0]).invert_yaxis()
        sns.heatmap(numerical_values_sigma_y.astype(float).T, cmap="YlGnBu", ax=axes[1]).invert_yaxis()
        sns.heatmap(numerical_values_tau_xy.astype(float).T, cmap="YlGnBu", ax=axes[2]).invert_yaxis()
        plt.savefig('./Timoshenko_beam_result_N=%s_J_n=%s_Q=%s_stress_value.pdf'%(Nx,J_n,Qx), dpi=300)
        '''

def main(Nx,Ny,J_n,Qx,Qy):
    # prepare models and collocation pointss
    models, points = Pre_Definition(Nx,Ny,J_n,Qx,Qy)
    
    # matrix assembly (Aw=f)
    A,f = Matrix_Assembly(models,points,Nx,Ny,J_n,Qx,Qy)
    f = np.array(f,dtype=np.float64)
    max_value = 10.0
    
    # rescaling
    for i in range(len(A)):
        ratio = max_value/np.abs(A[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    
    # solve
    w = lstsq(A,f)[0]
    
    # test
    Test(models,Nx,Ny,J_n,Qx,Qy,w,False)


if __name__ == '__main__':
    set_seed(100)
    J_n = 400
    for M_p in [1,4]:
        for Q in [20,40,60,80]:
            Nx = int(np.sqrt(M_p))
            Ny = int(np.sqrt(M_p))
            Qx = Q
            Qy = Q
            main(Nx,Ny,J_n,Qx,Qy)
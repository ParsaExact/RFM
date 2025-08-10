# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import itertools
torch.set_default_dtype(torch.float64)

# centers and radius for circles
XY = [[0.0,0.0]]
R = [1.0]

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



# domain \Omega
x_l = -10.0
x_r = 10.0
y_d = -5.0
y_u = 5.0
AA = 1

# coefficient of Elasticity
E = 2.1e5 # Young's modulus
mu = 0.3 # poisson ratio

a = E / (1 - mu**2)
b = (1 - mu) / 2
c = (1 + mu) / 2


# loads
P = 1.0
def px(x,y):
    if x < 0:
        return(-P)
    else:
        return(P)
vanal_px = np.vectorize(px)

# analysis solution
def stress_sigma_x(x,y):
    theta = np.arctan(y/(x + 1E-13))
    Rr = np.sqrt(x**2 + y**2)
    #print(theta)
    if x < 0:
        theta = theta + np.pi
    sigma_x = P * (1 - (AA**2) * (1.5*np.cos(2*theta) + np.cos(4*theta)) / (Rr**2) + 1.5*(AA**4) * np.cos(4*theta) / (Rr**4))
    return sigma_x


def stress_sigma_y(x,y):
    theta = np.arctan(y/(x + 1E-13))
    Rr = np.sqrt(x**2 + y**2)
    if x < 0:
        theta = theta + np.pi
    sigma_y = - P * (AA**2 * (0.5*np.cos(2*theta) - np.cos(4*theta)) / Rr**2 + 1.5*AA**4 * np.cos(4*theta) / Rr**4)
    return sigma_y


def stress_tau_xy(x,y):
    theta = np.arctan(y/(x + 1E-13))
    Rr = np.sqrt(x**2 + y**2)
    if x < 0:
        theta = theta + np.pi
    tau_xy = - P * (AA**2 * (0.5*np.sin(2*theta) + np.sin(4*theta)) / Rr**2 - 1.5*AA**4 * np.sin(4*theta) / Rr**4)
    return tau_xy

vanal_sigma_x = np.vectorize(stress_sigma_x)
vanal_sigma_y = np.vectorize(stress_sigma_y)
vanal_tau_xy = np.vectorize(stress_tau_xy)



def inner(xy):
    out = None
    for c in range(len(XY)):
        x,y = XY[c]
        r = R[c]
        out_now = ((xy[:,0]-x)**2 + (xy[:,1]-y)**2) > r**2  + 1E-10
        if out is None:
            out = out_now
        else:
            out = out*out_now
    return(out)


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


def Neumann_Boundary_Condition(models,Nx,Ny,J_n,c_1_x,c_1_y,nx1,ny1):
    B_c_1_x = np.zeros((c_1_x.shape[0],Nx*Ny*2*J_n))
    B_c_1_y = np.zeros((c_1_x.shape[0],Nx*Ny*2*J_n))
    for i in range(len(c_1_x)):
        x = c_1_x[i]
        y = c_1_y[i]
        in_ = torch.tensor([x,y],requires_grad=True)
        k = min(int((x - x_l) / ((x_r - x_l)/Nx)),Nx-1)
        n = min(int((y - y_d) / ((y_u - y_d)/Ny)),Ny-1)
        u = models[k][n][0](in_)
        v = models[k][n][1](in_)
        u_grads = []
        v_grads = []
        for m in range(J_n):
            u_x_y = torch.autograd.grad(outputs=u[m], inputs=in_,
                                      grad_outputs=torch.ones_like(u[m]),
                                      create_graph = True, retain_graph = True)[0]
            v_x_y = torch.autograd.grad(outputs=v[m], inputs=in_,
                                      grad_outputs=torch.ones_like(v[m]),
                                      create_graph = True, retain_graph = True)[0]
            u_grads.append(u_x_y.squeeze().detach().numpy())
            v_grads.append(v_x_y.squeeze().detach().numpy())
        
        u_grads = np.array(u_grads).swapaxes(0,1)
        v_grads = np.array(v_grads).swapaxes(0,1)
        B_c_1_x[i,k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n] = \
            a * (nx1[i] * u_grads[0,:] + ny1[i] * b * u_grads[1,:])
        B_c_1_x[i,Nx*Ny*J_n + k*Ny*J_n + n*J_n : Nx*Ny*J_n + k*Ny*J_n + n*J_n + J_n] = \
            a * (nx1[i] * mu * v_grads[1,:] + ny1[i] * b * v_grads[0,:])
        B_c_1_y[i,k*Ny*J_n + n*J_n : k*Ny*J_n + n*J_n + J_n] = \
            a * (ny1[i] * mu * u_grads[0,:] + nx1[i] * b * u_grads[1,:])
        B_c_1_y[i,Nx*Ny*J_n + k*Ny*J_n + n*J_n : Nx*Ny*J_n + k*Ny*J_n + n*J_n + J_n] = \
            a * (ny1[i] * v_grads[1,:] + nx1[i] * b * v_grads[0,:])
    return(np.concatenate((B_c_1_x,B_c_1_y),axis=0))


def Matrix_Assembly(models,points,Nx,Ny,J_n,Qx,Qy):
    # matrix define (Aw=b)
    A_u_bx = None
    A_v_bx = None
    A_u_by = None
    A_v_by = None
    f_bx = None
    f_by = None
    
    B_line_1_u_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_1_v_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_1_x = np.zeros([Nx*Qx,1])
    B_line_1_u_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_1_v_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_1_y = np.zeros([Nx*Qx,1])
    
    B_line_2_u_x = np.zeros([Ny*Qy,Nx*Ny*J_n])
    B_line_2_v_x = np.zeros([Ny*Qy,Nx*Ny*J_n])
    p_line_2_x = np.zeros([Ny*Qy,1])
    B_line_2_u_y = np.zeros([Ny*Qy,Nx*Ny*J_n])
    B_line_2_v_y = np.zeros([Ny*Qy,Nx*Ny*J_n])
    p_line_2_y = np.zeros([Ny*Qy,1])
    
    B_line_3_u_x = np.zeros([Ny*Qy,Nx*Ny*J_n])
    B_line_3_v_x = np.zeros([Ny*Qy,Nx*Ny*J_n])
    p_line_3_x = np.zeros([Ny*Qy,1])
    B_line_3_u_y = np.zeros([Ny*Qy,Nx*Ny*J_n])
    B_line_3_v_y = np.zeros([Ny*Qy,Nx*Ny*J_n])
    p_line_3_y = np.zeros([Ny*Qy,1])
    
    B_line_4_u_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_4_v_x = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_4_x = np.zeros([Nx*Qx,1])
    B_line_4_u_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    B_line_4_v_y = np.zeros([Nx*Qx,Nx*Ny*J_n])
    p_line_4_y = np.zeros([Nx*Qx,1])
    
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
    
    for k in range(Nx):
        for n in range(Ny):
            print(k,n)
            in_ = points[k][n].detach().numpy()
            in_index = inner(in_[:Qx,:Qy,:].reshape((-1,2)))
            if sum(in_index) == 0:
                continue
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
                
            u_grads = np.array(u_grads).swapaxes(0,3)
            v_grads = np.array(v_grads).swapaxes(0,3)
            
            u_grad_xx = np.array(u_grad_xx)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n)
            u_grad_xy = np.array(u_grad_xy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n)
            u_grad_yy = np.array(u_grad_yy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n)
            v_grad_xx = np.array(v_grad_xx)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n)
            v_grad_xy = np.array(v_grad_xy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n)
            v_grad_yy = np.array(v_grad_yy)[:,:Qx,:Qy].transpose(1,2,0).reshape(-1,J_n)
            
            u_bx = np.zeros((sum(in_index),Nx*Ny*J_n))
            v_bx = np.zeros((sum(in_index),Nx*Ny*J_n))
            u_by = np.zeros((sum(in_index),Nx*Ny*J_n))
            v_by = np.zeros((sum(in_index),Nx*Ny*J_n))
            u_bx[:, J_n_begin : J_n_begin + J_n] = -a * (u_grad_xx[in_index,:] + b * u_grad_yy[in_index,:])
            v_bx[:, J_n_begin : J_n_begin + J_n] = -a * c * v_grad_xy[in_index,:]
            v_by[:, J_n_begin : J_n_begin + J_n] = -a * (v_grad_yy[in_index,:] + b * v_grad_xx[in_index,:])
            u_by[:, J_n_begin : J_n_begin + J_n] = -a * c * u_grad_xy[in_index,:]
      
            if A_u_bx is None:
                A_u_bx = u_bx
                A_v_bx = v_bx
                A_u_by = u_by
                A_v_by = v_by
            else:
                A_u_bx = np.concatenate((A_u_bx,u_bx),axis = 0)
                A_v_bx = np.concatenate((A_v_bx,v_bx),axis = 0)
                A_u_by = np.concatenate((A_u_by,u_by),axis = 0)
                A_v_by = np.concatenate((A_v_by,v_by),axis = 0)
            
            # line 1: y = Y_min
            if n == 0:
                nx1 = 0
                ny1 = -1
                B_line_1_u_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx1 * u_grads[0,:Qx,0,:] + ny1 * b * u_grads[1,:Qx,0,:])
                B_line_1_v_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx1 * mu * v_grads[1,:Qx,0,:] + ny1 * b * v_grads[0,:Qx,0,:])
                B_line_1_u_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny1 * mu * u_grads[0,:Qx,0,:] + nx1 * b * u_grads[1,:Qx,0,:])
                B_line_1_v_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny1 * v_grads[1,:Qx,0,:] + nx1 * b * v_grads[0,:Qx,0,:])
                p_line_1_x[k*Qx : k*Qx+Qx,:] = 0
                p_line_1_y[k*Qx : k*Qx+Qx,:] = 0       
            
            # line 2 : x = X_min
            if k == 0:
                nx2 = -1
                ny2 = 0
                B_line_2_u_x[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (nx2 * u_grads[0,0,:Qy,:] + ny2 * b * u_grads[1,0,:Qy,:])
                B_line_2_v_x[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (nx2 * mu * v_grads[1,0,:Qy,:] + ny2 * b * v_grads[0,0,:Qy,:])
                B_line_2_u_y[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (ny2 * mu * u_grads[0,0,:Qy,:] + nx2 * b * u_grads[1,0,:Qy,:])
                B_line_2_v_y[n*Qy : n*Qy+Qy, J_n_begin : J_n_begin + J_n] = \
                    a * (ny2 * v_grads[1,0,:Qy,:] + nx2 * b * v_grads[0,0,:Qy,:])
                p_line_2_x[n*Qy : n*Qy+Qy,:] = vanal_px(in_[0,:Qy,0],in_[0,:Qy,1]).reshape((Qy,1))
                p_line_2_y[n*Qy : n*Qy+Qy,:] = 0
            
            # line 3 : x = X_max
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
                p_line_3_x[n*Qy : n*Qy+Qy,:] = vanal_px(in_[-1,:Qy,0],in_[-1,:Qy,1]).reshape((Qy,1))
                p_line_3_y[n*Qy : n*Qy+Qy,:] = 0
            
            # line 4 : y = Y_max
            if n == Ny-1:
                nx4 = 0
                ny4 = 1
                B_line_4_u_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx4 * u_grads[0,:Qx,-1,:] + ny4 * b * u_grads[1,:Qx,-1,:])
                B_line_4_v_x[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (nx4 * mu * v_grads[1,:Qx,-1,:] + ny4 * b * v_grads[0,:Qx,-1,:])
                B_line_4_u_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny4 * mu * u_grads[0,:Qx,-1,:] + nx4 * b * u_grads[1,:Qx,-1,:])
                B_line_4_v_y[k*Qx : k*Qx+Qx, J_n_begin : J_n_begin + J_n] = \
                    a * (ny4 * v_grads[1,:Qx,-1,:] + nx4 * b * v_grads[0,:Qx,-1,:])
                p_line_4_x[k*Qx : k*Qx+Qx,:] = 0
                p_line_4_y[k*Qx : k*Qx+Qx,:] = 0          
            
            # y_axis continuity
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
            
    circle_angles = np.linspace(-np.pi, np.pi, int(0.2*(2*np.pi*Ny*Qy)))
    B_circle = None
    p_circle = None
    
    for i in range(len(XY)):
        x,y = XY[i]
        r = R[i]
        wx = np.cos(circle_angles)*r
        wy = np.sin(circle_angles)*r
        c_x = wx + x
        c_y = wy + y
        nx = -wx / r
        ny = -wy / r
        B_c = Neumann_Boundary_Condition(models,Nx,Ny,J_n,c_x,c_y,nx,ny)
        p_c = np.zeros((B_c.shape[0],1))
        if B_circle is None:
            B_circle = B_c
            p_circle = p_c
        else:
            B_circle = np.concatenate((B_circle,B_c),axis=0)
            p_circle = np.concatenate((p_circle,p_c),axis=0)
    
    A_bx = np.concatenate((A_u_bx,A_v_bx),axis = 1)
    A_by = np.concatenate((A_u_by,A_v_by),axis = 1)
    
    B_line_1_x = np.concatenate((B_line_1_u_x,B_line_1_v_x),axis = 1)
    B_line_1_y = np.concatenate((B_line_1_u_y,B_line_1_v_y),axis = 1)
    p_line_1_x = p_line_1_x
    p_line_1_y = p_line_1_y
    
    B_line_2_x = np.concatenate((B_line_2_u_x,B_line_2_v_x),axis = 1)
    B_line_2_y = np.concatenate((B_line_2_u_y,B_line_2_v_y),axis = 1)
    p_line_2_x = p_line_2_x
    p_line_2_y = p_line_2_y
    
    B_line_3_x = np.concatenate((B_line_3_u_x,B_line_3_v_x),axis = 1)
    B_line_3_y = np.concatenate((B_line_3_u_y,B_line_3_v_y),axis = 1)
    p_line_3_x = p_line_3_x
    p_line_3_y = p_line_3_y
    
    B_line_4_x = np.concatenate((B_line_4_u_x,B_line_4_v_x),axis = 1)
    B_line_4_y = np.concatenate((B_line_4_u_y,B_line_4_v_y),axis = 1)
    p_line_4_x = p_line_4_x
    p_line_4_y = p_line_4_y
    
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
    A = np.concatenate((A_bx,A_by,B_line_1_x,B_line_1_y,B_line_2_x,B_line_2_y\
                        ,B_line_3_x,B_line_3_y,B_line_4_x,B_line_4_y,B_circle,A_continuity),axis=0)
    f_bx = np.zeros((A_bx.shape[0],1))
    f_by = np.zeros((A_by.shape[0],1))
    f = np.concatenate((f_bx.reshape((-1,1)),f_by.reshape((-1,1)),p_line_1_x,p_line_1_y,p_line_2_x,p_line_2_y\
                        ,p_line_3_x,p_line_3_y,p_line_4_x,p_line_4_y,p_circle,np.zeros((A_continuity.shape[0],1))),axis=0)
    
    return(A,f)



def Test(models,Nx,Ny,J_n,Qx,Qy,w,plot = False):
    epsilon_u = []
    epsilon_v = []
    epsilon_sigma_x = []
    epsilon_sigma_y = []
    epsilon_tau_xy = []
    
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
    
    mask = []
    test_Qx = 2*Qx
    test_Qy = 2*Qy
    for k in range(Nx):
        epsilon_u_x = []
        epsilon_v_x = []
        epsilon_sigma_x_x = []
        epsilon_sigma_y_x = []
        epsilon_tau_xy_x = []
        
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
        
        mask_x = []
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
            mask_x.append(inner(grid).reshape(test_Qx,test_Qy))
            for i in range(J_n):
                u_xy = torch.autograd.grad(outputs=out_u[:,i], inputs=test_point,
                                      grad_outputs=torch.ones_like(out_u[:,i]),
                                      create_graph = True, retain_graph = True)[0]
                v_xy = torch.autograd.grad(outputs=out_v[:,i], inputs=test_point,
                                      grad_outputs=torch.ones_like(out_v[:,i]),
                                      create_graph = True, retain_graph = True)[0]
                u_grads.append(u_xy.squeeze().detach().numpy())
                v_grads.append(v_xy.squeeze().detach().numpy())
            
            u_grads = np.array(u_grads).swapaxes(0,2) # (2,Qx*Qy,J_n)
            v_grads = np.array(v_grads).swapaxes(0,2) # (2,Qx*Qy,J_n)
            
            true_value_sigma_x = vanal_sigma_x(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            true_value_sigma_y = vanal_sigma_y(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            true_value_tau_xy = vanal_tau_xy(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qy)
            true_value_sigma_x[np.isnan(true_value_sigma_x)] = 0
            true_value_sigma_y[np.isnan(true_value_sigma_y)] = 0
            true_value_tau_xy[np.isnan(true_value_tau_xy)] = 0
            
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
            e_sigma_x = np.abs(true_value_sigma_x - numerical_value_sigma_x)
            e_sigma_y = np.abs(true_value_sigma_y - numerical_value_sigma_y)
            e_tau_xy = np.abs(true_value_tau_xy - numerical_value_tau_xy)
            true_value_sigma_x_x.append(true_value_sigma_x)
            true_value_sigma_y_x.append(true_value_sigma_y)
            true_value_tau_xy_x.append(true_value_tau_xy)
            numerical_value_u_x.append(numerical_value_u)
            numerical_value_v_x.append(numerical_value_v)
            numerical_value_sigma_x_x.append(numerical_value_sigma_x)
            numerical_value_sigma_y_x.append(numerical_value_sigma_y)
            numerical_value_tau_xy_x.append(numerical_value_tau_xy)
            epsilon_sigma_x_x.append(e_sigma_x)
            epsilon_sigma_y_x.append(e_sigma_y)
            epsilon_tau_xy_x.append(e_tau_xy)
        
        mask_x = np.concatenate(mask_x, axis=1)
        mask.append(mask_x)
        epsilon_sigma_x_x = np.concatenate(epsilon_sigma_x_x, axis=1)
        epsilon_sigma_y_x = np.concatenate(epsilon_sigma_y_x, axis=1)
        epsilon_tau_xy_x = np.concatenate(epsilon_tau_xy_x, axis=1)
        epsilon_sigma_x.append(epsilon_sigma_x_x)
        epsilon_sigma_y.append(epsilon_sigma_y_x)
        epsilon_tau_xy.append(epsilon_tau_xy_x)
        
        true_value_sigma_x_x = np.concatenate(true_value_sigma_x_x, axis=1)
        true_value_sigma_y_x = np.concatenate(true_value_sigma_y_x, axis=1)
        true_value_tau_xy_x = np.concatenate(true_value_tau_xy_x, axis=1)
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
    
    mask = np.concatenate(mask, axis=0)
    true_values_sigma_x = np.concatenate(true_values_sigma_x, axis=0)*mask.astype(float)
    true_values_sigma_y = np.concatenate(true_values_sigma_y, axis=0)*mask.astype(float)
    true_values_tau_xy = np.concatenate(true_values_tau_xy, axis=0)*mask.astype(float)
    numerical_values_u = np.concatenate(numerical_values_u, axis=0)*mask.astype(float)
    numerical_values_v = np.concatenate(numerical_values_v, axis=0)*mask.astype(float)
    numerical_values_sigma_x = np.concatenate(numerical_values_sigma_x, axis=0)*mask.astype(float)
    numerical_values_sigma_y = np.concatenate(numerical_values_sigma_y, axis=0)*mask.astype(float)
    numerical_values_tau_xy = np.concatenate(numerical_values_tau_xy, axis=0)*mask.astype(float)
    epsilon_sigma_x = np.concatenate(epsilon_sigma_x, axis=0)*mask.astype(float)
    epsilon_sigma_y = np.concatenate(epsilon_sigma_y, axis=0)*mask.astype(float)
    epsilon_tau_xy = np.concatenate(epsilon_tau_xy, axis=0)*mask.astype(float)
    
    
    e_sigma_x = epsilon_sigma_x.reshape((-1,1))
    e_sigma_y = epsilon_sigma_y.reshape((-1,1))
    e_tau_xy = epsilon_tau_xy.reshape((-1,1))
    value_sigma_x = true_values_sigma_x.reshape((-1,1))
    value_sigma_y = true_values_sigma_y.reshape((-1,1))
    value_tau_xy = true_values_tau_xy.reshape((-1,1))
    r = [math.sqrt(sum(e_sigma_x*e_sigma_x)/len(e_sigma_x))/math.sqrt(sum(value_sigma_x*value_sigma_x)/len(value_sigma_x)),\
         math.sqrt(sum(e_sigma_y*e_sigma_y)/len(e_sigma_y))/math.sqrt(sum(value_sigma_y*value_sigma_y)/len(value_sigma_y)),\
         math.sqrt(sum(e_tau_xy*e_tau_xy)/len(e_tau_xy))/math.sqrt(sum(value_tau_xy*value_tau_xy)/len(value_tau_xy))]
    print('********************* sigma_x ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_sigma_x.max(),'L_2=',math.sqrt(sum(e_sigma_x*e_sigma_x)/len(e_sigma_x)),'L_2 relative=',r[0])
    print('********************* sigma_y ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_sigma_y.max(),'L_2=',math.sqrt(sum(e_sigma_y*e_sigma_y)/len(e_sigma_y)),'L_2 relative=',r[1])
    print('********************* tau_xy ERROR *********************')
    print('Nx=%s,Ny=%s,J_n=%s,Qx=%s,Qy=%s'%(Nx,Ny,J_n,Qx,Qy))
    print('L_inf=',e_tau_xy.max(),'L_2=',math.sqrt(sum(e_tau_xy*e_tau_xy)/len(e_tau_xy)),'L_2 relative=',r[2])
    
    np.save('./Elasticity_2d_result_J_n_p=%s_J_n=%s_Q=%s_u.npy'%(Nx**2,J_n,Qx),numerical_values_u)
    np.save('./Elasticity_2d_result_J_n_p=%s_J_n=%s_Q=%s_v.npy'%(Nx**2,J_n,Qx),numerical_values_v)
    np.save('./Elasticity_2d_result_J_n_p=%s_J_n=%s_Q=%s_sigma_x.npy'%(Nx**2,J_n,Qx),numerical_values_sigma_x)
    np.save('./Elasticity_2d_result_J_n_p=%s_J_n=%s_Q=%s_sigma_y.npy'%(Nx**2,J_n,Qx),numerical_values_sigma_y)
    np.save('./Elasticity_2d_result_J_n_p=%s_J_n=%s_Q=%s_tau_xy.npy'%(Nx**2,J_n,Qx),numerical_values_tau_xy)
    
    #print('********************* ERROR *********************')
    if plot == True:
        
#        numerical_values_u[~mask] = np.NaN
#        plt.figure(figsize=[10, 10])
#        #plt.gca().invert_yaxis()
#        plt.imshow(numerical_values_u.T, cmap='jet')
#        plt.gca().invert_yaxis()
#        plt.colorbar()
#        plt.savefig('./Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_u.pdf'%(Nx**2,J_n,Qx))
#        
#        numerical_values_v[~mask] = np.NaN
#        plt.figure(figsize=[10, 10])
#        plt.imshow(numerical_values_v.T, cmap='jet')
#        plt.gca().invert_yaxis()
#        plt.colorbar()
#        plt.savefig('./Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_v.pdf'%(Nx**2,J_n,Qx))
        
        numerical_values_sigma_x[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(numerical_values_sigma_x.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_sigma_x.pdf'%(Nx**2,J_n,Qx))
        
        numerical_values_sigma_y[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(numerical_values_sigma_y.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_sigma_y.pdf'%(Nx**2,J_n,Qx))
        
        numerical_values_tau_xy[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(numerical_values_tau_xy.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_tau_xy.pdf'%(Nx**2,J_n,Qx))
        
        epsilon_sigma_x[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(epsilon_sigma_x.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./error_Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_sigma_x.pdf'%(Nx**2,J_n,Qx))
        
        epsilon_sigma_y[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(epsilon_sigma_y.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./error_Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_sigma_y.pdf'%(Nx**2,J_n,Qx))
        
        epsilon_tau_xy[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(epsilon_tau_xy.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./error_Elasticity_sample_Ⅲ_J_n_p=%s_J_n=%s_Q=%s_tau_xy.pdf'%(Nx**2,J_n,Qx))
        
        
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
    Test(models,Nx,Ny,J_n,Qx,Qy,w,True)
    

if __name__ == '__main__':
    set_seed(100)
    for M_p in [36]:
        for J_n in [600]:
            for Q in range(80,200,20):
                Nx = int(math.sqrt(M_p))
                Ny = int(math.sqrt(M_p))
                Qx = 2*Q
                Qy = Q
                main(Nx,Ny,J_n,Qx,Qy)
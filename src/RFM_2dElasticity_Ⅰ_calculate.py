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

# domain \Omega
x_l = -10.0
x_r = 10.0
y_d = -5.0
y_u = 5.0

# coefficient of Elasticity
E = 2.1e5 # Young's modulus
mu = 0.3 # poisson ratio

a = E / (1 - mu**2)
b = (1 - mu) / 2
c = (1 + mu) / 2


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
    x_r_test = 10.0
    x_l_test = -10.0
    y_u_test = 5.0
    y_d_test = -5.0
    mask = []
    test_Qx = int(4000/Nx)
    test_Qy = int(2000/Ny)
    for k in range(Nx):
        
        mask_x = []
        for n in range(Ny):
            x_min = (x_r_test - x_l_test)/Nx * k + x_l_test
            x_max = (x_r_test - x_l_test)/Nx * (k+1) + x_l_test
            t_min = (y_u_test - y_d_test)/Ny * n + y_d_test
            t_max = (y_u_test - y_d_test)/Ny * (n+1) + y_d_test
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_devide = np.linspace(t_min, t_max, test_Qy + 1)[:test_Qy]
            grid = np.array(list(itertools.product(x_devide,t_devide)))
            mask_x.append(inner(grid).reshape(test_Qx,test_Qy))
        
        mask_x = np.concatenate(mask_x, axis=1)
        mask.append(mask_x)
        
    
    mask = np.concatenate(mask, axis=0)
    numerical_values_sigma_x = np.load('./Elasticity_2d_result_M_p=36_J_n=400_Q=60_sigma_x.npy')*mask

    
    #print('********************* ERROR *********************')
    if plot == True:
        
#        numerical_values_u[~mask] = np.NaN
#        plt.figure(figsize=[10, 10])
#        #plt.gca().invert_yaxis()
#        plt.imshow(numerical_values_u.T, cmap='jet')
#        plt.gca().invert_yaxis()
#        plt.colorbar()
#        plt.savefig('./Elasticity_sample_Ⅲ_M_p=%s_J_n=%s_Q=%s_u.pdf'%(Nx**2,J_n,Qx))
#        
#        numerical_values_v[~mask] = np.NaN
#        plt.figure(figsize=[10, 10])
#        plt.imshow(numerical_values_v.T, cmap='jet')
#        plt.gca().invert_yaxis()
#        plt.colorbar()
#        plt.savefig('./Elasticity_sample_Ⅲ_M_p=%s_J_n=%s_Q=%s_v.pdf'%(Nx**2,J_n,Qx))
        
        numerical_values_sigma_x[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(numerical_values_sigma_x.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./Elasticity_sample_Ⅲ_M_p=%s_J_n=%s_Q=%s_sigma_x.pdf'%(Nx**2,J_n,Qx))
        
        numerical_values_sigma_y[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(numerical_values_sigma_y.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./Elasticity_sample_Ⅲ_M_p=%s_J_n=%s_Q=%s_sigma_y.pdf'%(Nx**2,J_n,Qx))
        
        numerical_values_tau_xy[~mask] = np.NaN
        plt.figure(figsize=[10, 10])
        plt.imshow(numerical_values_tau_xy.T, cmap='jet')
        plt.gca().invert_yaxis()
        plt.colorbar()
        plt.savefig('./Elasticity_sample_Ⅲ_M_p=%s_J_n=%s_Q=%s_tau_xy.pdf'%(Nx**2,J_n,Qx))
        
        
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
        for J_n in [400]:
            for Q in range(60,100,10):
                Nx = int(math.sqrt(M_p))
                Ny = int(math.sqrt(M_p))
                Qx = Q
                Qy = Q
                main(Nx,Ny,J_n,Qx,Qy)
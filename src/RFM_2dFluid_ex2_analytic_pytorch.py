# -*- coding: utf-8 -*-


import time
import math
import numpy as np
import sympy as sym
from scipy.linalg import lstsq
import torch
import torch.nn as nn
import itertools

# fix random seed
torch.set_default_dtype(torch.float64)
def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True

# random initialization for parameters in FC layer
rand_mag = 1.0
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -rand_mag, b = rand_mag)
        nn.init.uniform_(m.bias, a = -rand_mag, b = rand_mag)

# network definition
class local_rep(nn.Module):
    def __init__(self, in_features, M, x_max, x_min, t_max, t_min):
        super(local_rep, self).__init__()
        self.in_features = in_features
        self.hidden_features = M
        self.a = torch.tensor([2.0/(x_max - x_min),2.0/(t_max - t_min)])
        self.x_0 = torch.tensor([(x_max + x_min)/2,(t_max + t_min)/2])
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())

    def forward(self,x):
        x = self.a * (x - self.x_0)
        x = self.hidden_layer(x)
        return x

# geometric boundaries
x_l = 0.0
x_r = 1.0
y_d = 0.0
y_u = 1.0

# fluid viscosity
nu = 1.0

# exact solution
def u(x,y):
    u_ = x + x**2 - 2*x*y + x**3 - 3*x*(y**2) + (x**2)*y
    return u_

def v(x,y):
    v_ = -y - 2*x*y + y**2 - 3*(x**2)*y + y**3 - x*(y**2)
    return v_

def p(x,y):
    p_ = x*y + x + y + (x**3)*(y**2) - 4/3
    return p_

def fx(x0,y0):
    x, y = sym.symbols("x y")
    u_ = x + x**2 - 2*x*y + x**3 - 3*x*(y**2) + (x**2)*y
    #v_ = -y - 2*x*y + y**2 - 3*(x**2)*y + y**3 - x*(y**2)
    p_ = x*y + x + y + (x**3)*(y**2) - 4/3

    ux = sym.diff(u_, x)
    uy = sym.diff(u_, y)
    uxx = sym.diff(ux, x)
    uyy = sym.diff(uy, y)
    px = sym.diff(p_, x)
    
    fx_ = - nu * (uxx + uyy) + px
    fx = sym.lambdify((x, y), fx_, "numpy")
    return fx(x0,y0)

def fy(x0,y0):
    x, y = sym.symbols("x y")
    #u_ = x + x**2 - 2*x*y + x**3 - 3*x*(y**2) + (x**2)*y
    v_ = -y - 2*x*y + y**2 - 3*(x**2)*y + y**3 - x*(y**2)
    p_ = x*y + x + y + (x**3)*(y**2) - 4/3

    vx = sym.diff(v_, x)
    vy = sym.diff(v_, y)
    vxx = sym.diff(vx, x)
    vyy = sym.diff(vy, y)
    py = sym.diff(p_, y)
    
    fy_ = - nu * (vxx + vyy) + py
    fy = sym.lambdify((x, y), fy_, "numpy")
    return fy(x0,y0)


# vectorize exact solution functions
vanal_u = np.vectorize(u)
vanal_v = np.vectorize(v)
vanal_p = np.vectorize(p)
vanal_fx = np.vectorize(fx)
vanal_fy = np.vectorize(fy)


# determine if points are inside the given area \Omega
# output type: bool array
def inner(xy):
    out_circle1 = ((xy[:,0]-0.5)**2 + (xy[:,1]-0.2)**2) > 0.1**2  + 1E-14
    out_circle2 = ((xy[:,0]-0.2)**2 + (xy[:,1]-0.8)**2) > 0.1**2  + 1E-14
    out_circle3 = ((xy[:,0]-0.8)**2 + (xy[:,1]-0.8)**2) > 0.1**2  + 1E-14
    return(out_circle1*out_circle2*out_circle3)


# define the local-networks and points in the corresponding regions
def pre_define(Nx,Nt,M,Qx,Qt):
    models = []
    points = []
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = (x_r - x_l)/Nx * k + x_l
        x_max = (x_r - x_l)/Nx * (k+1) + x_l
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Nt):
            t_min = (y_u - y_d)/Nt * n + y_d
            t_max = (y_u - y_d)/Nt * (n+1) + y_d
            model_u = local_rep(in_features = 2, M = M, x_min = x_min, 
                              x_max = x_max, t_min = t_min, t_max = t_max)
            model_v = local_rep(in_features = 2, M = M, x_min = x_min, 
                              x_max = x_max, t_min = t_min, t_max = t_max)
            model_p = local_rep(in_features = 2, M = M, x_min = x_min, 
                              x_max = x_max, t_min = t_min, t_max = t_max)
            model_u = model_u.apply(weights_init)
            model_v = model_v.apply(weights_init)
            model_p = model_p.apply(weights_init)
            model_u = model_u.double()
            model_v = model_v.double()
            model_p = model_p.double()
            for param in model_u.parameters():
                param.requires_grad = False
            for param in model_v.parameters():
                param.requires_grad = False
            for param in model_p.parameters():
                param.requires_grad = False
            model_for_x.append([model_u,model_v,model_p])
            t_devide = np.linspace(t_min, t_max, Qt + 1)
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(Qx+1,Qt+1,2)
            point_for_x.append(torch.tensor(grid,requires_grad=True))
        models.append(model_for_x)
        points.append(point_for_x)
    return(models,points)


# calculate the dirichlet B.C. in circle boundaries
def circle_dirichlet(models,c_1_x,c_1_y):
    B_c_1_u = np.zeros((c_1_x.shape[0],Nx*Nt*3*M))
    B_c_1_v = np.zeros((c_1_x.shape[0],Nx*Nt*3*M))
    B_c_1_p = np.zeros((c_1_x.shape[0],Nx*Nt*3*M))
    u_c_1 = vanal_u(c_1_x,c_1_y).reshape((-1,1))
    v_c_1 = vanal_v(c_1_x,c_1_y).reshape((-1,1))
    #p_c_1 = vanal_p(c_1_x,c_1_y).reshape((-1,1))
    for i in range(len(c_1_x)):
        x = c_1_x[i]
        y = c_1_y[i]
        in_ = torch.tensor([x,y],requires_grad=True)
        k = min(int((x - x_l) / ((x_r - x_l)/Nx)),Nx-1)
        n = min(int((y - y_d) / ((y_u - y_d)/Nt)),Nt-1)
        u = models[k][n][0](in_).detach().numpy()
        v = models[k][n][1](in_).detach().numpy()
        p = models[k][n][2](in_).detach().numpy()
        B_c_1_u[i,k*Nt*M + n*M : k*Nt*M + n*M + M] = u
        B_c_1_v[i,Nx*Nt*M + k*Nt*M + n*M : Nx*Nt*M + k*Nt*M + n*M + M] = v
        B_c_1_p[i,2*Nx*Nt*M + k*Nt*M + n*M : 2*Nx*Nt*M + k*Nt*M + n*M + M] = p
    B_c_1 = np.concatenate((B_c_1_u,B_c_1_v),axis=0)
    return(B_c_1, np.concatenate((u_c_1,v_c_1),axis=0))


# calculate the matrix A,f in linear equations system 'Au=f'
def cal_matrix(models,points,Nx,Nt,M,Qx,Qt):
    # matrix define (Aw=b)
    A_u_fx = None
    A_v_fx = None
    A_p_fx = None
    A_u_fy = None
    A_v_fy = None
    A_p_fy = None
    
    A_u_delta = None
    A_v_delta = None
    
    f_fx = None
    f_fy = None
    
    B_line_1_u = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_1_v = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_1_p = np.zeros([Nt*Qt,Nx*Nt*M])
    p_line_1_u = np.zeros([Nt*Qt,1])
    p_line_1_v = np.zeros([Nt*Qt,1])
    p_line_1_p = np.zeros([Nt*Qt,1])
    
    B_line_2_u = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_2_v = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_2_p = np.zeros([Nt*Qt,Nx*Nt*M])
    p_line_2_u = np.zeros([Nt*Qt,1])
    p_line_2_v = np.zeros([Nt*Qt,1])
    p_line_2_p = np.zeros([Nt*Qt,1])
    
    B_line_3_u = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_3_v = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_3_p = np.zeros([Nt*Qt,Nx*Nt*M])
    p_line_3_u = np.zeros([Nt*Qt,1])
    p_line_3_v = np.zeros([Nt*Qt,1])
    p_line_3_p = np.zeros([Nt*Qt,1])
    
    B_line_4_u = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_4_v = np.zeros([Nt*Qt,Nx*Nt*M])
    B_line_4_p = np.zeros([Nt*Qt,Nx*Nt*M])
    p_line_4_u = np.zeros([Nt*Qt,1])
    p_line_4_v = np.zeros([Nt*Qt,1])
    p_line_4_p = np.zeros([Nt*Qt,1])
    
    A_t_u_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity
    A_t_v_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity
    A_t_p_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity
    A_x_u_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity
    A_x_v_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity
    A_x_p_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity
    
    A_t_ux_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity_C1
    A_t_uy_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity_C1
    A_t_vx_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity_C1
    A_t_vy_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity_C1
    
    A_x_ux_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity_C1
    A_x_uy_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity_C1
    A_x_vx_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity_C1
    A_x_vy_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity_C1
    
    for k in range(Nx):
        for n in range(Nt):
            #print(k,n)
            ''' first we apply equations at C_{I} '''
            in_ = points[k][n].detach().numpy()
            in_index = inner(in_[:Qx,:Qt,:].reshape((-1,2)))
            
            u = models[k][n][0](points[k][n])
            v = models[k][n][1](points[k][n])
            p = models[k][n][2](points[k][n])
            
            u_values = u.detach().numpy()
            v_values = v.detach().numpy()
            p_values = p.detach().numpy()
            
            M_begin = k*Nt*M + n*M
            
            u_grads = []
            v_grads = []
            p_grads = []
            
            u_grad_xx = []
            u_grad_yy = []
            v_grad_xx = []
            v_grad_yy = []
            
            for i in range(M):
                u_xy = torch.autograd.grad(outputs=u[:,:,i], inputs=points[k][n],
                                      grad_outputs=torch.ones_like(u[:,:,i]),
                                      create_graph = True, retain_graph = True)[0]
                v_xy = torch.autograd.grad(outputs=v[:,:,i], inputs=points[k][n],
                                      grad_outputs=torch.ones_like(v[:,:,i]),
                                      create_graph = True, retain_graph = True)[0]
                p_xy = torch.autograd.grad(outputs=p[:,:,i], inputs=points[k][n],
                                      grad_outputs=torch.ones_like(v[:,:,i]),
                                      create_graph = True, retain_graph = True)[0]
                
                u_grads.append(u_xy.squeeze().detach().numpy())
                v_grads.append(v_xy.squeeze().detach().numpy())
                p_grads.append(p_xy.squeeze().detach().numpy())
                
                u_x_xy = torch.autograd.grad(outputs=u_xy[:,:,0], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(u[:,:,i]),
                                  create_graph = False, retain_graph = True)[0]
                u_y_xy = torch.autograd.grad(outputs=u_xy[:,:,1], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(u[:,:,i]),
                                  create_graph = False, retain_graph = True)[0]
                u_grad_xx.append(u_x_xy[:,:,0].squeeze().detach().numpy())
                u_grad_yy.append(u_y_xy[:,:,1].squeeze().detach().numpy())
                
                v_x_xy = torch.autograd.grad(outputs=v_xy[:,:,0], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(v[:,:,i]),
                                  create_graph = False, retain_graph = True)[0]
                v_y_xy = torch.autograd.grad(outputs=v_xy[:,:,1], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(v[:,:,i]),
                                  create_graph = False, retain_graph = True)[0]
                v_grad_xx.append(v_x_xy[:,:,0].squeeze().detach().numpy())
                v_grad_yy.append(v_y_xy[:,:,1].squeeze().detach().numpy())
                
            u_grads = np.array(u_grads).swapaxes(0,3)
            v_grads = np.array(v_grads).swapaxes(0,3)
            p_grads = np.array(p_grads).swapaxes(0,3)
            ux = u_grads[0,:Qx,:Qt,:].reshape(-1,M)
            vy = v_grads[1,:Qx,:Qt,:].reshape(-1,M)
            px = p_grads[0,:Qx,:Qt,:].reshape(-1,M)
            py = p_grads[1,:Qx,:Qt,:].reshape(-1,M)
            
            u_grad_xx = np.array(u_grad_xx)[:,:Qx,:Qt].transpose(1,2,0).reshape(-1,M)
            u_grad_yy = np.array(u_grad_yy)[:,:Qx,:Qt].transpose(1,2,0).reshape(-1,M)
            v_grad_xx = np.array(v_grad_xx)[:,:Qx,:Qt].transpose(1,2,0).reshape(-1,M)
            v_grad_yy = np.array(v_grad_yy)[:,:Qx,:Qt].transpose(1,2,0).reshape(-1,M)
            
            u_fx = np.zeros((sum(in_index),Nx*Nt*M))
            v_fx = np.zeros((sum(in_index),Nx*Nt*M))
            p_fx = np.zeros((sum(in_index),Nx*Nt*M))
            
            u_fy = np.zeros((sum(in_index),Nx*Nt*M))
            v_fy = np.zeros((sum(in_index),Nx*Nt*M))
            p_fy = np.zeros((sum(in_index),Nx*Nt*M))
            
            u_delta = np.zeros((sum(in_index),Nx*Nt*M))
            v_delta = np.zeros((sum(in_index),Nx*Nt*M))
            
            u_fx[:, M_begin : M_begin + M] = -nu*(u_grad_xx[in_index,:] + u_grad_yy[in_index,:])
            v_fx[:, M_begin : M_begin + M] = 0
            p_fx[:, M_begin : M_begin + M] = px[in_index,:]
            u_delta[:, M_begin : M_begin + M] = ux[in_index,:]
            v_delta[:, M_begin : M_begin + M] = vy[in_index,:]
            
            v_fy[:, M_begin : M_begin + M] = -nu*(v_grad_xx[in_index,:] + v_grad_yy[in_index,:])
            u_fy[:, M_begin : M_begin + M] = 0
            p_fy[:, M_begin : M_begin + M] = py[in_index,:]
            
            f_in = in_[:Qx,:Qt,:].reshape((-1,2))[in_index]
            
            if A_u_fx is None:
                A_u_fx = u_fx
                A_v_fx = v_fx
                A_p_fx = p_fx
                
                A_u_fy = u_fy
                A_v_fy = v_fy
                A_p_fy = p_fy
                
                f_fx = vanal_fx(f_in[:,0],f_in[:,1])
                f_fy = vanal_fy(f_in[:,0],f_in[:,1])
                
                A_u_delta = u_delta
                A_v_delta = v_delta
            else:
                A_u_fx = np.concatenate((A_u_fx,u_fx),axis = 0)
                A_v_fx = np.concatenate((A_v_fx,v_fx),axis = 0)
                A_p_fx = np.concatenate((A_p_fx,p_fx),axis = 0)
                
                A_u_fy = np.concatenate((A_u_fy,u_fy),axis = 0)
                A_v_fy = np.concatenate((A_v_fy,v_fy),axis = 0)
                A_p_fy = np.concatenate((A_p_fy,p_fy),axis = 0)
                
                f_fx = np.concatenate((f_fx,vanal_fx(f_in[:,0],f_in[:,1])),axis = 0)
                f_fy = np.concatenate((f_fy,vanal_fy(f_in[:,0],f_in[:,1])),axis = 0)
                
                A_u_delta = np.concatenate((A_u_delta,u_delta),axis = 0)
                A_v_delta = np.concatenate((A_v_delta,v_delta),axis = 0)
            
            
            ''' then we apply boundary conditions at C_{B}'s square boundary '''
            # line 1 : x = 0
            if k == 0:
                B_line_1_u[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = u_values[0,:Qt,:]
                B_line_1_v[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = v_values[0,:Qt,:]
                B_line_1_p[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = p_values[0,:Qt,:]
                p_line_1_u[n*Qt : n*Qt+Qt,:] = vanal_u(in_[0,:Qt,0],in_[0,:Qt,1]).reshape((Qt,1))
                p_line_1_v[n*Qt : n*Qt+Qt,:] = vanal_v(in_[0,:Qt,0],in_[0,:Qt,1]).reshape((Qt,1))
                p_line_1_p[n*Qt : n*Qt+Qt,:] = vanal_p(in_[0,:Qt,0],in_[0,:Qt,1]).reshape((Qt,1))
            
            # line 3 : x = L
            if k == Nx - 1:
                B_line_3_u[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = u_values[-1,:Qt,:]
                B_line_3_v[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = v_values[-1,:Qt,:]
                B_line_3_p[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = p_values[-1,:Qt,:]
                p_line_3_u[n*Qt : n*Qt+Qt,:] = vanal_u(in_[-1,:Qt,0],in_[-1,:Qt,1]).reshape((Qt,1))
                p_line_3_v[n*Qt : n*Qt+Qt,:] = vanal_v(in_[-1,:Qt,0],in_[-1,:Qt,1]).reshape((Qt,1))
                p_line_3_p[n*Qt : n*Qt+Qt,:] = vanal_p(in_[-1,:Qt,0],in_[-1,:Qt,1]).reshape((Qt,1))
            
            # line 2 : y = D/2
            if n == Nt-1:
                B_line_2_u[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = u_values[:Qx,-1,:]
                B_line_2_v[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = v_values[:Qx,-1,:]
                B_line_2_p[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = p_values[:Qx,-1,:]
                p_line_2_u[k*Qx : k*Qx+Qx,:] = vanal_u(in_[:Qx,-1,0],in_[:Qx,-1,1]).reshape((Qx,1))
                p_line_2_v[k*Qx : k*Qx+Qx,:] = vanal_v(in_[:Qx,-1,0],in_[:Qx,-1,1]).reshape((Qx,1))
                p_line_2_p[k*Qx : k*Qx+Qx,:] = vanal_p(in_[:Qx,-1,0],in_[:Qx,-1,1]).reshape((Qx,1))
            
            # line 4 : y = -D/2
            if n == 0:
                B_line_4_u[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = u_values[:Qx,0,:]
                B_line_4_v[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = v_values[:Qx,0,:]
                B_line_4_p[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = p_values[:Qx,0,:]
                p_line_4_u[k*Qx : k*Qx+Qx,:] = vanal_u(in_[:Qx,0,0],in_[:Qx,0,1]).reshape((Qx,1))
                p_line_4_v[k*Qx : k*Qx+Qx,:] = vanal_v(in_[:Qx,0,0],in_[:Qx,0,1]).reshape((Qx,1))
                p_line_4_p[k*Qx : k*Qx+Qx,:] = vanal_p(in_[:Qx,0,0],in_[:Qx,0,1]).reshape((Qx,1))
            
            
            ''' finally we apply smoothness conditions at C_{S} '''
            # t_axis continuity
            if Nt > 1:
                t_axis_begin = k*(Nt-1)*Qx + n*Qx 
                if n == 0:
                    A_t_u_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = u_values[:Qx,-1,:]
                    A_t_v_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = v_values[:Qx,-1,:]
                    A_t_p_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = p_values[:Qx,-1,:]
                    A_t_ux_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = u_grads[0,:Qx,-1,:]
                    A_t_uy_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = u_grads[1,:Qx,-1,:]
                    A_t_vx_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = v_grads[0,:Qx,-1,:]
                    A_t_vy_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = v_grads[1,:Qx,-1,:]
                elif n == Nt-1:
                    A_t_u_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -u_values[:Qx,0,:]
                    A_t_v_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -v_values[:Qx,0,:]
                    A_t_p_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -p_values[:Qx,0,:]
                    A_t_ux_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -u_grads[0,:Qx,0,:]
                    A_t_uy_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -u_grads[1,:Qx,0,:]
                    A_t_vx_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -v_grads[0,:Qx,0,:]
                    A_t_vy_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -v_grads[1,:Qx,0,:]
                else:
                    A_t_u_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = u_values[:Qx,-1,:]
                    A_t_v_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = v_values[:Qx,-1,:]
                    A_t_p_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = p_values[:Qx,-1,:]
                    A_t_ux_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = u_grads[0,:Qx,-1,:]
                    A_t_uy_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = u_grads[1,:Qx,-1,:]
                    A_t_vx_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = v_grads[0,:Qx,-1,:]
                    A_t_vy_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = v_grads[1,:Qx,-1,:]
                    A_t_u_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -u_values[:Qx,0,:]
                    A_t_v_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -v_values[:Qx,0,:]
                    A_t_p_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -p_values[:Qx,0,:]
                    A_t_ux_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -u_grads[0,:Qx,0,:]
                    A_t_uy_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -u_grads[1,:Qx,0,:]
                    A_t_vx_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -v_grads[0,:Qx,0,:]
                    A_t_vy_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -v_grads[1,:Qx,0,:]
            
            # x_axis continuity
            if Nx > 1:
                x_axis_begin = n*(Nx-1)*Qt + k*Qt
                if k == 0:
                    A_x_u_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = u_values[-1,:Qt,:]
                    A_x_v_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = v_values[-1,:Qt,:]
                    A_x_p_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = p_values[-1,:Qt,:]
                    A_x_ux_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = u_grads[0,-1,:Qt,:]
                    A_x_uy_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = u_grads[1,-1,:Qt,:]
                    A_x_vx_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = v_grads[0,-1,:Qt,:]
                    A_x_vy_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = v_grads[1,-1,:Qt,:]
                elif k == Nx-1:
                    A_x_u_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -u_values[0,:Qt,:]
                    A_x_v_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -v_values[0,:Qt,:]
                    A_x_p_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -p_values[0,:Qt,:]
                    A_x_ux_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -u_grads[0,0,:Qt,:]
                    A_x_uy_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -u_grads[1,0,:Qt,:]
                    A_x_vx_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -v_grads[0,0,:Qt,:]
                    A_x_vy_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -v_grads[1,0,:Qt,:]
                else:
                    A_x_u_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = u_values[-1,:Qt,:]
                    A_x_v_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = v_values[-1,:Qt,:]
                    A_x_p_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = p_values[-1,:Qt,:]
                    A_x_ux_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = u_grads[0,-1,:Qt,:]
                    A_x_uy_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = u_grads[1,-1,:Qt,:]
                    A_x_vx_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = v_grads[0,-1,:Qt,:]
                    A_x_vy_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = v_grads[1,-1,:Qt,:]
                    A_x_u_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -u_values[0,:Qt,:]
                    A_x_v_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -v_values[0,:Qt,:]
                    A_x_p_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -p_values[0,:Qt,:]
                    A_x_ux_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -u_grads[0,0,:Qt,:]
                    A_x_uy_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -u_grads[1,0,:Qt,:]
                    A_x_vx_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -v_grads[0,0,:Qt,:]
                    A_x_vy_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -v_grads[1,0,:Qt,:]
    
    ''' apply boundary conditions at C_{B}'s inner circular boundaries '''
    # boundary: circle (0.5, 0.2) r=0.1
    # boundary: circle (0.2, 0.8) r=0.1
    # boundary: circle (0.8, 0.8) r=0.1
    c_angles = np.linspace(-np.pi, np.pi, int(2*np.pi*0.1*(Nt*Qt)))
    c_x_1 = np.cos(c_angles)*0.1 + 0.5
    c_y_1 = np.sin(c_angles)*0.1 + 0.2
    c_x_2 = np.cos(c_angles)*0.1 + 0.2
    c_y_2 = np.sin(c_angles)*0.1 + 0.8
    c_x_3 = np.cos(c_angles)*0.1 + 0.8
    c_y_3 = np.sin(c_angles)*0.1 + 0.8
    c_x = np.concatenate((c_x_1,c_x_2,c_x_3),axis=0)
    c_y = np.concatenate((c_y_1,c_y_2,c_y_3),axis=0)
    B_c,p_c = circle_dirichlet(models,c_x,c_y)
    
    
    ''' below we apply the matrix assembly '''
    A_fx = np.concatenate((A_u_fx,A_v_fx,A_p_fx),axis = 1)
    A_fy = np.concatenate((A_u_fy,A_v_fy,A_p_fy),axis = 1)
    A_delta = np.concatenate((A_u_delta,A_v_delta,np.zeros(A_u_delta.shape)),axis = 1)
    
    B_line_1_u = np.concatenate((B_line_1_u,np.zeros(B_line_1_u.shape),np.zeros(B_line_1_u.shape)),axis = 1)
    B_line_1_v = np.concatenate((np.zeros(B_line_1_v.shape),B_line_1_v,np.zeros(B_line_1_v.shape)),axis = 1)
    B_line_1_p = np.concatenate((np.zeros(B_line_1_p.shape),np.zeros(B_line_1_p.shape),B_line_1_p),axis = 1)
    
    B_line_2_u = np.concatenate((B_line_2_u,np.zeros(B_line_2_u.shape),np.zeros(B_line_2_u.shape)),axis = 1)
    B_line_2_v = np.concatenate((np.zeros(B_line_2_v.shape),B_line_2_v,np.zeros(B_line_2_v.shape)),axis = 1)
    B_line_2_p = np.concatenate((np.zeros(B_line_2_p.shape),np.zeros(B_line_2_p.shape),B_line_2_p),axis = 1)
    
    B_line_3_u = np.concatenate((B_line_3_u,np.zeros(B_line_3_u.shape),np.zeros(B_line_3_u.shape)),axis = 1)
    B_line_3_v = np.concatenate((np.zeros(B_line_3_v.shape),B_line_3_v,np.zeros(B_line_3_v.shape)),axis = 1)
    B_line_3_p = np.concatenate((np.zeros(B_line_3_p.shape),np.zeros(B_line_3_p.shape),B_line_3_p),axis = 1)
    
    B_line_4_u = np.concatenate((B_line_4_u,np.zeros(B_line_4_u.shape),np.zeros(B_line_4_u.shape)),axis = 1)
    B_line_4_v = np.concatenate((np.zeros(B_line_4_v.shape),B_line_4_v,np.zeros(B_line_4_v.shape)),axis = 1)
    B_line_4_p = np.concatenate((np.zeros(B_line_4_p.shape),np.zeros(B_line_4_p.shape),B_line_4_p),axis = 1)
    
    A_t_u_c = np.concatenate((A_t_u_c,np.zeros(A_t_u_c.shape),np.zeros(A_t_u_c.shape)),axis = 1)
    A_t_v_c = np.concatenate((np.zeros(A_t_v_c.shape),A_t_v_c,np.zeros(A_t_v_c.shape)),axis = 1)
    A_t_p_c = np.concatenate((np.zeros(A_t_p_c.shape),np.zeros(A_t_p_c.shape),A_t_p_c),axis = 1)
    A_x_u_c = np.concatenate((A_x_u_c,np.zeros(A_x_u_c.shape),np.zeros(A_x_u_c.shape)),axis = 1)
    A_x_v_c = np.concatenate((np.zeros(A_x_v_c.shape),A_x_v_c,np.zeros(A_x_v_c.shape)),axis = 1)
    A_x_p_c = np.concatenate((np.zeros(A_x_p_c.shape),np.zeros(A_x_p_c.shape),A_x_p_c),axis = 1)
    
    A_t_ux_c = np.concatenate((A_t_ux_c,np.zeros(A_t_ux_c.shape),np.zeros(A_t_ux_c.shape)),axis = 1)
    A_t_uy_c = np.concatenate((A_t_uy_c,np.zeros(A_t_uy_c.shape),np.zeros(A_t_uy_c.shape)),axis = 1)
    A_t_vx_c = np.concatenate((np.zeros(A_t_vx_c.shape),A_t_vx_c,np.zeros(A_t_vx_c.shape)),axis = 1)
    A_t_vy_c = np.concatenate((np.zeros(A_t_vy_c.shape),A_t_vy_c,np.zeros(A_t_vy_c.shape)),axis = 1)
    
    A_x_ux_c = np.concatenate((A_x_ux_c,np.zeros(A_x_ux_c.shape),np.zeros(A_x_ux_c.shape)),axis = 1)
    A_x_uy_c = np.concatenate((A_x_uy_c,np.zeros(A_x_uy_c.shape),np.zeros(A_x_uy_c.shape)),axis = 1)
    A_x_vx_c = np.concatenate((np.zeros(A_x_vx_c.shape),A_x_vx_c,np.zeros(A_x_vx_c.shape)),axis = 1)
    A_x_vy_c = np.concatenate((np.zeros(A_x_vy_c.shape),A_x_vy_c,np.zeros(A_x_vy_c.shape)),axis = 1)
    
    
    A_continuity = np.concatenate((A_t_u_c,A_t_v_c,A_t_p_c,A_x_u_c,A_x_v_c,A_x_p_c,A_t_ux_c,A_t_uy_c,A_t_vx_c,\
                                   A_t_vy_c,A_x_ux_c,A_x_uy_c,A_x_vx_c,A_x_vy_c),axis = 0)
    
    A_bc = np.concatenate((B_line_1_u,B_line_1_v,B_line_1_p,B_line_2_u,B_line_2_v,B_line_2_p,B_line_3_u,B_line_3_v,B_line_3_p,\
                           B_line_4_u ,B_line_4_v,B_line_4_p,B_c),axis=0)
    f_bc = np.concatenate((p_line_1_u,p_line_1_v,p_line_1_p,p_line_2_u,p_line_2_v,p_line_2_p,p_line_3_u,p_line_3_v,p_line_3_p,\
                           p_line_4_u ,p_line_4_v,p_line_4_p,p_c),axis=0)
    A_bc = np.concatenate((B_line_1_u,B_line_1_v,B_line_1_p[0,:].reshape((1,-1)),B_line_2_u,B_line_2_v,B_line_3_u,B_line_3_v,\
                           B_line_4_u ,B_line_4_v,B_c),axis=0)
    f_bc = np.concatenate((p_line_1_u,p_line_1_v,p_line_1_p[0,:].reshape((1,-1)),p_line_2_u,p_line_2_v,p_line_3_u,p_line_3_v,\
                           p_line_4_u ,p_line_4_v,p_c),axis=0)
    A = np.concatenate((A_fx,A_fy,A_delta,A_bc,A_continuity))
    f_delta = np.zeros((A_delta.shape[0],1))
    f = np.concatenate((f_fx.reshape((-1,1)),f_fy.reshape((-1,1)),f_delta,f_bc,np.zeros((A_continuity.shape[0],1))),axis=0)
    
    return(A,f)


# calculate the l^{2}-norm error for u,v,p
def test(models,Nx,Nt,M,Qx,Qt,w):
    numerical_values_u = []
    numerical_values_v = []
    numerical_values_p = []
    
    true_values_u = []
    true_values_v = []
    true_values_p = []

    epsilon_u = []
    epsilon_v = []
    epsilon_p = []
    
    point = []
    mask = []
    test_Qx = int(200/Nx)
    test_Qt = int(200/Nt)
    for k in range(Nx):
        numerical_value_u_x = []
        numerical_value_v_x = []
        numerical_value_p_x = []
        
        true_value_u_x = []
        true_value_v_x = []
        true_value_p_x = []
        
        epsilon_u_x = []
        epsilon_v_x = []
        epsilon_p_x = []
        
        point_x = []
        mask_x = []
        for n in range(Nt):
            #print("test ",k,n)
            # forward and grad
            x_min = (x_r - x_l)/Nx * k + x_l
            x_max = (x_r - x_l)/Nx * (k+1) + x_l
            t_min = (y_u - y_d)/Nt * n + y_d
            t_max = (y_u - y_d)/Nt * (n+1) + y_d
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_devide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
            grid = np.array(list(itertools.product(x_devide,t_devide)))
            #grid = grid[inner(grid),:]
            test_point = torch.tensor(grid,requires_grad=True)
            point_x.append(test_point.detach().numpy().reshape(test_Qx,test_Qt,2))
            out_u = models[k][n][0](test_point)
            value_u = out_u.detach().numpy()
            out_v = models[k][n][1](test_point)
            value_v = out_v.detach().numpy()
            out_p = models[k][n][2](test_point)
            value_p = out_p.detach().numpy()
            mask_x.append(inner(grid).reshape(test_Qx,test_Qt))
            
            true_value_u = vanal_u(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qt)
            true_value_v = vanal_v(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qt)
            true_value_p = vanal_p(grid[:,0],grid[:,1]).reshape(test_Qx,test_Qt)
            
            numerical_value_u = np.dot(value_u, w[k*Nt*M + n*M : k*Nt*M + n*M + M,:]).reshape(test_Qx,test_Qt)
            numerical_value_v = np.dot(value_v, w[Nx*Nt*M + k*Nt*M + n*M : Nx*Nt*M + k*Nt*M + n*M + M,:]).reshape(test_Qx,test_Qt)
            numerical_value_p = np.dot(value_p, w[2*Nx*Nt*M + k*Nt*M + n*M : 2*Nx*Nt*M + k*Nt*M + n*M + M,:]).reshape(test_Qx,test_Qt)
            
            e_u = np.abs(true_value_u - numerical_value_u)
            e_v = np.abs(true_value_v - numerical_value_v)
            e_p = np.abs(true_value_p - numerical_value_p)
            
            true_value_u_x.append(true_value_u)
            true_value_v_x.append(true_value_v)
            true_value_p_x.append(true_value_p)
            
            numerical_value_u_x.append(numerical_value_u)
            numerical_value_v_x.append(numerical_value_v)
            numerical_value_p_x.append(numerical_value_p)
            
            epsilon_u_x.append(e_u)
            epsilon_v_x.append(e_v)
            epsilon_p_x.append(e_p)
            
        
        mask_x = np.concatenate(mask_x, axis=1)
        mask.append(mask_x)
        
        epsilon_u_x = np.concatenate(epsilon_u_x, axis=1)
        epsilon_v_x = np.concatenate(epsilon_v_x, axis=1)
        epsilon_p_x = np.concatenate(epsilon_p_x, axis=1)
        epsilon_u.append(epsilon_u_x)
        epsilon_v.append(epsilon_v_x)
        epsilon_p.append(epsilon_p_x)
        
        numerical_value_u_x = np.concatenate(numerical_value_u_x, axis=1)
        numerical_value_v_x = np.concatenate(numerical_value_v_x, axis=1)
        numerical_value_p_x = np.concatenate(numerical_value_p_x, axis=1)
        numerical_values_u.append(numerical_value_u_x)
        numerical_values_v.append(numerical_value_v_x)
        numerical_values_p.append(numerical_value_p_x)
        
        true_value_u_x = np.concatenate(true_value_u_x, axis=1)
        true_value_v_x = np.concatenate(true_value_v_x, axis=1)
        true_value_p_x = np.concatenate(true_value_p_x, axis=1)
        true_values_u.append(true_value_u_x)
        true_values_v.append(true_value_v_x)
        true_values_p.append(true_value_p_x)
        
        point_x = np.concatenate(point_x, axis=1)
        point.append(point_x)
    
    mask = np.concatenate(mask, axis=0)
    epsilon_u = np.concatenate(epsilon_u, axis=0)*mask.astype(float)
    epsilon_v = np.concatenate(epsilon_v, axis=0)*mask.astype(float)
    epsilon_p = np.concatenate(epsilon_p, axis=0)*mask.astype(float)
    numerical_values_u = np.concatenate(numerical_values_u, axis=0)*mask.astype(float)
    numerical_values_v = np.concatenate(numerical_values_v, axis=0)*mask.astype(float)
    numerical_values_p = np.concatenate(numerical_values_p, axis=0)*mask.astype(float)
    true_values_u = np.concatenate(true_values_u, axis=0)*mask.astype(float)
    true_values_v = np.concatenate(true_values_v, axis=0)*mask.astype(float)
    true_values_p = np.concatenate(true_values_p, axis=0)*mask.astype(float)
    point = np.concatenate(point, axis=0)
    
    e_u = epsilon_u.reshape((-1,1))
    e_v = epsilon_v.reshape((-1,1))
    e_p = epsilon_p.reshape((-1,1))
    value_u = true_values_u.reshape((-1,1))
    value_v = true_values_v.reshape((-1,1))
    value_p = true_values_p.reshape((-1,1))
    
    r = [math.sqrt(sum(e_u*e_u)/len(e_u))/math.sqrt(sum(value_u*value_u)/len(value_u)),\
         math.sqrt(sum(e_v*e_v)/len(e_v))/math.sqrt(sum(value_v*value_v)/len(value_v)),\
         math.sqrt(sum(e_p*e_p)/len(e_p))/math.sqrt(sum(value_p*value_p)/len(value_p))]
    
    #print("************** error **************")
    print(r)
    #print("************** error **************")
    return(r)
    


def main(Nx,Nt,M,Qx,Qt,plot = False):
    # prepare models and collocation pointss
    time_begin = time.time()
    models,points = pre_define(Nx,Nt,M,Qx,Qt)
    time_end = time.time()
    print("pre define time:",time_end - time_begin)
    
    # matrix define (Aw=b)
    time_begin = time.time()
    A,f = cal_matrix(models,points,Nx,Nt,M,Qx,Qt)
    f = np.array(f,dtype=np.float64)
    time_end = time.time()
    print("cal time:",time_end - time_begin)
    max_value = 10.0
    for i in range(len(A)):
        if np.abs(A[i,:].max())==0:
            #print("error line : ",i)
            continue
        ratio = max_value/np.abs(A[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    #np.save('./A.npy',A)
    #np.save('./f.npy',f)
    
    # solve
    time_begin = time.time()
    w = lstsq(A,f)[0]
    time_end = time.time()
    print("solve time:",time_end - time_begin)
    #np.save('./u.npy',w)
    
    # test
    time_begin = time.time()
    r = test(models,Nx,Nt,M,Qx,Qt,w)
    time_end = time.time()
    print("test time:",time_end - time_begin)
    return(r)

if __name__ == '__main__':
    result = []
    for N in [1,2,4]:
        for Q in [5,10,20,40]:
            for M in [100,200,400]:
                Nx = N
                Nt = N
                Qx = Q
                Qt = Q
                r = [N,Q,M]
                r.extend(main(Nx,Nt,M,Qx,Qt))
                result.append(r)
    np.save('./fluid_result.npy',np.array(result))
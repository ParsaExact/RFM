# -*- coding: utf-8 -*-

import time
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import itertools
import seaborn as sns
torch.set_default_dtype(torch.float64)

def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True


rand_mag = 0.75
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -rand_mag, b = rand_mag)
        nn.init.uniform_(m.bias, a = -rand_mag, b = rand_mag)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)


class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, x_max, x_min, t_max, t_min):
        super(local_rep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers  = hidden_layers
        self.M = M
        self.a = torch.tensor([2.0/(x_max - x_min),2.0/(t_max - t_min)])
        self.x_0 = torch.tensor([(x_max + x_min)/2,(t_max + t_min)/2])
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())
        #print([x_min,x_max],[t_min,t_max])

    def forward(self,x):
        x = self.a * (x - self.x_0)
        x = self.hidden_layer(x)
        return x


L = 5.0
tF = 0.25
Nb = 1
tf = tF / Nb
nu = 0

def anal_u(x,y):
    u = (1 + x/10)*(1 + y/10)*\
        (2*np.cos(np.pi*x+2*np.pi/5) + 1.5*np.cos(2*np.pi*x-3*np.pi/5))*\
        (2*np.cos(np.pi*y+2*np.pi/5) + 1.5*np.cos(2*np.pi*y-3*np.pi/5))
    return u

def anal_u_x(x,y):
    u = 1/10 * (1 + y/10)*\
        (2*np.cos(np.pi*x+2*np.pi/5) + 1.5*np.cos(2*np.pi*x-3*np.pi/5))*\
        (2*np.cos(np.pi*y+2*np.pi/5) + 1.5*np.cos(2*np.pi*y-3*np.pi/5))-\
        (1 + x/10)*(1 + y/10)*\
        (2*np.pi*np.sin(np.pi*x+2*np.pi/5) + 3*np.pi*np.sin(2*np.pi*x-3*np.pi/5))*\
        (2*np.cos(np.pi*y+2*np.pi/5) + 1.5*np.cos(2*np.pi*y-3*np.pi/5))
    return u

def anal_u_y(x,y):
    return anal_u_x(y,x)

def anal_u_xx(x,y):
    u = -2/10 * (1 + y/10)*\
        (2*np.pi*np.sin(np.pi*x+2*np.pi/5) + 3*np.pi*np.sin(2*np.pi*x-3*np.pi/5))*\
        (2*np.cos(np.pi*y+2*np.pi/5) + 1.5*np.cos(2*np.pi*y-3*np.pi/5))-\
        (1 + x/10)*(1 + y/10)*\
        (2*np.pi*np.pi*np.cos(np.pi*x+2*np.pi/5) + 6*np.pi*np.pi*np.cos(2*np.pi*x-3*np.pi/5))*\
        (2*np.cos(np.pi*y+2*np.pi/5) + 1.5*np.cos(2*np.pi*y-3*np.pi/5))
    return u

def anal_f(x,y): # f(x,y) = u_y + u*u_x - nu*u_xx
    f = anal_u_y(x,y) + anal_u(x,y)*anal_u_x(x,y) - nu*anal_u_xx(x,y)
    return f

vanal_u = np.vectorize(anal_u)
vanal_f = np.vectorize(anal_f)


def pre_define(Nx,Nt,M,Qx,Qt):
    models = []
    points = []
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = L/Nx * k
        x_max = L/Nx * (k+1)
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Nt):
            t_min = tf/Nt * n
            t_max = tf/Nt * (n+1)
            model = local_rep(in_features = 2, out_features = 1, hidden_layers = 1, M = M, x_min = x_min, 
                              x_max = x_max, t_min = t_min, t_max = t_max)
            model = model.apply(weights_init)
            model = model.double()
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
            t_devide = np.linspace(t_min, t_max, Qt + 1)
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(Qx+1,Qt+1,2)
            point_for_x.append(torch.tensor(grid,requires_grad=True))
        models.append(model_for_x)
        points.append(point_for_x)
    return(models,points)


def cal_matrix(models,points,Nx,Nt,M,Qx,Qt,h_begin,plot_n):
    # matrix define (Aw=b)
    u = np.zeros([Nx*Nt*Qx*Qt,Nx*Nt*M])
    u_t = np.zeros([Nx*Nt*Qx*Qt,Nx*Nt*M])
    u_x = np.zeros([Nx*Nt*Qx*Qt,Nx*Nt*M])
    u_xx = np.zeros([Nx*Nt*Qx*Qt,Nx*Nt*M])
    f = np.zeros([Nx*Nt*Qx*Qt,1])
    
    g1 = np.zeros([Nt*Qt,Nx*Nt*M]) # u(a,t)
    g1_value = np.zeros([Nt*Qt,1]) # g1(t)
    
    g2 = np.zeros([Nt*Qt,Nx*Nt*M]) # u(b,t)
    g2_value = np.zeros([Nt*Qt,1]) # g2(t)
    
    h = np.zeros([Nx*Qx,Nx*Nt*M]) # u(x,0)
    h_value = np.zeros([Nx*Qx,1]) # h(x)
    
    A_t_c = np.zeros([Nx*Qx*(Nt-1),Nx*Nt*M]) # x_axis continuity
    f_t_c = np.zeros([Nx*Qx*(Nt-1),1])
    
    A_x_c = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity
    f_x_c = np.zeros([Nt*Qt*(Nx-1),1])
    
    A_x_c_1 = np.zeros([Nt*Qt*(Nx-1),Nx*Nt*M]) # t_axis continuity_C1
    f_x_c_1 = np.zeros([Nt*Qt*(Nx-1),1])
    
    k = 0
    n = 0
    for k in range(Nx):
        for n in range(Nt):
            # u_t - c*u_x = 0
            in_ = points[k][n].detach().numpy()
            out = models[k][n](points[k][n])
            values = out.detach().numpy()
            M_begin = k*Nt*M + n*M
            #print(values.shape)
            grads = []
            grads_2_xx = []
            for i in range(M):
                g_1 = torch.autograd.grad(outputs=out[:,:,i], inputs=points[k][n],
                                      grad_outputs=torch.ones_like(out[:,:,i]),
                                      create_graph = True, retain_graph = True)[0]
                grads.append(g_1.squeeze().detach().numpy())
                
                g_2_x = torch.autograd.grad(outputs=g_1[:,:,0], inputs=points[k][n],
                                  grad_outputs=torch.ones_like(out[:,:,i]),
                                  create_graph = True, retain_graph = True)[0]
                grads_2_xx.append(g_2_x[:,:,0].squeeze().detach().numpy())
                
            grads = np.array(grads).swapaxes(0,3)
            grads_2_xx = np.array(grads_2_xx)[:,:Qx,:Qt].transpose(1,2,0).reshape(-1,M)
            
            u[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin : M_begin + M] = values[:Qx,:Qt,:].reshape(-1,M)
            u_x[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin : M_begin + M] = grads[0,:Qx,:Qt,:].reshape(-1,M)
            u_t[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin : M_begin + M] = grads[1,:Qx,:Qt,:].reshape(-1,M)
            u_xx[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, M_begin : M_begin + M] = grads_2_xx
            f_in = in_[:Qx,:Qt,:].reshape((-1,2))
            f[k*Nt*Qx*Qt + n*Qx*Qt : k*Nt*Qx*Qt + n*Qx*Qt + Qx*Qt, :] = vanal_f(f_in[:,0],f_in[:,1]+plot_n*tf).reshape(-1,1)
            
            # u(x,0) = ..
            if n == 0 and len(h_begin)==0:
                h[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = values[:Qx,0,:]
                h_value[k*Qx : k*Qx+Qx,:] = \
                    vanal_u(in_[:Qx,0,0],in_[:Qx,0,1]).reshape((Qx,1))
            elif n == 0 and len(h_begin)>0:
                h[k*Qx : k*Qx+Qx, M_begin : M_begin + M] = values[:Qx,0,:]
                h_value[k*Qx : k*Qx+Qx,:] = h_begin[k*Qx:(k+1)*Qx].reshape((Qx,1))
            
            # u(a,t) = ..
            if k == 0:
                g1[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = values[0,:Qt,:]
                g1_value[n*Qt : n*Qt+Qt,:] =\
                    vanal_u(in_[0,:Qt,0],in_[0,:Qt,1]+plot_n*tf).reshape((Qt,1))
            
            # u(b,t) = ..
            if k == Nx-1:
                g2[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = values[-1,:Qt,:]
                g2_value[n*Qt : n*Qt+Qt,:] = \
                    vanal_u(in_[-1,:Qt,0],in_[-1,:Qt,1]+plot_n*tf).reshape((Qt,1))
            
            # t_axis continuity
            if Nt > 1:
                t_axis_begin = k*(Nt-1)*Qx + n*Qx 
                if n == 0:
                    A_t_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = values[:Qx,-1,:]
                    #A_t_c_1[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = grads[1,:Qx,-1,:]
                elif n == Nt-1:
                    A_t_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -values[:Qx,0,:]
                    #A_t_c_1[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -grads[1,:Qx,0,:]
                else:
                    A_t_c[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = values[:Qx,-1,:]
                    A_t_c[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -values[:Qx,0,:]
                    #A_t_c_1[t_axis_begin : t_axis_begin + Qx, M_begin : M_begin + M] = grads[1,:Qx,-1,:]
                    #A_t_c_1[t_axis_begin - Qx : t_axis_begin, M_begin : M_begin + M] = -grads[1,:Qx,0,:]
            
            # x_axis continuity
            if Nx > 1:
                x_axis_begin = n*(Nx-1)*Qt + k*Qt
                if k == 0:
                    A_x_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = values[-1,:Qt,:]
                    A_x_c_1[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = grads[0,-1,:Qt,:]
                elif k == Nx-1:
                    A_x_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -values[0,:Qt,:]
                    A_x_c_1[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -grads[0,0,:Qt,:]
                else:
                    A_x_c[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = values[-1,:Qt,:]
                    A_x_c[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -values[0,:Qt,:]
                    A_x_c_1[x_axis_begin : x_axis_begin + Qt, M_begin : M_begin + M] = grads[0,-1,:Qt,:]
                    A_x_c_1[x_axis_begin - Qt : x_axis_begin, M_begin : M_begin + M] = -grads[0,0,:Qt,:]
    value_and_continue = np.concatenate((g1,g2,h,A_t_c,A_x_c,A_x_c_1),axis=0)
    value_and_continue_value = np.concatenate((g1_value,g2_value,h_value,f_t_c,f_x_c,f_x_c_1),axis=0)
    return(u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value)


def test(models,Nx,Nt,M,Qx,Qt,w,plot_n,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Qx = 2*Qx
    test_Qt = 2*Qt
    h = []
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Nt):
            # forward and grad
            x_min = L/Nx * k
            x_max = L/Nx * (k+1)
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = tf/Nt * n
            t_max = tf/Nt * (n+1)
            t_devide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(test_Qx,test_Qt,2)
            test_point = torch.tensor(grid,requires_grad=True)
            in_ = test_point.detach().numpy()
            out = models[k][n](test_point)
            values = out.detach().numpy()
            ############ 2-order ############
            true_value = vanal_u(in_[:,:,0],in_[:,:,1] + plot_n*tf)
            numerical_value = np.dot(values, w[k*Nt*M + n*M : k*Nt*M + n*M + M]).reshape(test_Qx,test_Qt)
            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        
        x_devide_h = torch.tensor(np.linspace(x_min, x_max, Qx + 1)[:Qx]).unsqueeze(dim=1)
        point_h = x_devide_h.repeat((1,2))
        point_h[:,1] = tf
        out = models[k][n](point_h)
        values = out.detach().numpy()
        h.append(np.dot(values, w[k*Nt*M + Nt*M - M : k*Nt*M + Nt*M]).reshape(Qx,1))
        
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    h = np.array(h).reshape(Nx*Qx)
    epsilon = np.concatenate(epsilon, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1,1))
    print('********************* ERROR *********************')
    print('Nx=%s,Nt=%s,M=%s,Qx=%s,Qt=%s'%(Nx,Nt,M,Qx,Qt))
    print('L_inf=',e.max(),'L_2=',math.sqrt(sum(e*e)/len(e)))
    print("边值条件误差")
    print(max(epsilon[0,:]),max(epsilon[-1,:]),max(epsilon[:,0]),max(epsilon[:,-1]))
    print('********************* ERROR *********************')
    if plot == True:
        sns.heatmap(epsilon.T, cmap="YlGnBu").invert_yaxis()
        #sns.heatmap(true_values.T, cmap="YlGnBu").invert_yaxis()
        #sns.heatmap(numerical_values.T, cmap="YlGnBu").invert_yaxis()
    return(e.max(),math.sqrt(sum(e*e)/len(e)),h)

def cal_f(w,u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value):
    part_1 = np.dot(u_t,w) + np.dot(u,w)*np.dot(u_x,w) - nu*np.dot(u_xx,w) - f.reshape((-1))
    part_2 = np.dot(value_and_continue,w) - value_and_continue_value.reshape((-1))
    return(np.concatenate((part_1,part_2),axis=0))

def cal_j(w,u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value):
    part_1 = u_t + np.dot(u_x,w).reshape(-1,1)*u + np.dot(u,w).reshape(-1,1)*u_x- nu*u_xx
    part_2 = value_and_continue
    return(np.concatenate((part_1,part_2),axis=0))

def main(Nx,Nt,M,Qx,Qt,plot = False,moore = False):
    # prepare models and collocation pointss
    #print("pre define")
    models,points = pre_define(Nx,Nt,M,Qx,Qt)
    
    h_begin = []
    # matrix define (Aw=b)
    u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value = cal_matrix(models,points,Nx,Nt,M,Qx,Qt,h_begin,0)
    
    # solve
    #print("begin solve")
    delta = 0.5
    cost_max = 1e-3
    xi_1 = np.random.uniform(low=0.0, high=1.0, size=None)
    x_0 = np.random.uniform(low=-xi_1*delta, high=xi_1*delta, size=(Nx*Nt*M))
    w = least_squares(cal_f,x_0,cal_j,args=(u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value))
    print("time block = 0, cost = ",w.cost)
    while w.cost > cost_max:
        x_0 = np.random.uniform(low=-xi_1*delta, high=xi_1*delta, size=(Nx*Nt*M))
        w = least_squares(cal_f,x_0,cal_j,args=(u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value))
        print("time block = 0, cost = ",w.cost)
    #print(w)
    
    time_block = Nb
    for plot_n in range(1,time_block):
        # test
        error_Linf,error_L2,h_begin = test(models,Nx,Nt,M,Qx,Qt,w.x,plot_n-1,False)
        #print(h_begin.shape)
        
        # matrix define (Aw=b)
        u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value = cal_matrix(models,points,Nx,Nt,M,Qx,Qt,h_begin,plot_n)
        
        # solve
        xi_1 = np.random.uniform(low=0.0, high=1.0, size=None)
        x_0 = np.random.uniform(low=-xi_1*delta, high=xi_1*delta, size=(Nx*Nt*M))
        w = least_squares(cal_f,x_0,cal_j,args=(u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value))
        print("time block = %s, cost = "%(plot_n),w.cost)
        while w.cost > cost_max:
            x_0 = np.random.uniform(low=-xi_1*delta, high=xi_1*delta, size=(Nx*Nt*M))
            w = least_squares(cal_f,x_0,cal_j,args=(u,u_t,u_x,u_xx,f,value_and_continue,value_and_continue_value))
            print("time block = %s, cost = "%(plot_n),w.cost)
    # test
    #print("begin test")
    return(test(models,Nx,Nt,M,Qx,Qt,w.x,time_block-1,plot))


if __name__ == '__main__':
    #set_seed(100)
    Nx = 5 # the number of sub-domains 
    Nt = 1
    M = 200 # the number of training parameters per sub-domain
    Qx = 20 # the number of collocation pointss per sub-domain 
    Qt = 20
    a = main(Nx,Nt,M,Qx,Qt,True)
    
    '''
    # different Nt test
    N_list = [[1,1],[2,1],[2,2],[4,2]]
    error_inf = []
    error_L2 = []
    for [Nx,Nt] in N_list:
        e_inf,e_L2 = main(Nx,Nt,M,Qx,Qt,False)
        error_inf.append(e_inf)
        error_L2.append(e_L2)
    plt.figure()
    plt.plot([1,2,4,8], error_L2, label = "$L_2$ error", color='green')
    plt.plot([1,2,4,8], error_inf, label = "$L_{inf}$ error", color='orange', linestyle='-.')
    plt.yscale("log")
    plt.legend()
    plt.title('local ELM error : different N')
    plt.savefig('./result/2dim_different_N.pdf', dpi=100)
    '''
    
    '''
    # different Q test
    Q_list = list(range(5,25))
    error_inf = []
    error_L2 = []
    times = []
    for Q in Q_list:
        Qx,Qt = Q,Q
        tic = time.time()
        e_inf,e_L2 = main(Nx,Nt,M,Qx,Qt,False)
        toc = time.time()
        error_inf.append(e_inf)
        error_L2.append(e_L2)
        times.append(toc-tic)
    plt.figure()
    plt.plot(Q_list, error_L2, label = "$L_2$ error", color='green')
    plt.plot(Q_list, error_inf, label = "$L_{inf}$ error", color='orange', linestyle='-.')
    plt.yscale("log")
    plt.legend()
    plt.title('local ELM error : different Q')
    plt.savefig('./result/poisson_2-dim_different_Q.pdf', dpi=100)
    
    plt.figure()
    plt.plot(Q_list, times, label = "training time", color='orange')
    plt.legend()
    plt.title('local ELM training time : different Q')
    plt.savefig('./result/poisson_2-dim_different_Q_time.pdf', dpi=100)
    '''
    
    '''
    # different M test
    M_list = list(range(25,600,25))
    error_inf = []
    error_L2 = []
    for M in M_list:
        e_inf,e_L2 = main(Nx,Nt,M,Qx,Qt,False)
        error_inf.append(e_inf)
        error_L2.append(e_L2)
    plt.figure()
    plt.plot(M_list, error_L2, label = "$L_2$ error", color='green')
    plt.plot(M_list, error_inf, label = "$L_{inf}$ error", color='orange', linestyle='-.')
    plt.yscale("log")
    plt.legend()
    plt.title('local ELM error : different M')
    plt.savefig('./result/2dim_different_M.pdf', dpi=100)
    '''
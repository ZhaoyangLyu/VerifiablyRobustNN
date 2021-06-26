import torch
import numpy as np
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn import Sequential, Conv2d, Linear, ReLU, LeakyReLU
from model_defs import Flatten, model_mlp_any
import torch.nn.functional as F
from itertools import chain
import logging

class BoundActivation(nn.Module):
    def __init__(self):
        # need to implement this method for different activations
        super(BoundActivation, self).__init__()
        # upper and lower bounds of the output
        # self.out_U = None
        # self.out_L = None
        # upper and lower bounds of the input
        self.lower_l = None
        self.upper_u = None

    def forward(self, x):
        # need to implement this method for different activations
        return x
        

    def update_neuron_status(self):
        # need to implement this method for different activations
        pass

    @staticmethod
    def get_line_params_from_two_points(x1, y1, x2, y2):
        # compute the line slope and intercept that pass through the points (x1, y1), (x2, y2)
        diff = x2-x1
        small_values = diff.abs() < 1e-6
        large_values = ~small_values
        diff = large_values.float() * diff + small_values.float() * 1e-6
        # diff = diff.abs().clamp(min=1e-6) * diff.sign()
        k = (y2-y1)/diff
        # k x1 + b = y1
        b = y1-k*x1
        
        return k, b

    def get_bound_lines(self, l, u):
        # need to implement this methed for different activations
        ku = None
        bu = None
        kl = None
        bl = None
        return ku,bu,kl,bl

    def interval_propagate(self, norm, h_U, h_L, eps):
        assert norm == np.inf
        guard_eps = 1e-5
        self.unstab = ((h_L < -guard_eps) & (h_U > guard_eps))
        # self.unstab indicates that this neuron's activation is unsure
        # stored upper and lower bounds will be used for backward bound propagation

        # this is the upper and lower bounds of the input of this relu layer
        self.upper_u = h_U
        self.lower_l = h_L 
        self.update_neuron_status()
        tightness_loss = self.unstab.sum()
        # tightness_loss = torch.min(h_U_unstab * h_U_unstab, h_L_unstab * h_L_unstab).sum()
        out_U = self.forward(self.upper_u)
        out_L = self.forward(self.lower_l)
        # self.out_U = out_U
        # self.out_L = out_L
        return norm, out_U, out_L, tightness_loss, tightness_loss, \
               (h_U < 0).sum(), (h_L > 0).sum()

    def linear_propagate(self, h_U, h_L, last_uA, last_ub, last_lA, last_lb): 
        # in this relu layer we assume the input is z and output is a: a = relu(z)
        # hU and hL are the upper and lower bounds of z
        # we already know z can be bounded by two linear functions of x
        # last_lA  x + last_lb <= z <= last_uA x + last_ub
        # this function finds two linear functions of x to bound a
        # this function returns uA, ubias, lA, lbias such that
        # uA x + ubias <= a <= lA x + lbias
        # we don't need to compute the closed form bounds of a
        # the bounds of the next layer's preactivation should be computed in BoundLinear and BoundConv

        # x is of shape (batch, x_shape)
        # last_uA and last_lA are of shape (batch, this_layer_shape, x_shape)
        # last_ub, last_lb, z, a are of the same shape (batch, this_layer_shape)

        # define this_layer_dim = products of elements in this_layer_shape
        # this_layer_shape may have multi dimensions

        # this is the upper and lower bounds of the input of this relu layer
        self.upper_u = h_U # shape (batch, this_layer_shape)
        self.lower_l = h_L # shape (batch, this_layer_shape)
        self.update_neuron_status()

        # pdb.set_trace()
        upper_d, upper_b, lower_d, lower_b = self.get_bound_lines(self.lower_l, self.upper_u)

        # detach bounding line parameters
        # usually we should not detach, detach can give us bad trained network
        if self.bound_opts.get("detach", False):
            upper_d = upper_d.detach()
            upper_b = upper_b.detach()
            lower_d = lower_d.detach()
            lower_b = lower_b.detach()

        
        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            if len(last_uA.shape) == 3:
                # if the layer before this layer is a linear layer
                # upper_d shape (batch, this_layer_shape)
                # last_uA shape (batch, this_layer_shape, x_shape)
                uA = upper_d.unsqueeze(2) * last_uA # (batch, this_layer_shape, x_shape)
            elif len(last_uA.shape) == 5:
            # if the layer before this layer is a conv layer
            # last_uA is of shape (batch, CHW, C_out, H_out, W_out)
            # upper_d has the shape (batch, C_out, H_out, W_out)
                uA = upper_d.unsqueeze(1) * last_uA # (batch, CHW, C_out, H_out, W_out)
            else:
                raise Exception('The shape of last_uA is %s, which is not correct.' % str(last_uA.shape))

            # last_ub shape (batch, this_layer_shape)
            # upper_b shape (batch, this_layer_shape)
            ubias = upper_d * last_ub + upper_b # (batch, this_layer_shape)

        if last_lA is not None:
            if len(last_lA.shape) == 3:
                # lower_d shape (batch, this_layer_shape)
                # last_lA shape (batch, this_layer_shape, x_shape)
                lA = lower_d.unsqueeze(2) * last_lA # (batch, this_layer_shape, x_shape)
            elif len(last_lA.shape) == 5:
                lA = lower_d.unsqueeze(1) * last_lA
            else:
                raise Exception('The shape of last_lA is %s, which is not correct.' % str(last_lA.shape))

            # last_lb shape (batch, this_layer_shape)
            lbias = lower_d * last_lb + lower_b # (batch, this_layer_shape)
        
        return uA, ubias, lA, lbias

    def bound_backward(self, last_uA, last_lA): 
        # in this relu layer we assume the input is z and output is a: a = relu(z)
        # we already know the quantity in interest, obj, can be bounded by two linear functions of a
        # last_uA a + last_ub <= obj <= last_lA a + last_lb
        # this function finds two linear functions of z to bound obj
        # this function returns uA, ubias, lA, lbias such that
        # uA * z + ubias + last_ub <= obj <= lA * z + lbias + last_lb

        # last_uA and last_lA are of shape (batch, obj_dim, this_layer_shape)
        # define this_layer_dim = products of elements in this_layer_shape
        # this_layer_shape may have multi dimensions
        
        upper_d, upper_b, lower_d, lower_b = self.get_bound_lines(self.lower_l, self.upper_u)
        # upper_d, upper_b, lower_d, lower_b are of shape (batch, this_layer_shape)
        upper_d = upper_d.unsqueeze(1) # of shape (batch, 1, this_layer_shape)   
        lower_d = lower_d.unsqueeze(1) # of shape (batch, 1, this_layer_shape)   

        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0) # shape (batch, obj_dim, this_layer_shape)
            if self.bound_opts.get("same-slope", False):
                raise Exception('This activation layer does not support same-slope yet, it only supports adaptive slope')
                # same upper_d and lower_d, no need to check the sign
                uA = upper_d * last_uA # shape (batch, obj_dim, this_layer_shape)
            else:
                neg_uA = last_uA.clamp(max=0)
                uA = upper_d * pos_uA + lower_d * neg_uA # shape (batch, obj_dim, this_layer_shape)
            
            pos_mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1) # shape (batch, obj_dim, this_layer_dim)
            # upper_b.view(upper_b.size(0), -1, 1) is of shape (batch, this_layer_dim, 1)
            ubias = pos_mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1) # of shape (batch, obj_dim)
            neg_mult_uA = neg_uA.view(last_uA.size(0), last_uA.size(1), -1) # shape (batch, obj_dim, this_layer_dim)
            ubias = ubias + neg_mult_uA.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1) # of shape (batch, obj_dim)
            
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            if self.bound_opts.get("same-slope", False):
                raise Exception('This activation layer does not support same-slope yet, it only supports adaptive slope')
                lA = uA if uA is not None else lower_d * last_lA
            else:
                pos_lA = last_lA.clamp(min=0) 
                lA = upper_d * neg_lA + lower_d * pos_lA

            neg_mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = neg_mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            pos_mult_lA = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = lbias + pos_mult_lA.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
        
        
        return uA, ubias, lA, lbias


class BoundGeneralStep(BoundActivation):
    def __init__(self, bound_opts, x1,y1,x2,y2,left_slope=0, right_slope=0):
        super(BoundGeneralStep, self).__init__()
        # this activation is a piecewise linear function
        # On the left, it passes (x1,y1) and has a slope of left_slope
        # In the middle, it passes (x1,y1) and (x2,y2)
        # On the right, it passes (x2,y2) and has a slope if right_slope
        self.ignore_right_step = False # this controls whether we ignore the right point
        # if we ignore the right point, this activation function will be like a leaky_relu function

        self.bound_opts = bound_opts

        self.left_slope = left_slope
        # when x<x1, the line left_slope*x1 + left_b = y1
        self.left_b = y1 - self.left_slope * x1

        self.right_slope = right_slope
        # when x>x2, the line right_slope*x2 + right_b = y2
        self.right_b = y2 - self.right_slope*x2

        self.middle_slope = (y2-y1)/(x2-x1)
        # when x1<x<x2, middle_slope * x1 + middle_b = y1
        self.middle_b = y1-self.middle_slope*x1
        
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

        self.upper_u = None # shape (batch, this_layer_shape)
        self.lower_l = None
        # the lower and upper bounds of the preactivation will be recorded
        # as self.upper_u and self.lower_l if interval_propagate or linear_propagate is called
        self.neuron_status = {'left_dead':None, 'left_unstable':None, 'unstable':None, 
                                'alive':None, 'right_unstable':None, 'right_dead':None}


    def forward(self,x):
        if self.ignore_right_step:
            minus = x<self.x1
            plus = ~minus
            out = (self.left_slope*x+self.left_b)*minus.float() + (self.middle_slope*x+self.middle_b)*plus.float()
        else:
            minus = x<self.x1
            plus = x>self.x2
            medium =  ~ (minus | plus)
            out = ((self.left_slope*x+self.left_b)*minus.float() + (self.middle_slope*x+self.middle_b)*medium.float() + 
                    (self.right_slope*x+self.right_b)*plus.float())
        return out

    def get_bound_lines(self, l, u):
        u = torch.max(u, l + 1e-6)
        # this ensures u>l
        # 3 cases for u
        # u<=x1, x1<u<=x2, u>x2
        # case 1: u<=x1, l can only be l<u<=x1, left dead
        # case 2.1: x1<u<=x2, l<x1, left unstable
        # case 2.2: x1<u<=x2, l>=x1, alive
        # case 3.1: u>x2, l<x1, unstable
        # case 3.2: u>x2, x1<=l<x2, right unstable
        # case 3.3: u>x2, l>=x2, right dead
        yl = self.forward(l)
        yu = self.forward(u)
        k ,b = self.get_line_params_from_two_points(l, yl, u, yu)
        x1 = self.x1
        if self.ignore_right_step:
            x2 = np.inf
        else:
            x2 = self.x2
        y1 = self.y1
        y2 = self.y2

        # left dead: l<u<=x1 
        left_dead = (u <= x1).float()
        left_dead_kl = self.left_slope # maybe a number or of shape (1, input shape). where l and u are of shape (batch, input shape)
        left_dead_bl = self.left_b
        left_dead_ku = self.left_slope
        left_dead_bu = self.left_b

        idx_x1_u = ((u-x1) > (x1-l)).float()
        idx_x1_l = 1-idx_x1_u

        # left unstable: l<x1, x1<u<=x2
        left_unstable = ((l<x1) * (u>x1) * (u<=x2)).float()
        if self.bound_opts.get('adaptive-lb', False) or self.bound_opts.get('left-adap_right-neg', False):
            left_unstable_kl = idx_x1_u * self.middle_slope + idx_x1_l * self.left_slope
            left_unstable_bl = idx_x1_u * self.middle_b + idx_x1_l * self.left_b
        elif self.bound_opts.get('neg-slope-lb', False):
            left_unstable_kl = self.left_slope
            left_unstable_bl = self.left_b
        else:
            print(self.bound_opts)
            raise Exception('You have not specified a valid lower bounding line choosing method')
        left_unstable_ku = k
        left_unstable_bu = b

        if not self.ignore_right_step:
            # unstable: l<x1, u>x2
            unstable = ((l<x1) * (u>x2)).float()
            if self.bound_opts.get('adaptive-lb', False) or self.bound_opts.get('left-adap_right-neg', False):
                k1u, b1u = self.get_line_params_from_two_points(x1, y1, u, yu)
                unstable_kl = idx_x1_u * k1u + idx_x1_l * self.left_slope
                unstable_bl = idx_x1_u * b1u + idx_x1_l * self.left_b

                k2l, b2l = self.get_line_params_from_two_points(l, yl, x2, y2)
                idx_x2_u = ((u-x2) > (x2-l)).float()
                idx_x2_l = 1-idx_x2_u
                unstable_ku = idx_x2_u * self.right_slope + idx_x2_l * k2l
                unstable_bu = idx_x2_u * self.right_b + idx_x2_l * b2l
            elif self.bound_opts.get('neg-slope-lb', False):
                unstable_kl = self.left_slope
                unstable_bl = self.left_b
                unstable_ku = self.right_slope
                unstable_bu = self.right_b
            else:
                print(self.bound_opts)
                raise Exception('You have not specified a valid lower bounding line choosing method')


        # alive: l>=x1, u<=x2
        alive = ((l>=x1) * (u<=x2)).float()
        alive_kl = self.middle_slope
        alive_bl = self.middle_b
        alive_ku = self.middle_slope
        alive_bu = self.middle_b

        if not self.ignore_right_step:
            # right unstable: x1<=l<x2, u>x2
            right_unstable = ((l>=x1) * (l<x2) * (u>x2)).float()
            if self.bound_opts.get('adaptive-lb', False):
                right_unstable_ku = idx_x2_u * self.right_slope + idx_x2_l * self.middle_slope
                right_unstable_bu = idx_x2_u * self.right_b + idx_x2_l * self.middle_b
            elif self.bound_opts.get('neg-slope-lb', False) or self.bound_opts.get('left-adap_right-neg', False):
                right_unstable_ku = self.right_slope
                right_unstable_bu = self.right_b
            else:
                print(self.bound_opts)
                raise Exception('You have not specified a valid lower bounding line choosing method')
            right_unstable_kl = k
            right_unstable_bl = b

            # right dead: l>=x2
            right_dead = (l>=x2).float()
            right_dead_ku = self.right_slope
            right_dead_bu = self.right_b
            right_dead_kl = self.right_slope
            right_dead_bl = self.right_b

        if self.ignore_right_step:
            ku = (left_dead * left_dead_ku + left_unstable * left_unstable_ku +
                alive * alive_ku)
            bu = (left_dead * left_dead_bu + left_unstable * left_unstable_bu +
                alive * alive_bu)
            kl = (left_dead * left_dead_kl + left_unstable * left_unstable_kl +
                alive * alive_kl)
            bl = (left_dead * left_dead_bl + left_unstable * left_unstable_bl +
                alive * alive_bl)
        else:
            ku = (left_dead * left_dead_ku + left_unstable * left_unstable_ku + unstable * unstable_ku +
                alive * alive_ku + right_unstable * right_unstable_ku + right_dead * right_dead_ku)
            bu = (left_dead * left_dead_bu + left_unstable * left_unstable_bu + unstable * unstable_bu +
                alive * alive_bu + right_unstable * right_unstable_bu + right_dead * right_dead_bu)
            kl = (left_dead * left_dead_kl + left_unstable * left_unstable_kl + unstable * unstable_kl +
                alive * alive_kl + right_unstable * right_unstable_kl + right_dead * right_dead_kl)
            bl = (left_dead * left_dead_bl + left_unstable * left_unstable_bl + unstable * unstable_bl +
                alive * alive_bl + right_unstable * right_unstable_bl + right_dead * right_dead_bl)

        
        return ku, bu, kl, bl

    def update_neuron_status(self, l=None, u=None):
        if l is None:
            l = self.lower_l
        if u is None:
            u = self.upper_u
        x1 = self.x1
        if self.ignore_right_step:
            x2 = np.inf
        else:
            x2 = self.x2
        # left dead: l<u<=x1 
        left_dead = (u <= x1).float().mean()
        # left unstable: l<x1, x1<u<=x2
        left_unstable = ((l<x1) * (u>x1) * (u<=x2)).float().mean()
        # unstable: l<x1, u>x2
        unstable = ((l<x1) * (u>x2)).float().mean()
        # alive: l>=x1, u<=x2
        alive = ((l>=x1) * (u<=x2)).float().mean()
        # right unstable: x1<=l<x2, u>x2
        right_unstable = ((l>=x1) * (l<x2) * (u>x2)).float().mean()
        # right dead: l>=x2
        right_dead = (l>=x2).float().mean()

        self.neuron_status['left_dead'] = left_dead
        self.neuron_status['left_unstable'] = left_unstable
        self.neuron_status['unstable'] = unstable
        self.neuron_status['alive'] = alive
        self.neuron_status['right_unstable'] = right_unstable
        self.neuron_status['right_dead'] = right_dead
        return 0

class BoundLeakyReLUStep(BoundGeneralStep):
    def __init__(self, bound_opts, slope=0, right=1, parameterize=False, parameterize_slope=False, shape=None):
        # this is a piece wise linear function
        # On the left, it passes (0,0) and has a slope of slope
        # In the middle, it passes (0,0) and (right,right) and has a slope of 1
        # On the right, it passes (right,right) and has a slope if slope

        # need to set bound_opts['activation'] = 'leaky_relu_step' if want to use this activation
        # also need to set the value for bound_opts['neg_slope'] as slope
        # set bound_opts['activation'] = 'param_leaky_relu_step' if want to parameterize the right turning point of this function
        super(BoundLeakyReLUStep, self).__init__(bound_opts, 0,0,right,right,left_slope=slope, right_slope=slope)
        self.right = right
        self.parameterize = parameterize
        self.parameter_pending_initialize = False
        self.parameterize_slope = parameterize_slope # indicate whether parameterize slope as well
        if parameterize:
            # shape should be the shape of the input of this layer without batch dimension
            if shape is None:
                self.parameter_pending_initialize = True
            else:
                self.init_parameter(shape)

        self.record_mean_activation = False
        self.include_minus_values = False
        self.mean_act = 0
        self.num_examples = 0

        self.rand_right = bound_opts.get('rand_right', False)
        self.random_magnitude = bound_opts.get('rand_magnitude', 1e-2)
        

    def init_parameter(self, shape, device=torch.device('cpu')):
        # shape should be the shape of the input of this layer without batch dimension
        self.right = nn.Parameter(torch.ones(shape, device=device).unsqueeze(0))
        if self.parameterize_slope:
            self.right_slope = nn.Parameter(torch.ones(shape, device=device).unsqueeze(0)*1e-3)
            self.left_slope = nn.Parameter(torch.ones(shape, device=device).unsqueeze(0)*1e-3)
            self.neuron_status['mean_right_slope'] = 1e-3
            self.neuron_status['mean_left_slope'] = 1e-3
        self.neuron_status['mean_right_point'] = 1
        
        self.update_parameter()

    def update_parameter(self):
        # update the position of the right turning point for every neuron
        random_right = self.rand_right
        random_magnitude = self.random_magnitude
        
        if random_right and self.training:
            self.right.data.clamp_(min=1e-4)
            right_mean = self.right.mean().item()
            noise = torch.normal(0, right_mean*random_magnitude, size=self.right.shape)
            noise = noise.to(self.right.device)
            rand_right = (self.right*1 + noise)
            rand_right.data.clamp_(min=1e-4)
            self.x2 = rand_right
            self.y2 = rand_right
            if self.parameterize_slope:
                self.right_slope.data.clamp_(min=0)
                self.left_slope.data.clamp_(min=0)
            self.right_b = (1 - self.right_slope)*rand_right
        else:
            self.right.data.clamp_(min=1e-4)
            self.x2 = self.right * 1
            self.y2 = self.right * 1
            if self.parameterize_slope:
                self.right_slope.data.clamp_(min=0)
                self.left_slope.data.clamp_(min=0)
            self.right_b = (1 - self.right_slope)*self.right

        # don't need to update left_b. because it is always 0
        # self.left_b = y1 - self.left_slope * x1
        
    def update_slope(self, slope):
        if self.parameterize_slope:
            raise Exception('The slope has already been parameterized, you can not reset it')
        self.left_slope = slope
        self.right_slope = slope
        self.right_b = (1 - self.right_slope)*self.right
        # self.right_b = self.y2 - self.right_slope*self.x2

    def forward(self, x):
        if self.parameter_pending_initialize:
            shape = x.shape[1:]
            self.init_parameter(shape, device = x.device)
            self.parameter_pending_initialize = False
        
        if self.parameterize:
            self.update_parameter()

        out = super(BoundLeakyReLUStep, self).forward(x)

        if self.record_mean_activation:
            if self.include_minus_values:
                if self.num_examples == 0:
                    self.mean_act = out.mean(dim=0)
                    self.num_examples = out.shape[0]
                else:
                    N = out.shape[0]
                    current_num_examples = self.num_examples + N
                    self.mean_act = (self.mean_act * (self.num_examples/current_num_examples) + 
                                    out.sum(dim=0)/current_num_examples)
                    self.num_examples = current_num_examples
            else:
                if isinstance(self.num_examples, int) and self.num_examples == 0:
                    plus = (out>0).float()
                    self.num_examples = plus.sum(dim=0)
                    self.mean_act = (out*plus).sum(dim=0) / (torch.clamp(self.num_examples, min=1))
                else:
                    plus = (out>0).float()
                    num_plus_examples = plus.sum(dim=0)
                    current_examples = self.num_examples + num_plus_examples
                    current_examples_p = torch.clamp(current_examples, min=1)
                    # self.mean_act = (self.mean_act * self.num_examples + (out*plus).sum(dim=0)) / current_examples_p
                    self.mean_act = (self.mean_act * (self.num_examples/current_examples_p) + 
                                        (out*plus).sum(dim=0)/current_examples_p)
                    self.num_examples = current_examples

        return out

    def use_mean_act_as_param(self):
        # self.right = nn.Parameter(torch.clamp(self.mean_act, min=1e-4))
        self.right.data = torch.clamp(self.mean_act, min=1e-4).unsqueeze(0)
        self.neuron_status['mean_right_point'] = self.right.mean()
        self.update_parameter()

    def update_neuron_status(self, l=None, u=None):
        if self.parameterize:
            self.neuron_status['mean_right_point'] = self.right.mean()
        if self.parameterize_slope:
            self.neuron_status['mean_right_slope'] = self.right_slope.mean()
            self.neuron_status['mean_left_slope'] = self.left_slope.mean()
        super(BoundLeakyReLUStep, self).update_neuron_status(l=None, u=None)

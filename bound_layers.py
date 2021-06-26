import torch
import numpy as np
from torch.nn import DataParallel
from torch.nn import Sequential, Conv2d, Linear, ReLU, LeakyReLU
from model_defs import Flatten, model_mlp_any
import torch.nn.functional as F
from itertools import chain
import logging

from bound_param_ramp import BoundLeakyReLUStep

logging.basicConfig(level=logging.INFO)
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BoundFlatten(torch.nn.Module):
    # this layer unwrap the input x to a 2D tensor
    def __init__(self, bound_opts=None):
        super(BoundFlatten, self).__init__()
        self.bound_opts = bound_opts

    def forward(self, x):
        # usually the input is of shape (batch,C,H,W)
        # then the output is of shape (batch, C*H*W)
        self.shape = x.size()[1:] # it's of shape (C,H,W)
        return x.view(x.size(0), -1)

    def interval_propagate(self, norm, h_U, h_L, eps):
        return norm, h_U.view(h_U.size(0), -1), h_L.view(h_L.size(0), -1), 0, 0, 0, 0

    def linear_propagate(self, last_uA, last_ub, last_lA, last_lb):
        # assume input of this layer is a, of shape (N, C_in, H_in, W_in)
        # output of this layer is z, of shape (N, C_in H_in W_in)
        # this function reshape last_uA, last_ub, last_lA, last_lb to their corresponding shape
        # x is of shape (batch, C, H, W), 
        # last_uA and last_lA are of shape (batch, CHW, C_in, H_in, W_in)
        # last_ub, last_lb are of the same shape (batch, C_in, H_in, W_in)
        batch = last_uA.shape[0]
        CHW = last_uA.shape[1]

        last_uA = last_uA.view(batch, CHW, -1) # shape (batch, CHW, C_in H_in W_in)
        last_uA = torch.transpose(last_uA, 1,2) # shape (batch, C_in H_in W_in, CHW)
        last_ub = last_ub.view(batch, -1) # shape (batch, C_in H_in W_in)

        last_lA = last_lA.view(batch, CHW, -1) # shape (batch, CHW, C_in H_in W_in)
        last_lA = torch.transpose(last_lA, 1,2) # shape (batch, C_in H_in W_in, CHW)
        last_lb = last_lb.view(batch, -1) # shape (batch, C_in H_in W_in)
        return last_uA, last_ub, last_lA, last_lb

    def bound_backward(self, last_uA, last_lA):
        # assume input of this layer is a, of shape (batch,C,H,W)
        # output is z, of shape (batch, CHW)
        # the objective we want to bound is obj, of shape (batch, obj_dim)
        # last_uA and last_lA is of shape (batch, obj_dim, C*H*W)
        # and they satisfy last_lA z + some bias <= obj <= last_uA z + some bias

        # we want to get uA and lA of shape (batch, obj_dim, C,H,W)
        # such that lA a + some bias <= obj <= uA a + some bias
        # where lA, uA reshape to (batch, obj_dim,        C*H*W)
        #            a reshape to (batch, C*H*W,              1)

        # uA and lA can be obtained by reshape last_uA and last_lA
        def _bound_oneside(A):
            if A is None:
                return None
            return A.view(A.size(0), A.size(1), *self.shape)
        if self.bound_opts.get("same-slope", False) and (last_uA is not None) and (last_lA is not None):
            new_bound = _bound_oneside(last_uA)
            return new_bound, 0, new_bound, 0
        else:
            return _bound_oneside(last_uA), 0, _bound_oneside(last_lA), 0

class BoundLinear(Linear):
    def __init__(self, in_features, out_features, bias=True, bound_opts=None):
        super(BoundLinear, self).__init__(in_features, out_features, bias)
        self.bound_opts = bound_opts

    @staticmethod
    def convert(linear_layer, bound_opts=None):
        # extract weight and other useful information from a linear layer
        l = BoundLinear(linear_layer.in_features, linear_layer.out_features, linear_layer.bias is not None, bound_opts)
        l.weight.data.copy_(linear_layer.weight.data)
        l.bias.data.copy_(linear_layer.bias.data)
        return l

    def bound_backward(self, last_uA, last_lA):
        # in this linear layer we assume the input is a and output is z: z = weight a + bias
        # we already know the quantity in interest, obj, can be bounded by two linear functions of z
        # last_uA * z + last_ub <= obj <= last_lA * z + last_lb
        # this function finds two linear functions of a to bound obj
        # this function returns uA, ubias, lA, lbias such that
        # uA * a + ubias + last_ub <= obj <= lA * a + lbias + last_lb

        # a is of shape (batch, in_dim)
        # z is of shape (batch, out_dim)
        # last_A is of shape (batch, obj_dim, out_dim)
        # weight is of shape (out_dim, in_dim)
        # bias is of shape (out_dim)
        def _bound_oneside(last_A, compute_A=True):
            if last_A is None:
                return None, 0
            logger.debug('last_A %s', last_A.size())
            # propagate A to the next layer
            if compute_A:
                # last_A shape (batch, obj_dim, out_dim)
                # weight is of shape (out_dim, in_dim)
                next_A = last_A.matmul(self.weight) # (batch, obj_dim, in_dim)
                logger.debug('next_A %s', next_A.size())
            else:
                next_A = None
            # compute the bias of this layer
            # sum_bias need to to added to the last bias to get the real bias
            # last_A shape (batch, obj_dim, out_dim)
            # self.bias of shape (out_dim)
            sum_bias = last_A.matmul(self.bias) # shape (batch, obj_dim)
            logger.debug('sum_bias %s', sum_bias.size())
            return next_A, sum_bias

        if self.bound_opts.get("same-slope", False) and (last_uA is not None) and (last_lA is not None):
            uA, ubias = _bound_oneside(last_uA, True)
            _, lbias = _bound_oneside(last_lA, False)
            lA = uA
        else:
            uA, ubias = _bound_oneside(last_uA)
            lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias

    @staticmethod
    def get_closed_form_bound(Au, bu, Al, bl, x0, p_norm, eps, x_U=None, x_L=None):
        # find the maximum value of Au x + bu and minimum value of Al x + bl
        # when x is within the l-p ball ||x-x0||_{p_norm} <= eps
        # x is of shape (batch, x_shape), x could be 2D or 4D, namely x could be flattened or original shape
        # x_U and x_L if exist, they are the same shape as x: (batch, x_shape)
        # Au is of shape (batch, out_dim, x_shape)
        # bu is of shape (batch, out_dim) 

        batch = x0.shape[0]

        if (p_norm != np.inf) or (x_U is None):
            # print('Use x0 and eps to compute closed form bound')
            if p_norm == 1:
                dual_norm = np.inf
            elif p_norm == np.inf:
                dual_norm = 1
            else:
                dual_norm = 1/(1-1/p_norm)

            x0_temp = x0.view(batch,-1).unsqueeze(2) # (batch, x_shape, 1)

            # this part may have problem, we should figure out
            # whether eps is for the original data or normalized data
            upper = Au.matmul(x0_temp).squeeze(2) + bu + eps*torch.norm(Au, p=dual_norm, dim=2)
            lower = Al.matmul(x0_temp).squeeze(2) + bl - eps*torch.norm(Al, p=dual_norm, dim=2)
            # upper and lower are of shape (batch, out_dim)
        else: # if norm=np.inf and x_U, x_L are not None
            # x_L, x_U maybe tighter than x0-eps, x0+eps
            # because we need to clamp x0-eps, x0+eps to the range [0,1]
            # before feed it to the network
            # print('Use x_L and x_U to compute closed form bound')
            x_U_temp = x_U.view(batch,-1).unsqueeze(2)
            x_L_temp = x_L.view(batch,-1).unsqueeze(2)
            # x_L <= x <= x_U
            # Au x + bu <= relu(Au) x_U + neg(Bu) x_L + bu
            Au_relu = torch.clamp(Au, min=0)
            Au_neg = torch.clamp(Au, max=0)
            upper = Au_relu.matmul(x_U_temp).squeeze(2) + Au_neg.matmul(x_L_temp).squeeze(2) + bu
            
            # Al x + bl >= relu(Al) x_L + neg(Bl) x_U + bl
            Al_relu = torch.clamp(Al, min=0)
            Al_neg = torch.clamp(Al, max=0)
            lower = Al_relu.matmul(x_L_temp).squeeze(2) + Al_neg.matmul(x_U_temp).squeeze(2) + bl
            
        return upper, lower
        
    def linear_propagate(self, last_uA, last_ub, last_lA, last_lb, x0, norm, eps, 
                        C = None, x_U=None, x_L=None): 
        # in this linear layer we assume the input is a and output is z: z = weight a + bias
        # we already know a can be bounded by two linear functions of x
        # last_lA  x + last_lb <= a <= last_uA x + last_ub
        # this function finds two linear functions of x to bound z (Cz if C is not None)
        # this function returns uA, ubias, lA, lbias such that
        # uA x + ubias <= z <= lA x + lbias
        # we also need to compute the closed form bounds of z

        # a is of shape (batch, in_dim)
        # z is of shape (batch, out_dim)
        # x is of shape (batch, x_shape), 
        # note that we assume x_shape is 1D, namely, x is flattened 
        # last_uA and last_lA are of shape (batch, in_dim, x_shape)
        # last_ub, last_lb, a are of the same shape (batch, in_dim)
        # weight is of shape (out_dim, in_dim)
        # bias is of shape (out_dim)

        # define this_layer_dim = products of elements in this_layer_shape
        # this_layer_shape may have multi dimensions
        if C is not None:
            # C is of shape (batch, obj_dim, out_dim)
            weight = C.matmul(self.weight) # shape (batch, obj_dim, in_dim)
            bias = C.matmul(self.bias) # shape (batch, obj_dim)
        else:
            # weight dimension (this_layer_shape, prev_layer_shape)
            weight = self.weight.unsqueeze(0) # (1, out_dim, in_dim)
            bias = self.bias.unsqueeze(0) # (1, out_dim)
        
        relu_W = weight.clamp(min=0) # (1, out_dim, in_dim) or (batch, obj_dim, in_dim)
        nega_W = weight.clamp(max=0) # (1, out_dim, in_dim) or (batch, obj_dim, in_dim)
        # last_A (batch, in_dim, x_shape)
        lA = relu_W.matmul(last_lA) + nega_W.matmul(last_uA)
        # lA (batch, out_dim, x_shape) or (batch, obj_dim, x_shape)

        # W shape (1, out_dim, in_dim) or (batch, obj_dim, in_dim)
        # last_b.unsqueeze(2) shape (batch, in_dim, 1)
        lbias = relu_W.matmul(last_lb.unsqueeze(2)).squeeze(2) + nega_W.matmul(last_ub.unsqueeze(2)).squeeze(2) + bias
        # lbias shape (batch, out_dim) or (batch, obj_dim)

        uA = relu_W.matmul(last_uA) + nega_W.matmul(last_lA)
        ubias = relu_W.matmul(last_ub.unsqueeze(2)).squeeze(2) + nega_W.matmul(last_lb.unsqueeze(2)).squeeze(2) + bias

        h_U, h_L = BoundLinear.get_closed_form_bound(uA, ubias, lA, lbias, x0, norm, eps,
                                            x_U=x_U, x_L=x_L)
        #h_L <= z <= h_U, are of shape (batch, out_dim) or (batch, obj_dim)

        return uA, ubias, lA, lbias, h_U, h_L

    def interval_propagate(self, norm, h_U, h_L, eps, C = None):
        # h_U and h_L should be of shape (batch, prev_layer_shape) 
        # they are upper and lower bound of previous layer 
        # merge the specification
        if C is not None:
            # after multiplication with C, we have (batch, output_shape, prev_layer_shape)
            # we have batch dimension here because of each example has different C
            weight = C.matmul(self.weight)
            bias = C.matmul(self.bias)
        else:
            # weight dimension (this_layer_shape, prev_layer_shape)
            weight = self.weight
            bias = self.bias

        if norm == np.inf:
            # Linf norm
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = weight.abs()
            if C is not None:
                center = weight.matmul(mid.unsqueeze(-1)) + bias.unsqueeze(-1)
                deviation = weight_abs.matmul(diff.unsqueeze(-1))
                # these have an extra (1,) dimension as the last dimension
                center = center.squeeze(-1)
                deviation = deviation.squeeze(-1)
            else:
                # fused multiply-add
                center = torch.addmm(bias, mid, weight.t())
                deviation = diff.matmul(weight_abs.t())
        else:
            # L2 norm
            h = h_U # h_U = h_L, and eps is used
            dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
            if C is not None:
                center = weight.matmul(h.unsqueeze(-1)) + bias.unsqueeze(-1)
                center = center.squeeze(-1)
            else:
                center = torch.addmm(bias, h, weight.t())
                # center = bias + h * weight.t()
                # .t() is transpose of the tensor
            deviation = weight.norm(dual_norm, -1) * eps

        upper = center + deviation
        lower = center - deviation
        
        return np.inf, upper, lower, 0, 0, 0, 0
            

class BoundConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, bound_opts=None):
        super(BoundConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, 
                stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bound_opts = bound_opts

    @staticmethod
    def convert(l, bound_opts=None):
        nl = BoundConv2d(l.in_channels, l.out_channels, l.kernel_size, l.stride, l.padding, l.dilation, l.groups, l.bias is not None, bound_opts)
        nl.weight.data.copy_(l.weight.data)
        nl.bias.data.copy_(l.bias.data)
        logger.debug(nl.bias.size())
        logger.debug(nl.weight.size())
        return nl

    def forward(self, input):
        output = super(BoundConv2d, self).forward(input)
        self.output_shape = output.size()[1:]
        self.input_shape = input.size()[1:]
        return output

    def bound_backward(self, last_uA, last_lA):
        # assume input is a, of shape (batch, C_in, H_in, W_in),
        # output is z, of shape (batch, C_out, H_out, W_out),
        # the objective we want to bound is obj, of shape (batch, obj_dim)
        # then last_uA and last_lA are of shape (batch, obj_dim, C_out, H_out, W_out)
        # and they satisfy last_lA z + some bias <= obj <= last_uA z + some bias
        # where last_lA, last_uA reshape to (batch, obj_dim,          C_out*H_out*W_out)
        #                      z reshape to (batch, C_out*H_out*W_out,                1)

        # we want to compute uA and lA such of shape (batch, obj_dim, C_in, H_in, W_in)
        # such that lA a + some bias <= obj <= uA a + some bias
        # where lA, uA reshape to (batch, obj_dim,        C_in*H_in*W_in)
        #            a reshape to (batch, C_in*H_in*W_in,              1)

        def _bound_oneside(last_A, compute_A=True):
            if last_A is None:
                return None, 0
            logger.debug('last_A %s', last_A.size())
            shape = last_A.size()
            # propagate A to the next layer, with batch concatenated together
            if compute_A:
                
                output_padding0 = int(self.input_shape[1]) - (int(self.output_shape[1]) - 1) * self.stride[0] + 2 * self.padding[0] - int(self.weight.size()[2])
                output_padding1 = int(self.input_shape[2]) - (int(self.output_shape[2]) - 1) * self.stride[1] + 2 * self.padding[1] - int(self.weight.size()[3]) 
                next_A = F.conv_transpose2d(last_A.view(shape[0] * shape[1], *shape[2:]), self.weight, None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups, output_padding=(output_padding0, output_padding1))
                next_A = next_A.view(shape[0], shape[1], *next_A.shape[1:])
                logger.debug('next_A %s', next_A.size())
            else:
                next_A = False
            logger.debug('bias %s', self.bias.size())
            # dot product, compute the bias of this layer, do a dot product
            sum_bias = (last_A.sum((3,4)) * self.bias).sum(2)
            logger.debug('sum_bias %s', sum_bias.size()) 
            return next_A, sum_bias
        # if the slope is the same (Fast-Lin) and both matrices are given, only need to compute one of them
        if self.bound_opts.get("same-slope", False) and (last_uA is not None) and (last_lA is not None):
            uA, ubias = _bound_oneside(last_uA, True)
            _, lbias = _bound_oneside(last_lA, False)
            lA = uA
        else:
            uA, ubias = _bound_oneside(last_uA)
            lA, lbias = _bound_oneside(last_lA)
        return uA, ubias, lA, lbias

    def interval_propagate(self, norm, h_U, h_L, eps):
        if norm == np.inf:
            mid = (h_U + h_L) / 2.0
            diff = (h_U - h_L) / 2.0
            weight_abs = self.weight.abs()
            deviation = F.conv2d(diff, weight_abs, None, self.stride, self.padding, self.dilation, self.groups)
        else:
            # L2 norm
            mid = h_U
            logger.debug('mid %s', mid.size())
            # TODO: consider padding here?
            deviation = torch.mul(self.weight, self.weight).sum((1,2,3)).sqrt() * eps
            logger.debug('weight %s', self.weight.size())
            logger.debug('deviation %s', deviation.size())
            deviation = deviation.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            logger.debug('unsqueezed deviation %s', deviation.size())
        center = F.conv2d(mid, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        logger.debug('center %s', center.size())
        upper = center + deviation
        lower = center - deviation
        return np.inf, upper, lower, 0, 0, 0, 0

    def linear_propagate(self, last_uA, last_ub, last_lA, last_lb, x0, norm, eps, 
                        C = None, x_U=None, x_L=None): 
        # in this linear layer we assume the input is a and output is z: z = conv(a) + bias
        # we already know a can be bounded by two linear functions of x
        # last_lA  x + last_lb <= a <= last_uA x + last_ub
        # this function finds two linear functions of x to bound z
        # this function returns uA, ubias, lA, lbias such that
        # uA x + ubias <= z <= lA x + lbias
        # we also need to compute the closed form bounds of z
        # C has no use in this function, C is only used for the last layer, while a conv layer will never be the last layer of a network


        # a is of shape (batch, C_in, H_in, W_in)
        # z is of shape (batch, C_out, H_out, W_out)
        # x is of shape (batch, C, H, W), 
        # last_uA and last_lA are of shape (batch, CHW, C_in, H_in, W_in)
        # last_ub, last_lb, a are of the same shape (batch, C_in, H_in, W_in)
        # weight is of shape (C_out, C_in, kernel_size[0], kernel_size[1])
        # bias is of shape (C_out)

        # define this_layer_dim = products of elements in this_layer_shape
        # this_layer_shape may have multi dimensions
        weight = self.weight
        bias = self.bias
        
        relu_W = weight.clamp(min=0) 
        nega_W = weight.clamp(max=0)

        batch = last_uA.shape[0]
        CHW = last_uA.shape[1]
        C_in = last_uA.shape[2]
        H_in = last_uA.shape[3]
        W_in = last_uA.shape[4]
        
        
        lA = (F.conv2d(last_lA.reshape(-1,C_in,H_in,W_in), relu_W, bias=None, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups)
            + F.conv2d(last_uA.reshape(-1,C_in,H_in,W_in), nega_W, bias=None, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups))
        # lA now is of shape (batch*CHW, C_out, H_out, W_out)
        C_out = lA.shape[1]
        H_out = lA.shape[2]
        W_out = lA.shape[3]
        lA = lA.view(batch, CHW, C_out, H_out, W_out)

        
        lbias = (F.conv2d(last_lb, relu_W, bias=bias, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups)
               + F.conv2d(last_ub, nega_W, bias=None, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups))


        uA = (F.conv2d(last_uA.reshape(-1,C_in,H_in,W_in), relu_W, bias=None, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups)
            + F.conv2d(last_lA.reshape(-1,C_in,H_in,W_in), nega_W, bias=None, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups))
        uA = uA.view(batch, CHW, C_out, H_out, W_out)

        
        ubias = (F.conv2d(last_ub, relu_W, bias=bias, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups)
               + F.conv2d(last_lb, nega_W, bias=None, stride=self.stride, padding=self.padding, 
                    dilation=self.dilation, groups=self.groups))

        h_U, h_L = BoundConv2d.get_closed_form_bound(uA, ubias, lA, lbias, x0, norm, eps,
                                            x_U=x_U, x_L=x_L)

        return uA, ubias, lA, lbias, h_U, h_L

    @staticmethod
    def get_closed_form_bound(Au, bu, Al, bl, x0, p_norm, eps, x_U=None, x_L=None):
        # find the maximum value of Au x + bu and minimum value of Al x + bl
        # when x is within the l-p ball ||x-x0||_{p_norm} <= eps
        # x is of shape (batch, C,H,W)
        # x_U and x_L if exist, they are the same shape as x
        # Au,Al are of shape (batch, CHW, C_out, H_out, W_out)
        # bu,bl are of shape (batch, C_out, H_out, W_out) 
        
        batch = Au.shape[0]
        CHW = Au.shape[1]
        C_out = Au.shape[2]
        H_out = Au.shape[3]
        W_out = Au.shape[4]

        Au_temp = Au.view(batch, CHW, -1) # shape (batch, CHW, C_out H_out W_out)
        Au_temp = torch.transpose(Au_temp, 1, 2) # shape (batch, C_out H_out W_out, CHW)
        Al_temp = Al.view(batch, CHW, -1) # shape (batch, CHW, C_out H_out W_out)
        Al_temp = torch.transpose(Al_temp, 1, 2) # shape (batch, C_out H_out W_out, CHW)

        if (p_norm != np.inf) or (x_U is None):
            # print('Use x0 and eps to compute closed form bound')
            if p_norm == 1:
                dual_norm = np.inf
            elif p_norm == np.inf:
                dual_norm = 1
            else:
                dual_norm = 1/(1-1/p_norm)

            x0_temp = x0.view(batch, -1).unsqueeze(2) # (batch, CHW, 1)

            # this part may have problem, we should figure out
            # whether eps is for the original data or normalized data
            upper = Au_temp.matmul(x0_temp).squeeze(2) + eps*torch.norm(Au_temp, p=dual_norm, dim=2) #shape (batch, C_out H_out W_out)
            upper = upper.view(batch, C_out, H_out, W_out) + bu
            lower = Al_temp.matmul(x0_temp).squeeze(2) - eps*torch.norm(Al_temp, p=dual_norm, dim=2) #shape (batch, C_out H_out W_out)
            lower = lower.view(batch, C_out, H_out, W_out) + bl
            # upper and lower are of shape (batch, C_out, H_out, W_out)
        else: # if norm=np.inf and x_U, x_L are not None
            # x_L, x_U maybe tighter than x0-eps, x0+eps
            # because we need to clamp x0-eps, x0+eps to the range [0,1]
            # before feed it to the network
            # print('Use x_L and x_U to compute closed form bound')
            x_U_temp = x_U.view(batch, -1).unsqueeze(2) # (batch, CHW, 1)
            x_L_temp = x_L.view(batch, -1).unsqueeze(2) # (batch, CHW, 1)
            # x_L <= x <= x_U
            # Au x + bu <= relu(Au) x_U + neg(Bu) x_L + bu
            Au_relu = torch.clamp(Au_temp, min=0) # shape (batch, C_out H_out W_out, CHW)
            Au_neg = torch.clamp(Au_temp, max=0) # shape (batch, C_out H_out W_out, CHW)
            upper = Au_relu.matmul(x_U_temp).squeeze(2) + Au_neg.matmul(x_L_temp).squeeze(2) # shape (batch, C_out H_out W_out)
            upper = upper.view(batch, C_out, H_out, W_out) + bu # shape (batch, C_out, H_out, W_out)
            
            # Al x + bl >= relu(Al) x_L + neg(Bl) x_U + bl
            Al_relu = torch.clamp(Al_temp, min=0)
            Al_neg = torch.clamp(Al_temp, max=0)
            lower = Al_relu.matmul(x_L_temp).squeeze(2) + Al_neg.matmul(x_U_temp).squeeze(2) # shape (batch, C_out H_out W_out)
            lower = lower.view(batch, C_out, H_out, W_out) + bl # shape (batch, C_out, H_out, W_out)
            
        return upper, lower

def get_linear_bound_for_relu(l, u, bound_opts):
    # This function finds bounding lines for ReLU activation in the interval [l ,u]
    # bound_opts is a dictionary could contain keys use-constant, same-slope, zero-lb, one-lb, 
    # only one of the keys should have value True
    # if use-constant, we choose both boundling lines with 0 slopes at any cases
    # elif same-slope, we choose tight upper bounding line, lower bounding line with same slope when l<0 and u>0
    # elif zero-lb, we choose tight upper bounding line, lower bounding line with 0 slope when l<0 and u>0
    # elif one-lb, we choose tight upper bounding line, lower bounding line with 1 when l<0 and u>0
    # else, we choose tight upper bounding line, lower bounding lines with adaptive slope
    # except for use-constant, other choices don't affect how we choose bounding lines when l>=0 or u<=0
    # in these cases, we always chooose tightest upper and lower bounding lines
    device = l.device
    # don't change how to initialize them here
    ku = torch.zeros(u.shape, device = device)
    bu = torch.zeros(u.shape, device = device)
    kl = torch.zeros(l.shape, device = device)
    bl = torch.zeros(l.shape, device = device)
    if bound_opts.get('use-constant', False):
        bu = torch.clamp(u, min=0)
        bl = torch.clamp(l, min=0)
        # print('has use constant')
        return kl, bl, ku, bu

    # case u<=0, the 0 initialization already satisfy this case

    # case l>=0
    idx = (l>=0)
    kl[idx] = 1
    ku[idx] = 1
    # bl and kl is 0

    # case l<0 and u>0
    idx = (l<0) * (u>0)

    k = (u / (u-l))[idx]
    # k u + b = u -> b = (1-k) * u
    b = (1-k) * u[idx]

    ku[idx] = k
    bu[idx] = b
    # bl already 0
    # kl should be between 0 and 1
    if bound_opts.get('same-slope', False): # parallel to the upper line
        kl[idx] = k
    elif bound_opts.get('zero-lb', False): # always use 0 slope
        pass
        # kl[idx] = 0 # kl is initialized with 0, don't need to redo this
    elif bound_opts.get('one-lb', False): # always use 1 slope
        kl[idx] = 1
    elif bound_opts.get('adaptive-lb', False): # use adaptive
        u_geq_l = (u.abs()>=l.abs())
        new_idx = idx * u_geq_l
        kl[new_idx] = 1
        # new_idx = idx * (1-u_geq_l)
        # kl[new_idx] = 0 # kl is initialized with 0, don't need to redo this
    else:
        print('bound_opts:', bound_opts)
        raise Exception('bound-opts not supported')
    return kl, bl, ku, bu

class BoundReLU(ReLU):
    def __init__(self, prev_layer, inplace=False, bound_opts=None):
        super(BoundReLU, self).__init__(inplace)
        # ReLU needs the previous layer's bounds
        # self.prev_layer = prev_layer
        self.bound_opts = bound_opts
        self.upper_u = None # shape (batch, this_layer_shape)
        self.lower_l = None
        # the lower and upper bounds of the preactivation will be recorded
        # as self.upper_u and self.lower_l if interval_propagate or linear_propagate is called
        self.dead = None
        self.alive = None
        self.unstable = None

        # assume input of this relu layer is z and output is a: a = relu(z)
        # self.alpha_l = None
        # self.beta_l = None
        # self.alpha_u = None
        # self.beta_u = None
        # these quantities records the linear functions of x to bound z
        # alpha_l * z + beta_l <= z <= alpha_u * z + beta_u
        # For relu between linear layers
        # z is of shape (batch, n, 1)
        # x is of shape (batch, n0, 1)
        # alpha is of shape (batch, n, n0n)
        # beta is of shape (batch, n, 1)
        # In reality, those dimensions of width 1 may be squeezed

    
    ## Convert a ReLU layer to BoundReLU layer
    # @param act_layer ReLU layer object
    # @param prev_layer Pre-activation layer, used for get preactivation bounds
    def update_neuron_status(self):
        self.dead = (self.upper_u<=0).float().mean()
        self.alive = (self.lower_l>=0).float().mean()
        self.unstable = ((self.lower_l<0) * (self.upper_u>0)).float().mean()
    @staticmethod
    def convert(act_layer, prev_layer, bound_opts=None):
        l = BoundReLU(prev_layer, act_layer.inplace, bound_opts)
        return l

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
        return norm, F.relu(h_U), F.relu(h_L), tightness_loss, tightness_loss, \
               (h_U < 0).sum(), (h_L > 0).sum()

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
        lb_r = self.lower_l.clamp(max=0) # shape (batch, this_layer_shape), same as a or z
        ub_r = self.upper_u.clamp(min=0) # shape (batch, this_layer_shape), same as a or z
        # this step guarantees upper_d = 0, upper_b=0 if upper_u <= 0
        # upper_d = 1, upper_b=0 if lower_l >=0

        # avoid division by 0 when both lb_r and ub_r are 0
        ub_r = torch.max(ub_r, lb_r + 1e-8)

        # CROWN upper and lower linear bounds
        upper_d = ub_r / (ub_r - lb_r)
        upper_b = - lb_r * upper_d
        # note that there is no lower_b because the lower bounding line always passes the origin
        
        upper_d = upper_d.unsqueeze(1) # shape (batch, 1, this_layer_shape)
        if self.bound_opts.get("same-slope", False) or self.bound_opts.get("backward_same-slope", False):
            # the same slope for upper and lower
            lower_d = upper_d
        elif self.bound_opts.get("zero-lb", False) or self.bound_opts.get("backward_zero-lb", False):
            # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
            lower_d = (upper_d >= 1.0).float()
        elif self.bound_opts.get("one-lb", False) or self.bound_opts.get("backward_one-lb", False):
            # Always use slope 1 as lower bound
            lower_d = (upper_d > 0.0).float()
        elif self.bound_opts.get("adaptive-lb", False) or self.bound_opts.get("backward_adaptive-lb", False):
            lower_d = (upper_d > 0.5).float()
        else:
            raise Exception('The bounding line choice is not supported or you have not specified a bounding line choice method')
        uA = lA = None
        ubias = lbias = 0
        # Choose upper or lower bounds based on the sign of last_A
        if last_uA is not None:
            pos_uA = last_uA.clamp(min=0) # shape (batch, obj_dim, this_layer_shape)
            if self.bound_opts.get("same-slope", False):
                # same upper_d and lower_d, no need to check the sign
                uA = upper_d * last_uA # shape (batch, obj_dim, this_layer_shape)
            else:
                neg_uA = last_uA.clamp(max=0)
                uA = upper_d * pos_uA + lower_d * neg_uA # shape (batch, obj_dim, this_layer_shape)
            mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1) # shape (batch, obj_dim, this_layer_dim)
            ubias = mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1) # of shape (batch, obj_dim)
            # upper_b.view(upper_b.size(0), -1, 1) is of shape (batch, this_layer_dim, 1)
            # note that there is no lower_b because the lower bounding line always passes the origin
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            if self.bound_opts.get("same-slope", False) or self.bound_opts.get("backward_same-slope", False):
                lA = uA if uA is not None else lower_d * last_lA
            else:
                pos_lA = last_lA.clamp(min=0) 
                lA = upper_d * neg_lA + lower_d * pos_lA
            mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            # note that there is no lower_b because the lower bounding line always passes the origin
            # lbias and ubias need to to added to the last bias to get the real bias
        return uA, ubias, lA, lbias

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

        if self.bound_opts.get("use-constant", False):
            # avoid division by 0 when both lb_r and ub_r are 0
            h_U_new = torch.max(h_U, h_L + 1e-8)

            # CROWN upper and lower linear bounds
            lower_d, lower_b, upper_d, upper_b = get_linear_bound_for_relu(
                h_L, h_U_new, self.bound_opts)
        
        else:
            lb_r = self.lower_l.clamp(max=0) # shape (batch, this_layer_shape), same as a or z
            ub_r = self.upper_u.clamp(min=0) # shape (batch, this_layer_shape), same as a or z
            
            ub_r = torch.max(ub_r, lb_r + 1e-8)
            upper_d = ub_r / (ub_r - lb_r)
            upper_b = - lb_r * upper_d
            lower_b = torch.Tensor([0]).to(upper_d.device)
            if self.bound_opts.get("same-slope", False) or self.bound_opts.get("lbp_same-slope", False):
                # the same slope for upper and lower
                lower_d = upper_d
            elif self.bound_opts.get("zero-lb", False) or self.bound_opts.get("lbp_zero-lb", False):
                # Always use slope 0 as lower bound. Any value between 0 and 1 is a valid lower bound for CROWN
                lower_d = (upper_d >= 1.0).float()
            elif self.bound_opts.get("one-lb", False) or self.bound_opts.get("lbp_one-lb", False):
                # Always use slope 1 as lower bound
                lower_d = (upper_d > 0.0).float()
            elif self.bound_opts.get("adaptive-lb", False) or self.bound_opts.get("lbp_adaptive-lb", False):
                lower_d = (upper_d > 0.5).float()
            else:
                raise Exception('The bounding line choice is not supported or you have not specified a bounding line choice method')
        # detach bounding line parameters
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


class BoundLeakyReLU(LeakyReLU):
    def __init__(self, neg_slope=0.01, inplace=False, bound_opts=None, shape=None):
        super(BoundLeakyReLU, self).__init__(neg_slope, inplace)
        self.bound_opts = bound_opts
        self.neg_slope = neg_slope
        # need to set bound_opts['activation'] = 'hard_tanh' if want to use this activation
        # also need to set the value for bound_opts['neg_slope']
        
        self.upper_u = None # shape (batch, this_layer_shape)
        self.lower_l = None
        # the lower and upper bounds of the preactivation will be recorded
        # as self.upper_u and self.lower_l if interval_propagate or linear_propagate is called
        self.dead = None
        self.alive = None
        self.unstable = None

        if self.bound_opts.get('param-all-lb', False) or self.bound_opts.get('param-unstable-lb', False) or self.bound_opts.get('param-all-lb-tight', False):
            # shape is the shape of input of this layer, with out batch dimension
            self.kl = torch.rand(shape)
            self.kl = self.kl.clamp_(min=self.neg_slope)
            self.kl = self.kl.unsqueeze(0)
            self.kl = torch.nn.Parameter(self.kl)

    def update_slope(self, slope):
        self.neg_slope = slope

    def forward(self, x):
        out = F.leaky_relu(x, negative_slope=self.neg_slope)
        return out

    def update_neuron_status(self):
        self.dead = (self.upper_u<=0).float().mean()
        self.alive = (self.lower_l>=0).float().mean()
        self.unstable = ((self.lower_l<0) * (self.upper_u>0)).float().mean()

    @staticmethod
    def get_line_params_from_two_points(x1, y1, x2, y2):
        # compute the line slope and intercept that pass through the points (x1, y1), (x2, y2)
        diff = x2-x1
        small_values = diff.abs() < 1e-6
        large_values = ~small_values
        diff = large_values.float() * diff + small_values.float() * 1e-6

        k = (y2-y1)/diff
        # k x1 + b = y1
        # b = y1-k*x1
        # k x2 + b = y2
        b = y2-k*x2

        # if we assume x2>=x1, y2>=x1
        # we can guarantee the line k x + b is always above the two input points even if diff.abs() < 1e-6 
        return k, b

    def get_bound_lines(self, l, u):
        u = torch.max(u, l + 1e-6)
        

        yl = F.leaky_relu(l, negative_slope=self.neg_slope)
        yu = F.leaky_relu(u, negative_slope=self.neg_slope)
        ku, bu = self.get_line_params_from_two_points(l, yl, u, yu)

        alive = (l>=0).float()
        dead = (u<=0).float()
        unstable = ((l<0) * (u>0)).float()

        ku = self.neg_slope * dead + ku*unstable + alive
        bu = bu*unstable # because when a neuron is dead or alive, bu=0 

        if self.bound_opts.get("same-slope", False):
            # the same slope for upper and lower
            # this formulation is valid in any case: the neuron is dead, unstable, or alive
            kl = ku
            # bl = torch.Tensor([0]).to(ku.device)
            bl = torch.zeros(bu.shape).to(bu.device)

        elif self.bound_opts.get("zero-lb", False):
            # this is the tight strategy stated in the paper
            # Use slope self.neg_slope as lower slope when a neuron is dead or unstable, use 1 when it's alive
            kl = alive + (1-alive)*self.neg_slope
            # bl = torch.Tensor([0]).to(ku.device)
            bl = torch.zeros(bu.shape).to(bu.device)

        elif self.bound_opts.get("always-neg-slope-lb", False):
            # Always use slope self.neg_slope as lower slope, even when it's alive
            # when it's dead or unstable, bl=0
            # when it's alive, it passes (l,yl). yl = kl l + bl, bl = yl - kl l
            kl = torch.ones(l.shape).to(l.device)*self.neg_slope
            bl = torch.zeros(l.shape).to(l.device)
            bl = bl + alive * (yl-kl*l)

        elif self.bound_opts.get("always-zero-lb", False):
            # Always use slope 0 as lower slope
            kl = torch.zeros(l.shape).to(l.device)
            bl = yl
            
        elif self.bound_opts.get("one-lb", False):
            # Use slope 1 as lower slope when a neuron is alive or unstable, use self.neg_slope when it's dead
            kl = dead * self.neg_slope + (1-dead)
            # bl = torch.Tensor([0]).to(ku.device)
            bl = torch.zeros(bu.shape).to(bu.device)

        elif self.bound_opts.get("use-constant", False):
            kl = torch.zeros(u.shape).to(u.device)
            ku = torch.zeros(u.shape).to(u.device)
            bl = yl
            bu = yu

        elif self.bound_opts.get("adaptive-lb", False):
            # adaptive lower slope as default
            # this formulation is only valid when self.neg_slope is small
            u_big = u>(-l)
            l_big = ~u_big
            
            kl = (dead + unstable*l_big.float()) * self.neg_slope + (alive+unstable*u_big.float())
            # bl = torch.Tensor([0]).to(ku.device)
            bl = torch.zeros(bu.shape).to(bu.device)
        
        elif self.bound_opts.get('param-all-lb', False):
            # always use self.kl as lower bounding line no matter what status the neuron is in
            # and always use bl=0
            self.kl.data.clamp_(min=self.neg_slope, max=1)
            kl = self.kl
            # bl = torch.Tensor([0]).to(ku.device)
            bl = torch.zeros(bu.shape).to(bu.device)

        elif self.bound_opts.get('param-unstable-lb', False):
            # only use self.kl as lower bounding line for unstable neurons, use tight lower bounding line in other cases
            self.kl.data.clamp_(min=self.neg_slope, max=1)
            kl = self.kl*unstable + self.neg_slope*dead + 1*alive
            # bl = torch.Tensor([0]).to(ku.device)
            bl = torch.zeros(bu.shape).to(bu.device)

        elif self.bound_opts.get('param-all-lb-tight', False):
            # always use self.kl as lower bounding line no matter what status the neuron is in
            # but use largest possible bl
            self.kl.data.clamp_(min=self.neg_slope, max=1)
            kl = self.kl
            bl = torch.zeros(kl.shape).to(kl.device)
            # kl x + bl pass u,yu when the neuron is dead, yu = kl u + bl
            bl = bl + dead * (yu-kl*u)

            # kl x + bl pass l,yl when the neuron is alive, yl = kl l + bl
            bl = bl + alive * (yl-kl*l)
            # bl = 0 when the neuron is unstable, nothing need to be done 
        elif 'bound_specification' in self.bound_opts.keys():
            bound_spec = self.bound_opts['bound_specification']
            # bound_spec has the form 'tt-tc-tt'
            bound_s = bound_spec.split('-')
            # dead neurons
            if bound_s[0][0] == 't': # dead upper bounding line
                ku_dead = self.neg_slope
                bu_dead = 0
            elif bound_s[0][0] == 'c':
                ku_dead = 0
                bu_dead = yu
            else:
                raise Exception('bound specification elements must be t or c, but got', bound_spec)
            if bound_s[0][1] == 't': # dead lower bounding line
                kl_dead = self.neg_slope
                bl_dead = 0
            elif bound_s[0][1] == 'c':
                kl_dead = 0
                bl_dead = yl
            else:
                raise Exception('bound specification elements must be t or c, but got', bound_spec)
            
            # unstable neurons
            if bound_s[1][0] == 't': # unstable upper bounding line
                ku_unstable = ku
                bu_unstable = bu
            elif bound_s[1][0] == 'c':
                ku_unstable = 0
                bu_unstable = yu
            else:
                raise Exception('bound specification elements must be t or c, but got', bound_spec)
            if bound_s[1][1] == 't': # unstable lower bounding line
                kl_unstable = self.neg_slope
                bl_unstable = 0
            elif bound_s[1][1] == 'c':
                kl_unstable = 0
                bl_unstable = yl
            else:
                raise Exception('bound specification elements must be t or c, but got', bound_spec)

            # alive neurons
            if bound_s[1][0] == 't': # alive upper bounding line
                ku_alive = 1
                bu_alive = 0
            elif bound_s[1][0] == 'c':
                ku_alive = 0
                bu_alive = yu
            else:
                raise Exception('bound specification elements must be t or c, but got', bound_spec)
            if bound_s[1][1] == 't': # alive lower bounding line
                kl_alive = 1
                bl_alive = 0
            elif bound_s[1][1] == 'c':
                kl_alive = 0
                bl_alive = yl
            else:
                raise Exception('bound specification elements must be t or c, but got', bound_spec)

            ku = dead*ku_dead + unstable*ku_unstable + alive*ku_alive
            bu = dead*bu_dead + unstable*bu_unstable + alive*bu_alive

            kl = dead*kl_dead + unstable*kl_unstable + alive*kl_alive
            bl = dead*bl_dead + unstable*bl_unstable + alive*bl_alive

        else:
            raise Exception('You have not specified a valid method to choose lower bounding line.\n bound_opts: %s' % bound_opts)
        
        # if True in torch.isnan(ku) or True in torch.isnan(bu) or True in torch.isnan(kl) or True in torch.isnan(bl):
        #     pdb.set_trace()
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
        out_U = F.leaky_relu(h_U, negative_slope=self.neg_slope)
        out_L = F.leaky_relu(h_L, negative_slope=self.neg_slope)
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

        upper_d, upper_b, lower_d, lower_b = self.get_bound_lines(h_L, h_U)

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
            neg_uA = last_uA.clamp(max=0)
            uA = upper_d * pos_uA + lower_d * neg_uA # shape (batch, obj_dim, this_layer_shape)
            
            pos_mult_uA = pos_uA.view(last_uA.size(0), last_uA.size(1), -1) # shape (batch, obj_dim, this_layer_dim)
            # upper_b.view(upper_b.size(0), -1, 1) is of shape (batch, this_layer_dim, 1)
            ubias = pos_mult_uA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1) # of shape (batch, obj_dim)
            neg_mult_uA = neg_uA.view(last_uA.size(0), last_uA.size(1), -1) # shape (batch, obj_dim, this_layer_dim)
            ubias = ubias + neg_mult_uA.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1) # of shape (batch, obj_dim)
            
        if last_lA is not None:
            neg_lA = last_lA.clamp(max=0)
            pos_lA = last_lA.clamp(min=0) 
            lA = upper_d * neg_lA + lower_d * pos_lA

            neg_mult_lA = neg_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = neg_mult_lA.matmul(upper_b.view(upper_b.size(0), -1, 1)).squeeze(-1)
            pos_mult_lA = pos_lA.view(last_lA.size(0), last_lA.size(1), -1)
            lbias = lbias + pos_mult_lA.matmul(lower_b.view(lower_b.size(0), -1, 1)).squeeze(-1)
        return uA, ubias, lA, lbias

class BoundSequential(Sequential):
    def __init__(self, *args):
        super(BoundSequential, self).__init__(*args)
        self.contain_parameterized_act = False 
        self.parameterized_act_keys = [] # record keys of parameterized activation function modules
        self.contain_pending_init_parameters = False
        # indicate whether there are activation layers that contain parameters that need to be initialized
        for k in self._modules.keys():
            possible_param_act = isinstance(self._modules[k], BoundLeakyReLUStep)
            if possible_param_act and self._modules[k].parameterize == True:
                self.contain_parameterized_act = True
                self.parameterized_act_keys.append(k)
                if self._modules[k].parameter_pending_initialize:
                    self.contain_pending_init_parameters = True

    ## Convert a Pytorch model to a model with bounds
    # @param sequential_model Input pytorch model
    # @return Converted model

    def reset_ignore_right_step(self, value):
        for k in self._modules.keys():
            if isinstance(self._modules[k], BoundLeakyReLUStep):
                self._modules[k].ignore_right_step = value
        return 0

    def reset_bound_opts(self, opts_keys, opts_values, bound_opts=None):
        # opts_keys and opts_values are lists
        # they contain keys and values in bound_opts that we want to reset
        # if bound_opts is not None, we directly set it to bound_opts 
        for k in self._modules.keys():
            if bound_opts is None:
                for i, opts_k in enumerate(opts_keys):
                    self._modules[k].bound_opts[opts_k] = opts_values
            else:
                self._modules[k].bound_opts = bound_opts
        return 0
    @staticmethod
    def convert(sequential_model, bound_opts=None):
        # bound_opts is a dict and it looks like this 
        # {'same-slope': False, 'zero-lb': False, 'one-lb': False}
        layers = []
        if isinstance(sequential_model, Sequential):
            seq_model = sequential_model
        else:
            seq_model = sequential_model.module
        for l in seq_model:
            if isinstance(l, Linear):
                layers.append(BoundLinear.convert(l, bound_opts))
                # shape = l.weight.shape[0]
            if isinstance(l, Conv2d):
                layers.append(BoundConv2d.convert(l, bound_opts))
            if isinstance(l, ReLU):
                if 'activation' in bound_opts.keys():
                    if bound_opts['activation'] == 'leaky_relu':
                        # the shape is only valid when the layer before leaky relu layer is a linear layer
                        # layers.append(BoundLeakyReLU(neg_slope=bound_opts['neg_slope'], inplace=l.inplace, bound_opts=bound_opts, shape=shape))
                        layers.append(BoundLeakyReLU(neg_slope=bound_opts['neg_slope'], inplace=l.inplace, bound_opts=bound_opts))
                    elif 'leaky_relu_step' in bound_opts['activation']:
                        if bound_opts['activation'] == 'leaky_relu_step':
                            layers.append(BoundLeakyReLUStep(bound_opts, slope=bound_opts['neg_slope'], right=1))
                        elif bound_opts['activation'] == 'param_leaky_relu_step':
                            layers.append(BoundLeakyReLUStep(bound_opts, slope=bound_opts['neg_slope'], right=1, 
                                            parameterize=True, shape=None))
                        elif bound_opts['activation'] == 'param_slope_leaky_relu_step':
                            layers.append(BoundLeakyReLUStep(bound_opts, slope=bound_opts['neg_slope'], right=1, 
                                            parameterize=True, parameterize_slope=True, shape=None))
                        else:
                            raise Exception('Activation %s is not supported' % bound_opts['activation'])
                    elif bound_opts['activation'] == 'relu':
                        layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
                    else:
                        raise Exception('Activation %s is not supported' % bound_opts['activation'])
                else: # if not specify activation function, we use relu as default
                    layers.append(BoundReLU.convert(l, layers[-1], bound_opts))
            if isinstance(l, Flatten):
                layers.append(BoundFlatten(bound_opts))
        return BoundSequential(*layers)

    def update_parameter(self):
        # update useful values in BoundLeakyReLUStep
        # if their inside parameters have changed (e.g., optimizer.step() is called or load pretrained parameters)
        # this function is only meaningful when you use parameterized BoundLeakyReLUStep
        # also need to call this function when you change device of this model

        if self.contain_parameterized_act:
            for k in self.parameterized_act_keys:
                self._modules[k].update_parameter()
        return 0

    def update_slope(self, slope):
        for m in self._modules.values():
            if isinstance(m, BoundLeakyReLUStep) or isinstance(m, BoundLeakyReLU):
                m.update_slope(slope)


    def use_mean_act_as_param(self):
        if self.contain_parameterized_act:
            for k in self.parameterized_act_keys:
                self._modules[k].use_mean_act_as_param()
        return 0

    ## The __call__ function is overwritten for DataParallel
    def __call__(self, *input, **kwargs):
        
        if "method_opt" in kwargs:
            opt = kwargs["method_opt"]
            kwargs.pop("method_opt")
        else:
            opt = "forward"
            # raise ValueError("Please specify the 'method_opt' as the last argument.")
        if "disable_multi_gpu" in kwargs:
            kwargs.pop("disable_multi_gpu")
        if "need_to_replicate" in kwargs:
            kwargs.pop("need_to_replicate")
        if opt == "full_backward_range":
            # pure crown
            return self.full_backward_range(*input, **kwargs)
        elif opt == "backward_range":
            # this function compute final layer bound given bounds of intermediate layers
            return self.backward_range(*input, **kwargs)
        elif opt == "interval_range": 
            # pure IBP
            return self.interval_range(*input, **kwargs)
        elif opt == "lbp": 
            # Linear bound propogation
            return self.linear_range(*input, **kwargs)
        else:
            return super(BoundSequential, self).__call__(*input, **kwargs)

    ## Full CROWN bounds with all intermediate layer bounds computed by CROWN
    ## This can be slow for training, and it is recommend to use it for verification only
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    def full_backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=True, lower=True):
        h_U = x_U
        h_L = x_L
        modules = list(self._modules.values())
        # IBP through the first weight (it is the same bound as CROWN for 1st layer, and IBP can be faster)
        for i, module in enumerate(modules):
            norm, h_U, h_L, _, _, _, _ = module.interval_propagate(norm, h_U, h_L, eps)
            # skip the first flatten and linear layer, until we reach the first ReLU layer
            if isinstance(module, BoundReLU):
                # now the upper and lower bound of this ReLU layer has been set in interval_propagate()
                last_module = i
                break
        # CROWN propagation for all rest layers
        # outer loop, starting from the 2nd layer until we reach the output layer
        for i in range(last_module + 1, len(modules)):
            # we do not need bounds after ReLU/flatten layers; we only need the bounds
            # before a ReLU layer
            if isinstance(modules[i], BoundReLU):
                # we set C as the weight of previous layer
                if isinstance(modules[i-1], BoundLinear):
                    # add a batch dimension; all images have the same C in this case
                    newC = modules[i-1].weight.unsqueeze(0)
                    # we skip the layer i, and use CROWN to compute pre-activation bounds
                    # starting from layer i-2 (layer i-1 passed as specification)
                    ub, _, lb, _ = self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = newC, upper = True, lower = True, modules = modules[:i-1])
                    # add the missing bias term (we propagate newC which do not have bias)
                    ub += modules[i-1].bias
                    lb += modules[i-1].bias
                elif isinstance(modules[i-1], BoundConv2d):
                    # we need to unroll the convolutional layer here
                    c, h, w = modules[i-1].output_shape
                    newC = torch.eye(c*h*w, device = x_U.device, dtype = x_U.dtype)
                    newC = newC.view(1, c*h*w, c, h, w)
                    # use CROWN to compute pre-actiation bounds starting from layer i-1
                    ub, _, lb, _ = self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = newC, upper = True, lower = True, modules = modules[:i])
                    # reshape to conv output shape; these are pre-activation bounds
                    ub = ub.view(ub.size(0), c, h, w)
                    lb = lb.view(lb.size(0), c, h, w)
                else:
                    raise RuntimeError("Unsupported network structure")
                # set pre-activation bounds for layer i (the ReLU layer)
                modules[i].upper_u = ub
                modules[i].lower_l = lb
                modules[i].update_neuron_status()
        # get the final layer bound with spec C
        return self.backward_range(norm = norm, x_U = x_U, x_L = x_L, eps = eps, C = C, upper = upper, lower = lower)


    ## High level function, will be called outside
    # @param norm perturbation norm (np.inf, 2)
    # @param x_L lower bound of input, shape (batch, *image_shape)
    # @param x_U upper bound of input, shape (batch, *image_shape)
    # @param eps perturbation epsilon (not used for Linf)
    # @param C vector of specification, shape (batch, specification_size, output_size)
    # @param upper compute CROWN upper bound
    # @param lower compute CROWN lower bound
    def backward_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None, upper=False, lower=True, modules=None):
        # start propagation from the last layer
        modules = list(self._modules.values()) if modules is None else modules
        upper_A = C if upper else None
        lower_A = C if lower else None
        upper_sum_b = lower_sum_b = x_U.new([0])
        for i, module in enumerate(reversed(modules)):
            # if isinstance(module, BoundFlatten):
            #     pdb.set_trace()
            upper_A, upper_b, lower_A, lower_b = module.bound_backward(upper_A, lower_A)
            # squeeze is for using broadcasting in the cast that all examples use the same spec
            upper_sum_b = upper_b + upper_sum_b
            lower_sum_b = lower_b + lower_sum_b
        # sign = +1: upper bound, sign = -1: lower bound
        def _get_concrete_bound(A, sum_b, sign = -1):
            if A is None:
                return None
            A = A.view(A.size(0), A.size(1), -1)
            # A has shape (batch, specification_size, flattened_input_size)
            logger.debug('Final A: %s', A.size())
            if norm == np.inf:
                x_ub = x_U.view(x_U.size(0), -1, 1)
                x_lb = x_L.view(x_L.size(0), -1, 1)
                # x_ub and x_lb are of shape (batch, flattened_input_size, 1)
                center = (x_ub + x_lb) / 2.0
                diff = (x_ub - x_lb) / 2.0
                logger.debug('A_0 shape: %s', A.size())
                logger.debug('sum_b shape: %s', sum_b.size())
                # we only need the lower bound
                bound = A.bmm(center) + sign * A.abs().bmm(diff)
                # bound is of shape (batch, specification_size, 1)
                logger.debug('bound shape: %s', bound.size())
            else:
                x = x_U.view(x_U.size(0), -1, 1)
                dual_norm = np.float64(1.0) / (1 - 1.0 / norm)
                deviation = A.norm(dual_norm, -1) * eps
                # this part may have problem, we should figure out
                # whether eps is for the original data or normalized data
                bound = A.bmm(x) + sign * deviation.unsqueeze(-1)
            bound = bound.squeeze(-1) + sum_b
            return bound
        lb = _get_concrete_bound(lower_A, lower_sum_b, sign = -1)
        ub = _get_concrete_bound(upper_A, upper_sum_b, sign = +1)
        if ub is None:
            ub = x_U.new([np.inf])
        if lb is None:
            lb = x_L.new([-np.inf]) 
        return ub, upper_sum_b, lb, lower_sum_b

    def interval_range(self, norm=np.inf, x_U=None, x_L=None, eps=None, C=None):
        losses = 0
        unstable = 0
        dead = 0
        alive = 0
        h_U = x_U
        h_L = x_L
        for i, module in enumerate(list(self._modules.values())[:-1]):
            # all internal layers should have Linf norm, except for the first layer
            norm, h_U, h_L, loss, uns, d, a = module.interval_propagate(norm, h_U, h_L, eps)
            # this is some stability loss used for initial experiments, not used in CROWN-IBP as it is not very effective
            # pdb.set_trace()
            losses += loss
            unstable += uns
            dead += d
            alive += a
        # last layer has C to merge
        norm, h_U, h_L, loss, uns, d, a = list(self._modules.values())[-1].interval_propagate(norm, h_U, h_L, eps, C)
        losses += loss
        unstable += uns
        dead += d
        alive += a
        return h_U, h_L, losses, unstable, dead, alive

    def linear_range(self, x_U=None, x_L=None, x0=None, norm=np.inf, eps=None, C=None):
        module_list = list(self._modules.values())
        # if isinstance(module_list[0], BoundFlatten):
        #     flatten_module = module_list.pop(0)
        #     x0,x_U,x_L = flatten_module.linear_propagate(x0,x_U,x_L)

        for i, module in enumerate(module_list[:-1]):
            # all internal layers should have Linf norm, except for the first layer
            if i == 0:
                if isinstance(module, BoundLinear):
                    # in this case, the input x must be already flattened
                    # for the first layer z1 = weight a0 + bias = = weight x + bias
                    # we already know I x + 0 <= a0=x <= I x + 0
                    # or x_L <= a0=x <= x_U
                    # if x_L = x0-eps, x_U = x0+eps
                    # then these two methods are the same
                    in_features = x0.shape[1]

                    init_A = torch.eye(in_features, device=x0.device).unsqueeze(0)
                    init_b = torch.zeros(x0.shape, device=x0.device)
                    uA, ubias, lA, lbias, h_U, h_L = module.linear_propagate(
                        init_A, init_b, init_A, init_b, x0, norm, eps, 
                        C = None, x_U=x_U, x_L=x_L)
                    
                elif isinstance(module, BoundConv2d):
                    # init_A should be of shape (batch, CHW, C0,H0,W0)
                    # init_b should be of shape (batch, C0, H0, W0) 
                    # x is of shape (batch, C,H,W) 
                    # A_temp = init_A.view(batch, CHW, -1) # shape (batch, CHW, C0 H0 W0)
                    # A_temp = torch.transpose(Au_temp, 1, 2) # shape (batch, C0 H0 W0, CHW)
                    # x0_temp = x0.view(batch, -1).unsqueeze(2) # (batch, CHW, 1)
                    # we need to ensure A_temp x0_temp = x0_temp, of shape (batch, C0 H0 W0, 1)
                    batch = x0.shape[0]
                    Ch = x0.shape[1]
                    H = x0.shape[2]
                    W = x0.shape[3]
                    init_A = torch.eye(Ch*H*W, device=x0.device).unsqueeze(0).expand(batch, -1, -1)
                    init_A = init_A.view(batch, Ch*H*W, Ch,H,W)
                    init_b = torch.zeros(batch, Ch, H, W, device=x0.device)
                    uA, ubias, lA, lbias, h_U, h_L = module.linear_propagate(
                            init_A, init_b, init_A, init_b, x0, norm, eps, 
                            C = None, x_U=x_U, x_L=x_L)
                else:
                    raise Exception('First layer must be linear layer or conv layer')
            else:
                if isinstance(module, BoundLinear):
                    uA, ubias, lA, lbias, h_U, h_L = module.linear_propagate(
                        uA, ubias, lA, lbias, x0, norm, eps, C = None, x_U=x_U, x_L=x_L)
                elif isinstance(module, BoundConv2d):
                    uA, ubias, lA, lbias, h_U, h_L = module.linear_propagate(
                        uA, ubias, lA, lbias, x0, norm, eps, C = None, x_U=x_U, x_L=x_L)
                elif (isinstance(module, BoundReLU) or isinstance(module, BoundLeakyReLU)
                     or isinstance(module, BoundLeakyReLUStep)):
                    uA, ubias, lA, lbias = module.linear_propagate(
                        h_U, h_L, uA, ubias, lA, lbias)
                elif isinstance(module, BoundFlatten):
                    uA, ubias, lA, lbias = module.linear_propagate(uA, ubias, lA, lbias)
                else:
                    print('The module is', module)
                    raise Exception('Only support linear, conv, flatten and activation layer now')
        
        last_module = module_list[-1]
        use_ibp_for_last_layer = False
        # last layer has C to merge
        if isinstance(last_module, BoundLinear):
            if use_ibp_for_last_layer:
                _, h_U, h_L, _, _, _, _ = last_module.interval_propagate(norm, h_U, h_L, eps, C)
            else:
                uA, ubias, lA, lbias, h_U, h_L = last_module.linear_propagate(
                uA, ubias, lA, lbias, x0, norm, eps, C, x_U=x_U, x_L=x_L)

        else:
            raise Exception('Only support linear layer for last layer now')
        
        # now we have h_L <= C z^m <= h_U
        return h_U, h_L


class BoundDataParallel(DataParallel):
    # This is a customized DataParallel class for our project
    def __init__(self, *inputs, **kwargs):
        super(BoundDataParallel, self).__init__(*inputs, **kwargs)
        self._replicas = None
        # self.replicate_time = 0
    # Overide the forward method
    def update_parameter(self):
        # update useful values in BoundLeakyReLUStep
        # if their inside parameters have changed (e.g., optimizer.step() is called or load pretrained parameters)
        # this function is only meaningful when you use parameterized or BoundLeakyReLUStep
        # also need to call this function when you change device of this model

        self.module.update_parameter()
        if not self._replicas is None:
            for rep in self._replicas:
                rep.update_parameter()
        return 0

    def update_slope(self, slope):
        self.module.update_slope(slope)
        if not self._replicas is None:
            for rep in self._replicas:
                rep.update_slope(slope)
        return 0

    def forward(self, *inputs, **kwargs):
        disable_multi_gpu = False
        if "disable_multi_gpu" in kwargs:
            disable_multi_gpu = kwargs["disable_multi_gpu"]
            kwargs.pop("disable_multi_gpu")
        
        need_to_replicate = True
        if 'need_to_replicate' in kwargs:
            need_to_replicate = kwargs['need_to_replicate']
            kwargs.pop('need_to_replicate') # this step is to ensure other parts don't get the unexpected keyward 'need_to_replicate'

        if not self.device_ids or disable_multi_gpu: 
            return self.module(*inputs, **kwargs)
       
        # Only replicate during forwarding propagation. Not during interval bounds
        # and CROWN-IBP bounds, since weights have not been updated. This saves 2/3
        # of communication cost.
        # bound computation will still be performed on multi gpus
        if self._replicas is None or kwargs.get("method_opt", "forward") == "forward":
            if need_to_replicate:
                # self.replicate_time = self.replicate_time + 1
                # start = time.time()
                self._replicas = self.replicate(self.module, self.device_ids)
                
                # this is necessary, because thay don't copy x2, y2 to the corresponding devices automatically 
                # for rep in self._replicas:
                #     rep.update_parameter()
                # replicate_time = time.time()-start
                # print('Have replicated %d times, this time use %f seconds.' % (self.replicate_time, replicate_time))  

        for t in chain(self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError("module must have its parameters and buffers "
                                   "on device {} (device_ids[0]) but found one of "
                                   "them on device: {}".format(self.src_device_obj, t.device))
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids) 
        if len(self.device_ids) == 1:
            return self.module(*inputs[0], **kwargs[0])
        outputs = self.parallel_apply(self._replicas[:len(inputs)], inputs, kwargs)
        return self.gather(outputs, self.output_device)


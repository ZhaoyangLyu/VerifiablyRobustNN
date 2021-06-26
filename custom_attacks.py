import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import numpy as np


def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon,
                  num_steps,
                  random,
                  device,
                  step_size,
                  data_max,
                  data_min,
                  mean,
                  std,
                  custom_clamp=False,
                  **kwargs):
    out = model(X, method_opt="forward")
    err = (out.data.max(1)[1] != y.data).float().sum()
    X_pgd = Variable(X.data, requires_grad=True)
    if random:
        random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
        X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)
    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd, method_opt="forward"), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        if custom_clamp:
            eta = torch.min(torch.max(X_pgd.data - X.data, -epsilon/std), epsilon/std)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(
                torch.max(torch.min(X_pgd, data_max), data_min), requires_grad=True)
        else:
            eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
            X_pgd = Variable(X.data + eta, requires_grad=True)
            X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)

    pgd_out = model(X_pgd, method_opt="forward")
    err_pgd = (pgd_out.data.max(1)[1] != y.data).float().sum()
    
    # error is the number of clean samples that are wrong classified
    # err_pgd is the number of samples that are successfully attacked by pgd
    # pdg_out is the output logits for the found adversarial examples
    return err, err_pgd, pgd_out.detach().cpu().numpy()


def eval_pgd_restart_v2(model, device, test_loader, attack_method,
                         data_max, data_min, mean, std,
                         logger=None,
                         test_epsilon=0.007843, test_num_steps=200,
                         test_step_size=0.003921, num_classes=10,
                         restarts=5, custom_clamp=False
                         ):
    """
    evaluate model by white-box attack
    """
    print("Evaluating {} Attack".format(attack_method))
    model.eval()
    params = dict(epsilon=test_epsilon, num_steps=test_num_steps,
                  step_size=test_step_size, num_classes=num_classes,
                  data_max=data_max, data_min=data_min, mean=mean, std=std,
                  custom_clamp=custom_clamp)
    
    for i_re in range(restarts):
        print('Restart: {}'.format(i_re + 1))
        robust_err_total = 0
        total = 0
        all_labels = []
        all_pgd_outs = []
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            # pgd attack
            X, y = Variable(data, requires_grad=True), Variable(target)
            if attack_method == 'PGD':
                _, err_robust, pgd_out = _pgd_whitebox(model, X, y, random=True, device=device, **params)
            else:
                raise NotImplementedError

            robust_err_total += err_robust
            total += len(data)

            all_pgd_outs.append(pgd_out)
            all_labels.extend(target.cpu().detach().numpy().tolist())

        print('This round PGD Err: %.4f' % (100 * robust_err_total / total).cpu().item())
        all_pgd_outs = np.vstack(all_pgd_outs)
        
        if i_re == 0:
            success_ones = np.argmax(all_pgd_outs,1) != np.asarray(all_labels)
        else:
            this_success_ones = np.argmax(all_pgd_outs,1) != np.asarray(all_labels)
            success_ones = this_success_ones | success_ones

        print('Cummulated PGD Err: {:.4f}'.format(100 * np.asarray(success_ones).sum() / len(all_labels)))
    
    Cummulated_PGD_Err = 100 * np.asarray(success_ones).sum() / len(all_labels)
    return Cummulated_PGD_Err



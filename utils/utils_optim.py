import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors, \
    _take_tensors
import math


## SMD optimizers

class SMD_compress(Optimizer):

    def __init__(self, params, lr=0.01, momentum=0, weight_decay = 0, dampening=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening)
        super(SMD_compress, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(SMD_compress, self).__setstate__(state)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            
#             all_grads = []

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf
    #           (1+eps) norm potential function
                eps = 0.1
                update = (1+eps)* (torch.abs(p.data)**eps) * torch.sign(p.data) - group['lr'] * d_p
                p.data = (torch.abs(update/(1+eps))**(1/eps)) * torch.sign(update)

        return loss 
    

import torch
from torch.optim.optimizer import Optimizer

class SMD_qnorm(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0, dampening=0, q=3, nesterov=False):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 1.01 <= q:
            raise ValueError(f"Invalid q_norm value: {q}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.q = q
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, dampening=dampening, q=q, nesterov=nesterov)
        super(SMD_qnorm, self).__init__(params, defaults)

    def step(self, closure=None):
        '''Performs a single optimization step.'''
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            lr = group['lr']
            q = group['q']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(p.data, alpha=weight_decay)
                
                param_state = self.state[p]
                if 'velocity' not in param_state:
                    param_state['velocity'] = torch.zeros_like(p.data)
                
                v = param_state['velocity']
                v.mul_(momentum).add_(d_p, alpha=1 - dampening)

                if nesterov:
                    d_p = d_p.add(v, alpha=momentum)
                else:
                    d_p = v

                # Compute q-norm update
                update = torch.abs(p.data)** (self.q - 1) * torch.sign(p.data) - lr * d_p
                p.data = torch.abs(update)**(1 / (self.q - 1)) * torch.sign(update)
                
                

        return loss
    
    
# RMD Loss

class RMD_Loss(nn.Module):
    def __init__(self, sample_loss_func, num_datapoints, device, reduction='mean'):
        super(RMD_Loss, self).__init__()
        self.sample_loss_func = sample_loss_func
        self.z = torch.zeros(num_datapoints,device=device)
        self.idx = 0
        self.reduction = reduction
    
    def set_z_values(self, z, idx):
        self.z = z
        self.idx = idx

    def forward(self, predictions, target):
        # sample_loss = self.sample_loss_func(predictions, target)

        # adjusted_sample_loss = torch.sqrt(2 * self.sample_loss_func(predictions, target))

        # squared_loss = 0.5 * torch.square(self.z[self.idx] - adjusted_sample_loss)
        squared_loss = 0.5 * torch.square(self.z[self.idx] - torch.sqrt(2 * self.sample_loss_func(predictions, target)))
        if self.reduction == 'mean':
            return torch.mean(squared_loss)
        elif self.reduction == 'sum':
            return torch.sum(squared_loss)
        else:
            return squared_loss

class RMD_Loss_Cores(nn.Module):
    def __init__(self, sample_loss_func, num_datapoints, device, reduction='mean'):
        super(RMD_Loss_Cores, self).__init__()
        self.sample_loss_func = sample_loss_func
        self.z = torch.zeros(num_datapoints,device=device)
        self.idx = 0
        self.reduction = reduction
    
    def set_z_values(self, z, idx):
        self.z = z
        self.idx = idx

    def forward(self, predictions, target,epoch,prior):
        # sample_loss = self.sample_loss_func(predictions, target)

        # adjusted_sample_loss = torch.sqrt(2 * self.sample_loss_func(predictions, target))

        # squared_loss = 0.5 * torch.square(self.z[self.idx] - adjusted_sample_loss)
        loss,_=self.sample_loss_func(epoch,predictions, target,prior,True)
        squared_loss = 0.5 * torch.square(self.z[self.idx] -
                                           torch.sqrt(2 * loss ))
        if self.reduction == 'mean':
            return torch.mean(squared_loss)
        elif self.reduction == 'sum':
            return torch.sum(squared_loss)
        else:
            return squared_loss    
        
        
## Custom CE Loss 


class customCrossEntropy(nn.Module):
    def __init__(self, device, reduction='none'):
        super(customCrossEntropy, self).__init__()
        self.vanillaCrossEntropy = nn.CrossEntropyLoss(reduction='none').to(device)
        self.reduction = reduction
        self.device = device

    def forward(self, predictions, target):
        # Compute the logits relative to the correct class to stabilize the computation.
        reduced_exponents = predictions - predictions.gather(1, target.unsqueeze(1)).squeeze()[:, None]
        
        # Use torch.logsumexp for numerical stability.
        # We keep dimensions consistent with the input for compatibility with further operations.
        loss_elements = torch.logsumexp(reduced_exponents, dim=1)
        
        # # Adjust for extremely small values to prevent underflow.
        # loss_elements[loss_elements < 1e-38] = 1e-38

        # Calculate the conditional loss: custom loss for correct classifications,
        # and vanilla cross-entropy for incorrect ones.
        loss_return = torch.where(torch.argmax(predictions, dim=1) == target, 
                                  loss_elements,
                                  self.vanillaCrossEntropy(predictions, target))

        # Apply the specified reduction to the loss.
        if self.reduction == 'mean':
            return loss_return.mean()
        elif self.reduction == 'sum':
            return loss_return.sum()
        else:
            return loss_return


class customCrossEntropy(nn.Module):
  def __init__(self, device, reduction = 'none'):
    super(customCrossEntropy, self).__init__()
    self.vanillaCrossEntropy = nn.CrossEntropyLoss(reduction='none').to(device)
    self.reduction = reduction
    self.device = device

  def forward(self, predictions, target):

    reduced_exponents = predictions - predictions.gather(1, target.unsqueeze(1)).squeeze()[:, None]
    loss_elements = torch.logsumexp(reduced_exponents.type(torch.float64), dim=1) #+ 1e-38
    # loss_elements = torch.logsumexp(reduced_exponents, dim=1) #+ 1e-38
    # exp = torch.expm1(reduced_exponents.type(torch.float64))+ 1
    # loss_elements = torch.log1p(torch.sum(exp, dim = 1) - 1).type(torch.float64)

    # At a certain point PyTorch division will recognize a value as zero even if it is not
    # this helps set extremely small values above that treshhold (happens rarely)
    # loss_elements[loss_elements < 1e-38] = 1e-38

    loss_return = torch.where(torch.argmax(predictions, dim=1) == target, 
                   loss_elements,
                   self.vanillaCrossEntropy(predictions, target))

    if self.reduction == 'mean':
        return loss_return.mean()
    elif self.reduction == 'sum':
        return loss_return.sum()
    else:
        return loss_return        
    
    
import torch
from torch import Tensor
from torch.optim.optimizer import (Optimizer, _use_grad_for_differentiable, _get_value, _dispatch_sqrt,
                        _stack_if_compiling, _capturable_doc, _differentiable_doc, _foreach_doc,
                        _fused_doc, _maximize_doc, _default_to_fused_or_foreach)
from typing import List, Optional
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

# __all__ = ["AdamW", "adamw"]


class AdamWq(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        q=2,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 1.01 <= q:
            raise ValueError(f"Invalid q_norm value: {q}")        
        self.q = q        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            foreach=foreach,
            maximize=maximize,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        super().__init__(params, defaults)

        if fused:
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # Suppor AMP with FP16/BF16 model params which would need
            # higher prec copy of params to do update math in higher prec to
            # alleviate the loss of information.
            if not all(
                p.is_cuda and torch.is_floating_point(p)
                for pg in self.param_groups for p in pg['params']
            ):
                raise RuntimeError("`fused=True` requires all the params to be CUDA, floating point Tensor")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("fused", None)
        state_values = list(self.state.values())
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(float(s["step"]))

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["step"] = (
                    torch.zeros((1,), dtype=torch.float, device=p.device)
                    if group["capturable"] or group["fused"]
                    else torch.tensor(0.0)
                )
                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

            exp_avgs.append(state["exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])

            if amsgrad:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            state_steps.append(state["step"])

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                q=self.q,
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


AdamWq.__doc__ = r"""Implements AdamW algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{(lr)}, \: \beta_1, \beta_2
                \text{(betas)}, \: \theta_0 \text{(params)}, \: f(\theta) \text{(objective)},
                \: \epsilon \text{ (epsilon)}                                                    \\
            &\hspace{13mm}      \lambda \text{(weight decay)},  \: \textit{amsgrad},
                \: \textit{maximize}                                                             \\
            &\textbf{initialize} : m_0 \leftarrow 0 \text{ (first moment)}, v_0 \leftarrow 0
                \text{ ( second moment)}, \: \widehat{v_0}^{max}\leftarrow 0              \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \theta_t \leftarrow \theta_{t-1} - \gamma \lambda \theta_{t-1}         \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm}\widehat{v_t}^{max} \leftarrow \mathrm{max}(\widehat{v_t}^{max},
                \widehat{v_t})                                                                   \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}^{max}} + \epsilon \big)                                 \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\theta_t \leftarrow \theta_t - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Decoupled Weight Decay Regularization`_.
    """ + r"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {maximize}
        {foreach}
        {capturable}
        {differentiable}
        {fused}
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ

    """.format(maximize=_maximize_doc,
               foreach=_foreach_doc,
               fused=_fused_doc,
               capturable=_capturable_doc,
               differentiable=_differentiable_doc)


def adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    q:float,
    weight_decay: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(params, differentiable, use_fused=False)
    if fused is None:
        fused = False
    if foreach is None:
        foreach = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")


    
    func = _single_tensor_adamw

    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        q=q,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_adamw(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    lr: float,
    q:float,
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
):

    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        if capturable:
            assert (
                param.is_cuda and step_t.is_cuda
            ), "If capturable=True, params and state_steps must be CUDA tensors."

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            param = torch.view_as_real(param)

        # update step
        step_t += 1

        # Perform stepweight decay
        param.mul_(1 - lr * weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            # 1 - beta1 ** step can't be captured in a CUDA graph, even if step is a CUDA tensor
            # (incurs "RuntimeError: CUDA error: operation not permitted when stream is capturing")
            bias_correction1 = 1 - torch.pow(beta1, step)
            bias_correction2 = 1 - torch.pow(beta2, step)

            step_size = lr / bias_correction1
            step_size_neg = step_size.neg()

            bias_correction2_sqrt = bias_correction2.sqrt()

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sqs_i = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sqs_i = max_exp_avg_sqs[i]
                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sqs_i, exp_avg_sq))
                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)
            else:
                denom = (
                    exp_avg_sq.sqrt() / (bias_correction2_sqrt * step_size_neg)
                ).add_(eps / step_size_neg)

            param.addcdiv_(exp_avg, denom)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            step_size = lr / bias_correction1

            bias_correction2_sqrt = _dispatch_sqrt(bias_correction2)

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)

            # param.addcdiv_(exp_avg, denom, value=-step_size)
            # update = q*torch.abs(param.data)** (q - 1) * torch.sign(param.data) + exp_avg / denom * -step_size
            # param.data = torch.abs(update/q)**(1 / (q - 1)) * torch.sign(update)
            update = torch.abs(param.data)** (q - 1) * torch.sign(param.data) + exp_avg / denom * -step_size
            param.data = torch.abs(update)**(1 / (q - 1)) * torch.sign(update)    
    
## get extras required for RMD
def get_RMD_losses(device,loader):
    N = len(loader)
    z =  torch.normal(0.0, 0.0000005, size=(N,), device=device)#.type(torch.float64) #small z
    per_sample_loss = torch.zeros(N, device=device).type(torch.float64)
    per_sample_criterion = customCrossEntropy(device, reduction = 'none').to(device)#.type(torch.float64)
    total_loss = torch.zeros(200, device=device)
    lmbda = 0.1
    rmd_criterion = RMD_Loss(per_sample_criterion,device=device, num_datapoints=N) #losses are averaged in minibatch
    return z, per_sample_loss, per_sample_criterion, total_loss, lmbda, rmd_criterion
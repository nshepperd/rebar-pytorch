import numpy as np
import torch
import torch.nn.functional as F
import functools

def conditional_gumbel(logits, D, k=1):
    """Outputs k samples of Y = logits + StandardGumbel(), such that the
    argmax is given by D (one hot vector).

    """
    # iid. exponential
    E = torch.distributions.exponential.Exponential(rate=torch.ones_like(logits)).sample()
    # E of the chosen class
    Ei = (D * E).sum(dim=-1, keepdim=True)
    # partition function (normalization constant)
    Z = logits.exp().sum(dim=-1, keepdim=True)
    # Sampled gumbel-adjusted logits
    adjusted = (D * (-torch.log(E) + torch.log(Z)) +
                (1 - D) * -torch.log(E/torch.exp(logits) + Ei / Z))
    return adjusted

def unconditional_gumbel(logits):
    """Outputs samples of Y = logits + StandardGumbel()."""
    return logits + torch.rand_like(logits).log().neg().log().neg()

def rebar(f, logits, lm, I):
        num_vocab = logits.shape[-1]
        assert I.shape == logits.shape[:-1]

        one_hot = F.one_hot(I, num_vocab).float()
        # one_hot.requires_grad_()


        z = unconditional_gumbel(logits)
        zbar = conditional_gumbel(logits, one_hot)
        fb = f(one_hot)
        fz = f(torch.sigmoid(z/lm))
        fzbar = f(torch.sigmoid(zbar/lm))

        batch_size = np.product(fb.shape)

        log_probs = F.log_softmax(logits, dim=-1)
        chosen_log_probs = torch.gather(log_probs, dim=-1, index=I.unsqueeze(-1))
        error = (fb - fzbar)
        error = error.reshape(list(error.shape) + [1] * (len(logits.shape) - len(error.shape)))
        grad = (error * torch.autograd.grad(chosen_log_probs.sum(), logits, create_graph=True)[0]
                + torch.autograd.grad(fz.sum(), logits, create_graph=True)[0]
                - torch.autograd.grad(fzbar.sum(), logits, create_graph=True)[0])

        # Temperature optimization: tries to minimize the variance of the gradient estimator.
        lm_grad = torch.autograd.grad(grad.square().sum(), lm)[0]
        return fb.mean(), grad/batch_size, lm_grad/batch_size

class REBARFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, logits, lm, I):
        num_vocab = logits.shape[-1]
        one_hot = F.one_hot(I, num_vocab).float()
        fb = f(one_hot).mean()
        ctx.save_for_backward(logits, lm, I)
        ctx.f = f
        # with torch.enable_grad():
        #     (fb, grad, lm_grad) = rebar(f, logits, lm, I)
        # ctx.save_for_backward(grad.detach(), lm_grad.detach())
        return fb

    @staticmethod
    def backward(ctx, grad_output):
        logits, lm, I = ctx.saved_tensors
        logits.requires_grad_()
        with torch.enable_grad():
            (_, grad, lm_grad) = rebar(ctx.f, logits, lm, I)
        # grad, lm_grad = ctx.saved_tensors
        return (None, grad_output * grad, grad_output * lm_grad, None)

class REBAR(torch.nn.Module):
    """A bit janky. Use it like this:

    rebar_f = REBAR(f)
    opt = torch.optim.Adam(list(model.parameters()) + list(rebar_f.parameters))
    ...
    opt.zero_grad()
    logits = model(data) # [batch_size, ..., num_vocab]
    loss = rebar_f(logits, I=your_sampled_tokens_or_None, any_other_arguments_for_f)
    loss.backward()
    opt.step()

    Returns the f(one_hot).mean(). Calling .backward() propagates
    gradients both to `logits` (to minimize loss) and to rebar_f (to
    minimize variance).

    """
    def __init__(self, f):
        torch.nn.Module.__init__(self)
        self.f = f
        self.llm = torch.nn.Parameter(torch.zeros([], dtype=torch.float32)) # log-temperature

    def forward(self, logits, I=None, *args, **kwargs):
        if I is None:
            I = torch.distributions.categorical.Categorical(logits=logits).sample()
        f = functools.partial(self.f, *args, **kwargs)
        return REBARFunction.apply(f, logits, self.llm.exp(), I)

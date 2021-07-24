import numpy as np
import torch
import torch.nn.functional as F
import functools

from rebar import REBAR

if __name__ == '__main__':
    # Use the test function from the RELAX paper:
    # y = (x - 0.499)^2
    # where x in [0,1]

    z01 = torch.tensor([0.0, 1.0])
    outcomes = torch.tensor([0.499**2, (1-0.499)**2])
    impl = 'linear'
    def f(one_hot):
        if impl == 'normal':
            # Normal implementation yields a quadratic function, REBAR
            # provides slight variance reduction (about 10%).
            val = one_hot @ z01
            return (val - 0.499).square()
        elif impl == 'linear':
            # Direct interpolation between 0,1 yields a linear
            # function, for which REBAR can take the variance
            # arbitrarily low, approaching the exact gradient which is
            # just d/dlogits (softmax(logits) @ outcomes).  The
            # variance reduction is about 3x at temperature=1.
            return one_hot @ outcomes
    logits = torch.nn.Parameter(torch.zeros(2))
    rebar_f = REBAR(f)

    opt = torch.optim.Adam(list(rebar_f.parameters()), lr=0.001)

    BATCH_SIZE=32

    total = torch.zeros(2)
    r_total = torch.zeros(2)
    counter = 1
    while True:
        # Gradient estimation with REBAR.
        ls = logits.reshape(1,2).expand(BATCH_SIZE,2)
        loss = rebar_f(ls)
        grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        grad_ = torch.autograd.grad(loss, ls)[0]
        total.add_(grad)

        # Temperature optimization to minimize variance.
        ls = logits.reshape(1,2).expand(BATCH_SIZE,2)
        loss = rebar_f(ls).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Gradient estimation with REINFORCE.
        ls = logits.reshape(1,2).expand(BATCH_SIZE,2)
        I = torch.distributions.categorical.Categorical(logits=ls).sample()
        one_hot = F.one_hot(I, 2).float()
        lp_chosen = torch.gather(F.log_softmax(ls, dim=-1), dim=-1, index=I.unsqueeze(-1))
        val = f(one_hot).unsqueeze(-1) # [b, 1]
        gp_chosen = torch.autograd.grad(lp_chosen.sum(), ls)[0] # [b, v]
        r_grad_ = (val * gp_chosen)
        r_grad = r_grad_.mean(dim=0)
        r_total.add_(r_grad)

        # Exact gradient.
        e = F.softmax(logits, dim=-1) @ torch.tensor([0.499**2, (1-0.499)**2])
        exact = torch.autograd.grad(e, logits)[0]

        print(counter, 1000*total/counter, 1000*r_total/counter, 1000*exact,
              (BATCH_SIZE*grad_ - exact).square().mean().sqrt(),
              (r_grad_ - exact).square().mean().sqrt(), rebar_f.llm.exp().item())
        counter += 1

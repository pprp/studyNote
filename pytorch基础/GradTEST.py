import torch

x = torch.randn(1, 2)
y = torch.randn(1, 1)

w1 = torch.randn(2, 4)
w2 = torch.randn(4, 6)

epoch_n = 20
learning_rate = 1e-6

for epoch in range(epoch_n):
    h1 = x.mm(w1)
    h1 = h1.clamp(min=0)
    y_pred = h1.mm(w2)

    loss=(y_pred-y).pow(2).sum()
    print("epoch:%d, loss:%.3f" % (epoch, loss))

    grad_y_pred = 2*(y_pred-y)
    grad_w2 = h1.t().mm(grad_y_pred)

    grad_h = grad_y_pred.clone()
    grad_h = grad_h.mm(w2.t())
    grad_h.clamp_(min=0)
    grad_w1 = x.t().mm(grad_h)

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
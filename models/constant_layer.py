from torch.autograd.function import InplaceFunction


class ConstantLayer(InplaceFunction):

    @staticmethod
    def forward(ctx, input1, forward_c, backward_c):
        ctx.backward_c = backward_c
        return input1 * forward_c

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.backward_c, None, None


def constant_layer(input1, forward_c, backward_c):
    return ConstantLayer.apply(input1, forward_c, backward_c)

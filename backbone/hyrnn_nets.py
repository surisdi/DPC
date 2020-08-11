"""
Network definitions from https://github.com/ferrine/hyrnn
"""

import itertools
import torch.nn
import torch.nn.functional
import math
import geoopt.manifolds.stereographic.math as gmath
import geoopt


def mobius_linear(
    input,
    weight,
    bias=None,
    hyperbolic_input=True,
    hyperbolic_bias=True,
    nonlin=None,
    k=-1.0,
):
    k = torch.tensor(k)
    if hyperbolic_input:
        output = mobius_matvec(weight, input, k=k)
    else:
        output = torch.nn.functional.linear(input, weight)
        output = gmath.expmap0(output, k=k)
    if bias is not None:
        if not hyperbolic_bias:
            bias = gmath.expmap0(bias, k=k)
        output = gmath.mobius_add(output, bias.unsqueeze(0).expand_as(output), k=k)
    if nonlin is not None:
        output = gmath.mobius_fn_apply(nonlin, output, k=k)
    output = gmath.project(output, k=k)
    return output


def mobius_matvec(m: torch.Tensor, x: torch.Tensor, *, k: torch.Tensor, dim=-1):
    return _mobius_matvec(m, x, k, dim=dim)


def _mobius_matvec(m: torch.Tensor, x: torch.Tensor, k: torch.Tensor, dim: int = -1):
    if m.dim() > 2 and dim != -1:
        raise RuntimeError(
            "broadcasted Möbius matvec is supported for the last dim only"
        )
    x_norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    if dim != -1 or m.dim() == 2:
        # mx = torch.tensordot(x, m, [dim], [1])
        mx = torch.matmul(m, x.transpose(1, 0)).transpose(1, 0)  # TODO I modified this. Probably not the best thing to do
    else:
        mx = torch.matmul(m, x.unsqueeze(-1)).squeeze(-1)
    mx_norm = mx.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    res_c = gmath.tan_k(mx_norm / x_norm * gmath.artan_k(x_norm, k), k) * (mx / mx_norm)
    cond = (mx == 0).prod(dim=dim, keepdim=True, dtype=torch.uint8)
    res_0 = torch.zeros(1, dtype=res_c.dtype, device=res_c.device)
    res = torch.where(cond, res_0, res_c)
    return res


def one_rnn_transform(W, h, U, x, b, k):
    W_otimes_h = gmath.mobius_matvec(W, h, k=k)
    U_otimes_x = gmath.mobius_matvec(U, x, k=k)
    Wh_plus_Ux = gmath.mobius_add(W_otimes_h, U_otimes_x, k=k)
    return gmath.mobius_add(Wh_plus_Ux, b, k=k)


def mobius_gru_cell(
    input: torch.Tensor,
    hx: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    k: torch.Tensor,
    nonlin=None,
):
    W_ir, W_ih, W_iz = weight_ih.chunk(3)
    b_r, b_h, b_z = bias
    W_hr, W_hh, W_hz = weight_hh.chunk(3)

    z_t = gmath.logmap0(one_rnn_transform(W_hz, hx, W_iz, input, b_z, k), k=k).sigmoid()
    r_t = gmath.logmap0(one_rnn_transform(W_hr, hx, W_ir, input, b_r, k), k=k).sigmoid()

    rh_t = gmath.mobius_pointwise_mul(r_t, hx, k=k)
    h_tilde = one_rnn_transform(W_hh, rh_t, W_ih, input, b_h, k)

    if nonlin is not None:
        h_tilde = gmath.mobius_fn_apply(nonlin, h_tilde, k=k)
    delta_h = gmath.mobius_add(-hx, h_tilde, k=k)
    h_out = gmath.mobius_add(hx, gmath.mobius_pointwise_mul(z_t, delta_h, k=k), k=k)
    return h_out


def mobius_gru_loop(
    input: torch.Tensor,
    h0: torch.Tensor,
    weight_ih: torch.Tensor,
    weight_hh: torch.Tensor,
    bias: torch.Tensor,
    k: torch.Tensor,
    batch_sizes=None,
    hyperbolic_input: bool = False,
    hyperbolic_hidden_state0: bool = False,
    nonlin=None,
):
    if not hyperbolic_hidden_state0:
        hx = gmath.expmap0(h0, k=k)
    else:
        hx = h0
    if not hyperbolic_input:
        input = gmath.expmap0(input, k=k)
    outs = []
    if batch_sizes is None:
        input_unbinded = input.unbind(0)
        for t in range(input.size(0)):
            hx = mobius_gru_cell(
                input=input_unbinded[t],
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=k,
            )
            outs.append(hx)
        outs = torch.stack(outs)
        h_last = hx
    else:
        h_last = []
        T = len(batch_sizes) - 1
        for i, t in enumerate(range(batch_sizes.size(0))):
            ix, input = input[: batch_sizes[t]], input[batch_sizes[t] :]
            hx = mobius_gru_cell(
                input=ix,
                hx=hx,
                weight_ih=weight_ih,
                weight_hh=weight_hh,
                bias=bias,
                nonlin=nonlin,
                k=k,
            )
            outs.append(hx)
            if t < T:
                hx, ht = hx[: batch_sizes[t+1]], hx[batch_sizes[t+1]:]
                h_last.append(ht)
            else:
                h_last.append(hx)
        h_last.reverse()
        h_last = torch.cat(h_last)
        outs = torch.cat(outs)
    return outs, h_last


class MobiusLinear(torch.nn.Linear):
    def __init__(
        self,
        *args,
        hyperbolic_input=True,
        hyperbolic_bias=True,
        nonlin=None,
        k=-1.0,
        **kwargs
    ):
        k = torch.tensor(k)
        super().__init__(*args, **kwargs)
        if self.bias is not None:
            if hyperbolic_bias:
                self.ball = manifold = geoopt.PoincareBall(c=k.abs())
                self.bias = geoopt.ManifoldParameter(self.bias, manifold=manifold)
                with torch.no_grad():
                    self.bias.set_(gmath.expmap0(self.bias.normal_() / 4, k=k))
        with torch.no_grad():
            self.weight.normal_(std=1e-2)
        self.hyperbolic_bias = hyperbolic_bias
        self.hyperbolic_input = hyperbolic_input
        self.nonlin = nonlin
        self.k = k

    def forward(self, input):
        return mobius_linear(
            input,
            weight=self.weight,
            bias=self.bias,
            hyperbolic_input=self.hyperbolic_input,
            nonlin=self.nonlin,
            hyperbolic_bias=self.hyperbolic_bias,
            k=self.k,
        )

    def extra_repr(self):
        info = super().extra_repr()
        info += "c={}, hyperbolic_input={}".format(self.ball.c, self.hyperbolic_input)
        if self.bias is not None:
            info = ", hyperbolic_bias={}".format(self.hyperbolic_bias)
        return info


class MobiusDist2Hyperplane(torch.nn.Module):
    def __init__(self, in_features, out_features, k=-1.0):
        k = torch.tensor(k)
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ball = ball = geoopt.PoincareBall(c=k.abs())
        self.sphere = sphere = geoopt.manifolds.Sphere()
        self.scale = torch.nn.Parameter(torch.zeros(out_features))
        point = torch.randn(out_features, in_features) / 4
        point = gmath.expmap0(point, k=k)
        tangent = torch.randn(out_features, in_features)
        self.point = geoopt.ManifoldParameter(point, manifold=ball)
        with torch.no_grad():
            self.tangent = geoopt.ManifoldParameter(tangent, manifold=sphere).proj_()

    def forward(self, input):
        input = input.unsqueeze(-2)
        distance = gmath.dist2plane(
            x=input, p=self.point, a=self.tangent, k=self.ball.c, signed=True
        )
        return distance * self.scale.exp()

    def extra_repr(self):
        return (
            "in_features={in_features}, out_features={out_features}, "
            "c={ball.c}".format(
                **self.__dict__
            )
        )


class MobiusGRU(torch.nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=2,
        bias=True,
        nonlin=None,
        hyperbolic_input=True,
        hyperbolic_hidden_state0=True,
        k=-1.0,
    ):
        super().__init__()
        '''
        TODO: generalize to any number of layers
        current problem: ParameterList doesn't get copied to
        multiple GPUs when model is wrapped in DataParallel
        
        bug source: https://github.com/pytorch/pytorch/issues/36035
        '''
        assert num_layers == 2, '====[hyrnn_nets.py] current version only support 2-layer GRU===='
        
        self.ball = geoopt.PoincareBall(c=k.abs())
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        weight_ih = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.Tensor(3 * hidden_size, input_size if i == 0 else hidden_size)
                )
                for i in range(num_layers)
            ]
        )
        weight_hh = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
                for _ in range(num_layers)
            ]
        )
        if bias:
            biases = []
            for i in range(num_layers):
                bias = torch.randn(3, hidden_size) * 1e-5
                bias = geoopt.ManifoldParameter(
                    gmath.expmap0(bias, k=self.ball.c), manifold=self.ball
                )
                biases.append(bias)
            self.bias = torch.nn.ParameterList(biases)
        else:
            self.register_buffer("bias", None)
            
        #====ONLY SUPPORT 2 LAYERS====#
        self.weight_ih_1 = weight_ih[0]
        self.weight_ih_2 = weight_ih[1]
        self.weight_hh_1 = weight_hh[0]
        self.weight_hh_2 = weight_hh[1]
        self.bias_1 = self.bias[0]
        self.bias_2 = self.bias[1]
        
        self.nonlin = nonlin
        self.hyperbolic_input = hyperbolic_input
        self.hyperbolic_hidden_state0 = hyperbolic_hidden_state0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        #====ONLY SUPPORT 2 LAYERS====#
        for weight in itertools.chain.from_iterable([self.weight_ih_1, self.weight_hh_1, self.weight_ih_2, self.weight_hh_2]):
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, input: torch.Tensor, h0=None):
        # input shape: seq_len, batch, input_size
        # hx shape: batch, hidden_size
        is_packed = isinstance(input, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            input, batch_sizes = input[:2]
            max_batch_size = int(batch_sizes[0])
        else:
            batch_sizes = None
            max_batch_size = input.size(1)
        if h0 is None:
            h0 = input.new_zeros(
                self.num_layers, max_batch_size, self.hidden_size, requires_grad=False
            )
        h0 = h0.unbind(0)


        #====ONLY SUPPORT 2 LAYERS====#
        weight_ih = [self.weight_ih_1, self.weight_ih_2]
        weight_hh = [self.weight_hh_1, self.weight_hh_2]
        if self.bias is not None:
            biases = [self.bias_1, self.bias_2]
        else:
            biases = (None,) * self.num_layers
        outputs = []
        last_states = []
        out = input
        for i in range(self.num_layers):
            out, h_last = mobius_gru_loop(
                input=out,
                h0=h0[i],
                weight_ih=weight_ih[i],
                weight_hh=weight_hh[i],
                bias=biases[i],
                k=self.ball.c,
                hyperbolic_hidden_state0=self.hyperbolic_hidden_state0 or i > 0,
                hyperbolic_input=self.hyperbolic_input or i > 0,
                nonlin=self.nonlin,
                batch_sizes=batch_sizes,
            )
            outputs.append(out)
            last_states.append(h_last)
        if is_packed:
            out = torch.nn.utils.rnn.PackedSequence(out, batch_sizes)
        ht = torch.stack(last_states)
        # default api assumes
        # out: (seq_len, batch, num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        # if packed:
        # out: (sum(seq_len), num_directions * hidden_size)
        # ht: (num_layers * num_directions, batch, hidden_size)
        return out, ht

    def extra_repr(self):
        return (
            "{input_size}, {hidden_size}, {num_layers}, "
            "weight_ih_1={weight_ih_1}, weight_ih_2={weight_ih_2}, "
            "weight_hh_1={weight_hh_1}, weight_hh_2={weight_hh_2}, "
            "bias_1={bias_1}, bias_2={bias_2}, "
            "hyperbolic_input={hyperbolic_input}, "
            "hyperbolic_hidden_state0={hyperbolic_hidden_state0}, "
            "c={self.ball.c}"
        ).format(**self.__dict__, self=self,
                 weight_ih_1=self.weight_ih_1 is not None, weight_ih_2=self.weight_ih_2 is not None,
                 weight_hh_1=self.weight_hh_1 is not None, weight_hh_2=self.weight_hh_2 is not None,
                 bias_1=self.bias_1 is not None, bias_2=self.bias_2 is not None,
                )

    
    
    
    
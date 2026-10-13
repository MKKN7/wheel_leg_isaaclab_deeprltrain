import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class Identity(torch.nn.Identity):
    pass


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int = None,
            hidden_dims: tuple = (256, 128),
            activation: str = "elu",
            output_activation: str = "identity",
    ):
        super().__init__()
        activation = resolve_nn_activation(activation)
        output_activation = resolve_nn_activation(output_activation)

        # model
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(activation)
        for layer_index in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[layer_index], hidden_dims[layer_index + 1]))
            layers.append(activation)
        if output_dim is not None:
            layers.append(nn.Linear(hidden_dims[-1], output_dim))
            layers.append(output_activation)
        self.model = nn.Sequential(*layers)

    @staticmethod
    def init_weights(sequential, scales):
        [
            torch.nn.init.orthogonal_(module.weight, gain=scales[idx])
            for idx, module in enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))
        ]

    def reset(self, dones=None):
        pass

    def forward(self, x):
        return self.model(x)

    def __getitem__(self, idx):
        return self.model[idx]


class MLPBatchNorm(MLP):
    """每层隐藏 Linear 后接 BatchNorm1d，再接激活；可选输出 Linear 与末尾激活（与原先工厂逻辑一致）。

    不调用 :meth:`MLP.__init__`，避免先建无 BN 的 ``self.model`` 再覆盖；仅继承 ``forward`` / ``reset`` / ``getitem`` / ``init_weights`` 等行为。
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        hidden_dims: tuple | list = (256, 128),
        activation: str = "elu",
        last_act: bool = False,
        bias: bool = True,
    ):
        nn.Module.__init__(self)
        act = resolve_nn_activation(activation)
        hidden_dims = tuple(hidden_dims)
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dims[0], bias=bias),
            nn.BatchNorm1d(hidden_dims[0]),
            act,
        ]
        for l in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1], bias=bias))
            layers.append(nn.BatchNorm1d(hidden_dims[l + 1]))
            layers.append(act)
        if output_dim:
            layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=bias))
        if last_act:
            layers.append(act)
        self.model = nn.Sequential(*layers)

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class OursInjectedLinear(nn.Module):
    """A fused-``qkv`` linear layer with Ours low-rank branches.

    Each continual adaptation stage uses two low-rank branches attached to the frozen
    ``qkv`` weight. Each branch is a ``B @ A`` with a FROZEN down-projection ``A`` and a
    TRAINABLE up-projection ``B``. In this implementation the live branches touch only
    the q and v slices of the fused [query | key | value] projection; k remains base-only:

      - the SHARED / general subspace ``S`` (``s_down`` = A_s, ``ours_s_up`` = B_s), and
      - the truly task-specific subspace ``P`` (``p_down`` = A_p, ``ours_p_up`` = B_p).

    During CTTA, each corruption segment is treated as one adaptation domain. At every
    domain boundary the method designs the two frozen downs from the layer's input
    covariance (S from SVD of ``cur+prev``; P from a Cholesky-whitened generalised-eigen of
    ``cur`` w.r.t. ``prev``), trains the ups online by entropy minimisation, then closed-form
    energy-merges the gated update into ``linear_ours.weight`` and zeros the ups -- so old
    domains' knowledge is baked into the backbone, not kept as live adapters.

    Naming convention (load-bearing):
      - ``ours_s_up`` / ``ours_p_up`` carry the ``"ours_"`` marker -> the ONLY params in the
        adapter optimizer group; zero-init; trainable only for the current domain. They
        output [query | value] deltas, not key deltas.
      - ``linear_ours`` (the base; name has ``"ours"`` but NOT ``"ours_"``) is excluded from
        that group; backbone is frozen here (base LR 0), it only changes via the merge.
      - ``s_down`` / ``p_down`` (designed downs) and ``cur_matrix`` / ``prev_matrix`` (input
        covariances) are BUFFERS: no ``"ours_"`` marker so they are excluded from the
        optimizer, yet carried in ``state_dict`` for reset/deepcopy.

    forward adds the live branches to q and v of the base output (ups are 0 right after a
    domain boundary, so the layer == its merged base -- no output jump), and, when
    ``collecting``, accumulates the raw input gram ``X^T X`` into ``cur_matrix`` for the
    next design step.
    """

    def __init__(self, in_features, out_features, bias=False, r=32, n_tasks=15):
        super().__init__()
        assert out_features % 3 == 0, "expects a fused qkv Linear (out = 3*dim)"
        self.in_features = in_features
        self.out_features = out_features
        self.qkv_dim = out_features // 3
        self.r = r
        self.n_tasks = n_tasks

        self.linear_ours = nn.Linear(in_features, out_features, bias)  # base (fused qkv)

        # trainable up branches (B): zero-init, frozen until their domain becomes current
        query_value_out = 2 * self.qkv_dim
        self.ours_s_up = nn.Linear(r, query_value_out, bias=False)
        self.ours_p_up = nn.Linear(r, query_value_out, bias=False)
        for up in (self.ours_s_up, self.ours_p_up):
            nn.init.zeros_(up.weight)
            up.weight.requires_grad = False

        # designed frozen down projections (A), filled at each domain boundary
        self.register_buffer("s_down", torch.zeros(r, in_features))
        self.register_buffer("p_down", torch.zeros(r, in_features))
        # input-covariance accumulators driving the subspace design / energy merge
        self.register_buffer("cur_matrix", torch.zeros(in_features, in_features))
        self.register_buffer("prev_matrix", torch.zeros(in_features, in_features))

        self.cur_task = -1        # highest domain index whose branch is active in forward
        self.collecting = False   # when True, forward accumulates X^T X into cur_matrix
        self.slora_gamma = 0.5    # set from cfg by configure_model
        self.plora_gamma = 1.0

    def forward(self, input):
        out = self.linear_ours(input)
        if self.cur_task >= 0:
            # S (shared) is present from the first domain; P (task-specific) from the 2nd on.
            q, k, v = out.chunk(3, dim=-1)
            dq, dv = self.ours_s_up(F.linear(input, self.s_down)).chunk(2, dim=-1)
            q = q + self.slora_gamma * dq
            v = v + self.slora_gamma * dv
            if self.cur_task >= 1:
                dq, dv = self.ours_p_up(F.linear(input, self.p_down)).chunk(2, dim=-1)
                q = q + self.plora_gamma * dq
                v = v + self.plora_gamma * dv
            out = torch.cat([q, k, v], dim=-1)
        if self.collecting:
            x = input.detach().reshape(-1, self.in_features)
            self.cur_matrix = self.cur_matrix + x.t() @ x
        return out


def inject_trainable_ours(
    model: nn.Module,
    target_replace_module: List[str] = ["CrossAttention", "Attention"],
    r: int = 32,
    n_tasks: int = 15,
):
    """Inject Ours into the fused ``qkv`` Linear of every Attention module.

    The injected branches adapt only query and value; key and ``proj`` remain unchanged.
    Returns ``(require_grad_params, names)``. ``require_grad_params`` collects BOTH up
    branches of every layer: they all join the LoRA optimizer group but stay
    ``requires_grad=False`` until activated per-domain by the method (``_refresh_trainable``);
    a frozen up simply gets no gradient. The designed downs are buffers (non-trainable);
    ``linear_ours`` keeps its weight from the source model and is frozen via base LR 0.
    """

    require_grad_params = []
    names = []

    for _module in model.modules():
        if _module.__class__.__name__ in target_replace_module:
            for name, _child_module in _module.named_modules():
                if _child_module.__class__.__name__ == "Linear" and name == "qkv":

                    weight = _child_module.weight
                    bias = _child_module.bias
                    _tmp = OursInjectedLinear(
                        _child_module.in_features,
                        _child_module.out_features,
                        _child_module.bias is not None,
                        r,
                        n_tasks,
                    )
                    _tmp.linear_ours.weight = weight
                    if bias is not None:
                        _tmp.linear_ours.bias = bias

                    # switch the module
                    _module._modules[name] = _tmp

                    _injected = _module._modules[name]
                    require_grad_params.extend(list(_injected.ours_s_up.parameters()))
                    require_grad_params.extend(list(_injected.ours_p_up.parameters()))
                    names.append(name)

    return require_grad_params, names

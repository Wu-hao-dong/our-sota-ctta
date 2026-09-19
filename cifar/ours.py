"""Released Ours implementation.

This single module contains the core adapter, boundary-free detector,
and history-aware subspace specialization while preserving the public helper interface.
"""

from collections import OrderedDict
from copy import deepcopy

import hashlib
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit

from inject_ours import inject_trainable_ours, OursInjectedLinear

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- run config fingerprint
# Some method choices are selected by the leaf class rather than the YAML file. The runtime
# fingerprint records those choices together with the public configuration, making runs
# reproducible without relying on output filenames.
FINGERPRINT_VERSION = 1

# Ablation switch for the self-renewing cycle (paper Table 3). A domain boundary does two
# separable things -- CONSOLIDATE the finished domain's update into the frozen backbone, and
# REGENERATE a fresh subspace A for the new domain -- plus clearing B, which every mode that
# runs the boundary path at all does. The four modes are that 2x2:
#
#   mode               consolidate  regenerate  notes
#   full               yes          yes         the shipped method (unchanged code path)
#   consolidate_only   yes          no          A designed once at each branch's birth, then frozen
#   reset_only         no           yes         the finished domain's B is DISCARDED, backbone never written
#   fixed              no           no          boundary path is a no-op after the first call;
#                                               B accumulates across all 15 domains. S-branch only
#                                               (task never advances, so P never activates).
RENEW_MODES = ("full", "consolidate_only", "reset_only", "fixed")

# Ablation switch for the CROSS-SUBSET update structure (paper Table 4). Every mode below is the
# same GAO primitive -- ascent-perturb theta with subset S's CE gradient, then descend subset T's
# CE at the perturbed theta -- and differs ONLY in which subsets fill S and T, and how many times:
#
#   mode                       directions            isolates
#   within_batch_symmetric     A->B then B->A        THE SHIPPED METHOD (unchanged code path)
#   within_batch_oneway        A->B                  the cost of not alternating (1 step)
#   within_batch_oneway_twice  A->B then A->B        step-matched control for oneway: proves the
#                                                    gain is the ALTERNATION, not the extra step
#   same_subset                A->A then B->B        plain SAM per subset -- perturbation and
#                                                    descent on the SAME samples. This is the row
#                                                    that separates "sharpness-aware" from
#                                                    "cross-subset": identical step count and
#                                                    identical sample budget as symmetric, so the
#                                                    ONLY difference is S != T.
#   cross_batch_symmetric      prev->cur, cur->prev  the two subsets are two CONSECUTIVE BATCHES
#                                                    instead of two class-disjoint halves of one.
#                                                    Needs a 1-batch buffer => a mild online-CTTA
#                                                    deviation; report it as such, and note the
#                                                    HARD CONSTRAINT that it must not become the
#                                                    default (see the repo memory).
#
# The within-batch A/B split is pseudo-label-disjoint (_split_pseudo_label_groups), so "cross
# subset" also means "disjoint label sets" -- the perturbation direction cannot be fitted by the
# same samples it is then evaluated on, which is what breaks the pseudo-label confirmation loop.
GAO_UPDATES = ("within_batch_symmetric", "within_batch_oneway", "within_batch_oneway_twice",
               "same_subset", "cross_batch_symmetric")


def _resolve_as_design(m):
    """Source that designs the SHARED down-projection A_s.

    Branch order mirrors ``init_down_from_cov`` exactly (s_weight wins, then s_histalign, then
    use_shared_grad_init, else the covariance SVD default). Reports the CONFIGURED source; the two
    gradient-based sources fall back to the raw-gradient top-r at the first boundary, where no
    history exists yet.
    """
    if getattr(m, "s_weight", False):
        return "weight_svd"
    if getattr(m, "s_histalign", False):
        return "histalign"
    if getattr(m, "use_shared_grad_init", False):
        return "grad_svd"
    return "cov_svd"


def _resolve_ap_design(m):
    """Source that designs the task-specific down-projection A_p."""
    if not getattr(m, "use_plora", True):
        return "disabled"
    if getattr(m, "use_private_grad_init", False):
        thr = float(getattr(m, "private_history_proj_thr", 0.0) or 0.0)
        # A positive threshold projects the gradient away from the history subspace; zero
        # keeps the unprojected gradient-SVD variant.
        return "history_projected_grad" if thr > 0.0 else "gradient_svd"
    return "cov_whiten"


# --------------------------------------------------------------------------- adapter math
# Covariance-driven frozen down-projections and a closed-form energy merge of the trained
# up-projections into the backbone.
# Linear algebra is done in float64 on CPU for numerical stability (the Cholesky-whitening
# path is delicate) and cast back to the buffers' float32; it runs 12 layers x 15 domain
# boundaries, so the cost is negligible.

def _get_L(M, eps=1e-12, max_tries=100):
    """Cholesky factor of an SPD matrix M = L @ L.T, regularising M += eps*I (growing eps)
    until it is positive definite. Faithful port of the upstream _get_L."""
    eye = torch.eye(M.size(0), device=M.device, dtype=M.dtype)
    for _ in range(max_tries):
        try:
            return eps, torch.linalg.cholesky(M)
        except RuntimeError:
            M = M + eps * eye
            eps *= 10
    raise RuntimeError("Matrix not SPD even after regularization")


def init_down_from_cov(m, task, cur_cov, use_plora, lora_init_scale=3.0,
                       use_private_grad_init=False, p_grad=None, private_history_proj_thr=0.0,
                       use_shared_grad_init=False, s_weight=False, s_histalign=False):
    """Design the frozen down-projections (A) for the current domain from its input
    covariance, and zero the up-projections (B) for a fresh start.

      - ``s_down`` (shared): top-r left singular vectors of ``cur + prev`` (the energy the
        new domain shares with everything seen so far).
      - ``p_down`` (task-specific, domains >= 2 only): directions active for the new domain
        but inactive for past domains, via a Cholesky-whitening of ``cur`` w.r.t. ``prev``.
    Both rows are scaled by ``1/sqrt(lora_init_scale)`` exactly as upstream.

    ``use_private_grad_init=True`` designs ``p_down`` from a history-aware entropy gradient
    instead of Cholesky whitening: top-r left singular vectors of the merged backbone's
    per-domain averaged entropy gradient over the query/value output block (``p_grad`` is the
    averaged gradient of the fused
    qkv weight, [out, in]). When ``private_history_proj_thr>0`` the gradient is first projected
    onto the orthogonal complement of past domains' input principal subspace (top eigenvectors
    of ``prev_matrix`` covering that fraction of the S^2 energy) so
    ``p_down`` avoids directions past domains already used -- the orthogonalisation that (like
    the whitening) keeps P from collapsing onto the shared ``s_down`` subspace.

    ``use_shared_grad_init=True`` designs the SHARED ``s_down`` with the top-r left singular vectors of the
    RAW (un-projected) query/value entropy gradient -- the same ``p_grad`` used by ``p_down``,
    but before history projection -- instead of ``svd(cur+prev)``. With
    ``use_private_grad_init`` this makes both A matrices
    come from one entropy-gradient estimate (As = its dominant directions, Ap = its directions
    projected off the past subspace). Off by default (=covariance ``svd(cur+prev)``).
    """
    r = m.r
    cur = cur_cov.detach().to("cpu", torch.float64)
    prev = m.prev_matrix.detach().to("cpu", torch.float64)

    if s_weight:
        # As = top-r input-side principal directions of the merged backbone weight (no data, no
        # gradient). k rows are dropped exactly as in the gradient path where applicable.
        projection_dim = m.qkv_dim
        W = m.linear_ours.weight.detach()
        W_query_value = torch.cat([W[:projection_dim], W[2 * projection_dim:]], dim=0).to("cpu", torch.float64).t()  # [in, 2*projection_dim]
        Uw, _, _ = torch.linalg.svd(W_query_value, full_matrices=False)
        s_down = Uw[:, :r].t() / math.sqrt(lora_init_scale)        # (r, in)
    elif s_histalign and p_grad is not None:
        # As = top-r left singular vectors of the HISTORY-ALIGNED entropy gradient U U^T g
        # (U = past domains' input principal subspace, the SAME subspace pillar-3 removes for
        # p_down). As spans span(U), p_down spans its complement -> As _|_ Ap, i.e. the
        # shared/private split IS one gradient's history-aligned / history-residual split.
        # prev = 0 at the first boundary (k=0) -> no projection -> raw-gradient top-r fallback.
        projection_dim = m.qkv_dim
        g_query_value = torch.cat([p_grad[:projection_dim], p_grad[2 * projection_dim:]], dim=0).to("cpu", torch.float64).t()  # [in, 2*projection_dim]
        if private_history_proj_thr > 0.0:
            Up, Sp, _ = torch.linalg.svd(prev)
            tot = (Sp ** 2).sum()
            if tot > 0:
                ratio = (Sp ** 2) / tot
                k = int(torch.sum(torch.cumsum(ratio, 0) < private_history_proj_thr).item())
                if k > 0:
                    Uk = Up[:, :k]                                # [in, k] history subspace
                    g_query_value = Uk @ (Uk.t() @ g_query_value)                   # history-ALIGNED component
        Ug, _, _ = torch.linalg.svd(g_query_value, full_matrices=False)
        s_down = Ug[:, :r].t() / math.sqrt(lora_init_scale)        # (r, in)
    elif use_shared_grad_init and p_grad is not None:
        # As = top-r left singular vectors of the raw, unprojected query/value entropy gradient.
        projection_dim = m.qkv_dim
        g_query_value = torch.cat([p_grad[:projection_dim], p_grad[2 * projection_dim:]], dim=0).to("cpu", torch.float64).t()  # [in, 2*projection_dim]
        Ug, _, _ = torch.linalg.svd(g_query_value, full_matrices=False)
        s_down = Ug[:, :r].t() / math.sqrt(lora_init_scale)        # (r, in)
    else:
        U, _, _ = torch.linalg.svd(cur + prev)
        s_down = U[:, :r].t() / math.sqrt(lora_init_scale)         # (r, in)
    m.s_down.copy_(s_down.to(m.s_down.device, m.s_down.dtype))

    if task >= 1 and use_plora:
        if use_private_grad_init:
            projection_dim = m.qkv_dim
            g_query_value = torch.cat([p_grad[:projection_dim], p_grad[2 * projection_dim:]], dim=0).to("cpu", torch.float64).t()  # [in, 2*projection_dim]
            if private_history_proj_thr > 0.0:
                # Project the gradient away from past domains' input principal
                # subspace (top eigvecs of prev_matrix covering private_history_proj_thr of the
                # S^2 energy) before the SVD.
                Up, Sp, _ = torch.linalg.svd(prev)
                tot = (Sp ** 2).sum()
                if tot > 0:
                    ratio = (Sp ** 2) / tot
                    k = int(torch.sum(torch.cumsum(ratio, 0) < private_history_proj_thr).item())
                    if k > 0:
                        Uk = Up[:, :k]                            # [in, k] past subspace
                        g_query_value = g_query_value - Uk @ (Uk.t() @ g_query_value)        # orthogonal complement
            Ug, _, _ = torch.linalg.svd(g_query_value, full_matrices=False)
            p_down = Ug[:, :r].t() / math.sqrt(lora_init_scale)   # (r, in)
            m.p_down.copy_(p_down.to(m.p_down.device, m.p_down.dtype))
        else:
            _, L = _get_L(prev)                                    # prev = L @ L.T
            Linv = torch.linalg.inv(L)
            A = Linv @ cur @ Linv.t()                             # whitened current cov
            V_res, _, _ = torch.linalg.svd(A)
            U_res = torch.linalg.inv(L.t()) @ V_res[:, :r]
            U_res, _ = torch.linalg.qr(U_res)
            p_down = U_res[:, :r].t() / math.sqrt(lora_init_scale) # (r, in)
            m.p_down.copy_(p_down.to(m.p_down.device, m.p_down.dtype))
    else:
        m.p_down.zero_()

    nn.init.zeros_(m.ours_s_up.weight)
    nn.init.zeros_(m.ours_p_up.weight)


def _energy_merge(B_new, A_fixed, prev_mat, cur_mat, gamma, eps, out_type):
    """Closed-form (gated) fold of a low-rank update ``B_new @ A_fixed`` into a weight delta.

    ``merge``: per design direction i, gate the update by ``eta_i = g_prev /
    (g_prev + gamma*g_cur + eps)`` where ``g_* = a_i P_* a_i^T`` are the projection energies
    of direction i onto the previous / current input covariances -- directions that carry a
    lot of OLD energy are damped (recalibrated toward the feature-level joint optimum).
    ``add``: ungated (used for the first domain's S and every task-specific P).
    Returns ``(delta, mean_eta)`` with ``mean_eta=None`` for the ``add`` path.
    """
    r = A_fixed.shape[0]
    P_prev = prev_mat
    P_cur = cur_mat
    out_cols = []
    etas = []
    for i in range(r):
        if out_type == "merge":
            a_i = A_fixed[i].unsqueeze(0)
            g_prev = (a_i @ P_prev @ a_i.t()).clamp_min(0.0).squeeze()
            g_cur = (a_i @ P_cur @ a_i.t()).clamp_min(0.0).squeeze()
            eta = g_prev / (g_prev + float(gamma) * g_cur + eps)
            out_cols.append((1.0 - eta) * B_new[:, i])
            etas.append(eta)
        else:  # 'add'
            out_cols.append(B_new[:, i])
    delta = torch.stack(out_cols, dim=1).contiguous() @ A_fixed    # (out, in)
    mean_eta = torch.stack(etas).mean() if etas else None
    return delta, mean_eta


def _expand_query_value_up_to_qkv(B_query_value, out_features):
    """Expand a [query | value] up-projection matrix to [query | key | value] with zero k rows."""
    assert out_features % 3 == 0, "expects a fused qkv Linear (out = 3*dim)"
    qkv_dim = out_features // 3
    assert B_query_value.shape[0] == 2 * qkv_dim, "expects Ours up rows to be [query | value]"
    zeros_k = B_query_value.new_zeros(qkv_dim, B_query_value.shape[1])
    return torch.cat([B_query_value[:qkv_dim], zeros_k, B_query_value[qkv_dim:]], dim=0)


def energy_merge_into_base(m, task, cur_cov, merge_gamma, eps, use_slora, use_plora,
                           force_s_add=False):
    """End-of-domain consolidation: fold the trained query/value ups (gated for S on domains >= 2,
    plain for the rest) into ``linear_ours.weight`` and zero the ups. The fused k slice gets
    a zero delta. Mirrors upstream after_task. Returns the mean S gate (eta) for logging, or
    None.

    ``force_s_add=True`` DISABLES the S-branch energy-merge gating: S is folded in ungated
    (``"add"`` -- a plain ``B_s @ A_s`` sum of rank-1 outer products, exactly like the task-
    specific P branch) on every domain, with no ``(1 - eta_i)`` per-direction damping. Used by
    the ONLINE variant; the oracle method keeps the default (gated) behaviour."""
    W0 = m.linear_ours.weight.data
    cur = cur_cov.to(W0.device, W0.dtype)
    prev = m.prev_matrix.to(W0.device, W0.dtype)
    delta = torch.zeros_like(W0)
    mean_eta = None
    if use_slora:
        B_s_query_value = (m.slora_gamma * m.ours_s_up.weight.detach()).to(W0.device, W0.dtype)
        B_s = _expand_query_value_up_to_qkv(B_s_query_value, W0.shape[0])
        s_type = "add" if force_s_add else ("merge" if task >= 1 else "add")
        d_s, mean_eta = _energy_merge(B_s, m.s_down.to(W0.device, W0.dtype),
                                      prev, cur, merge_gamma, eps, s_type)
        delta = delta + d_s
    if use_plora and task >= 1:
        B_p_query_value = (m.plora_gamma * m.ours_p_up.weight.detach()).to(W0.device, W0.dtype)
        B_p = _expand_query_value_up_to_qkv(B_p_query_value, W0.shape[0])
        d_p, _ = _energy_merge(B_p, m.p_down.to(W0.device, W0.dtype),
                               prev, cur, merge_gamma, eps, "add")
        delta = delta + d_p
    m.linear_ours.weight.data.copy_(W0 + delta)
    nn.init.zeros_(m.ours_s_up.weight)
    nn.init.zeros_(m.ours_p_up.weight)
    return mean_eta


@torch.no_grad()
def export_merged_backbone(model, inplace=False):
    """DEPLOYMENT EXPORT: fold every Ours branch into its base qkv weight and drop the
    adapter modules, returning a model structurally identical to the source ViT.
    Mirror of the imagenet tree's function (the two trees do not share code).

    Each injected layer's forward is ``W0 x`` plus ``gamma_s * B_s (A_s x)`` (and, from
    domain 1 on, ``gamma_p * B_p (A_p x)``) added to the q and v slices -- every term is
    LINEAR in the input, so the whole branch collapses to a weight delta

        dW = gamma_s * expand_query_value(B_s @ A_s) [+ gamma_p * expand_query_value(B_p @ A_p)]

    with zero rows on k. This holds at ANY point in the stream, not only right after a
    consolidation (where the ups are zero and dW vanishes): mid-domain the branch is still
    exactly foldable. That is what makes the deployment cost equal to the source model's
    rather than "equal at boundaries only" -- see bench_export.py, which asserts it.

    NOTE this is a plain algebraic fold (``B @ A``), NOT the method's energy-gated
    consolidation (``energy_merge_into_base``): it reproduces the CURRENT forward exactly,
    whereas the gated merge deliberately damps directions before committing them. Use this
    for deployment snapshots and cost measurement, never inside the adaptation loop.

    ``inplace=False`` (default) deep-copies first, so the live adapting model is untouched.
    """
    tgt = model if inplace else deepcopy(model)
    # collect first: we mutate the parents' _modules while walking them
    targets = [(parent, name, child)
               for parent in tgt.modules()
               for name, child in parent.named_children()
               if isinstance(child, OursInjectedLinear)]
    for parent, name, m in targets:
        W = m.linear_ours.weight.data.clone()
        if m.cur_task >= 0:
            B_s = m.ours_s_up.weight.detach().to(W.dtype)
            W += m.slora_gamma * _expand_query_value_up_to_qkv(B_s @ m.s_down.to(W.dtype), W.shape[0])
            if m.cur_task >= 1:
                B_p = m.ours_p_up.weight.detach().to(W.dtype)
                W += m.plora_gamma * _expand_query_value_up_to_qkv(B_p @ m.p_down.to(W.dtype), W.shape[0])
        has_bias = m.linear_ours.bias is not None
        lin = nn.Linear(m.in_features, m.out_features, bias=has_bias).to(W.device, W.dtype)
        lin.weight.data.copy_(W)
        if has_bias:
            lin.bias.data.copy_(m.linear_ours.bias.data)
        parent._modules[name] = lin
    return tgt


# ----------------------------------------------------------------------------- the method
class _OursBase(nn.Module):
    """Core continual test-time adaptation implementation.

    Each corruption segment is treated as one adaptation domain, and the domains run in
    the benchmark's fixed evaluation order. The continual machinery is driven from
    ``prepare_domain``, called by
    the evaluate loop at every domain boundary (after the domain's data is loaded):
      - the just-finished domain's update is closed-form energy-merged into the frozen
        ``qkv`` weight and its covariance rolled into ``prev_matrix``;
      - a no-grad pre-pass estimates the new domain's first-batch input covariance;
      - the new domain's frozen down-projections are DESIGNED (S from SVD of ``cur+prev``,
        P from a Cholesky-whitened generalised-eigen of ``cur`` w.r.t. ``prev``);
      - only the new domain's up-projections are then trainable.

    Since CTTA has no labels, the method uses an optional high-confidence pseudo-label GAO
    step on the current logits, with plain entropy minimisation (Tent) as the fallback -- no
    mean teacher. The backbone is frozen (base LR 0); the query and value slices of the
    ``qkv`` weight evolve only through the merge.

    reset() is only called at the first corruption (i_x == 0), before any merge, so the
    strict state_dict load matches the pristine injected model (ups=0, downs=0, prev=0,
    cur_task=-1); it additionally clears the per-domain continual state.
    """

    def __init__(self, model, optimizer, steps=1, episodic=False, rank=32,
                 slora_gamma=0.5, plora_gamma=1.0, merge_gamma=3.0, lora_eps=1e-5,
                 use_slora=True, use_plora=True, avg=False, cov_batch_size=50,
                 use_gao=False, gao_conf_thr=0.7, gao_rho=0.3, gao_min_samples=2,
                 use_private_grad_init=False, private_history_proj_thr=0.0, use_shared_grad_init=False,
                 s_weight=False, s_histalign=False, renew_mode="full",
                 gao_update="within_batch_symmetric"):
        super().__init__()
        assert renew_mode in RENEW_MODES, \
            "renew_mode must be one of {}, got {!r}".format(sorted(RENEW_MODES), renew_mode)
        assert gao_update in GAO_UPDATES, \
            "gao_update must be one of {}, got {!r}".format(sorted(GAO_UPDATES), gao_update)
        # Which subsets fill the GAO ascent/descent slots (see GAO_UPDATES and _gao_step).
        # "within_batch_symmetric" is the shipped method and the only mode whose update path is
        # byte-for-byte the pre-ablation code.
        self.gao_update = gao_update
        # one-batch buffer (high-confidence images + pseudo labels) used ONLY by
        # cross_batch_symmetric; dropped at every domain boundary so no stale-domain samples
        # ever cross a switch
        self._xb_prev = None
        # Which halves of the self-renewing cycle run at a domain boundary (see prepare_domain
        # and RENEW_MODES). "full" is the shipped method and the only mode whose boundary path
        # is byte-for-byte the pre-ablation code.
        self.renew_mode = renew_mode
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "_OursBase requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        self.rank = rank
        self.merge_gamma = merge_gamma
        self.lora_eps = lora_eps
        self.use_slora = use_slora
        self.use_plora = use_plora
        # When True, the end-of-domain merge folds the S branch in UNGATED (like P), i.e. the
        # energy-merge gating is disabled. Default False (gated) = oracle behaviour; the ONLINE
        # subclass flips this on.
        self.s_merge_ungated = False
        self.cov_batch_size = cov_batch_size
        self.use_gao = use_gao
        self.gao_conf_thr = gao_conf_thr
        self.gao_rho = gao_rho
        self.gao_min_samples = gao_min_samples
        # History-aware private-subspace initialization: gradient SVD followed by an optional
        # projection away from the historical subspace. Off by default (covariance whitening).
        self.use_private_grad_init = use_private_grad_init
        self.private_history_proj_thr = private_history_proj_thr
        # Shared-subspace initialization from the raw, unprojected entropy-gradient SVD top-r
        # (needs p_grad, so the gradient is estimated at every boundary including domain 0).
        # Off by default (=svd(cur+prev)).
        self.use_shared_grad_init = use_shared_grad_init
        # As from the MERGED BACKBONE WEIGHT (no data, no gradient). Off by default.
        self.s_weight = s_weight
        # As from the HISTORY-ALIGNED entropy-gradient component U U^T g -- As _|_ Ap by
        # construction. Needs p_grad every boundary; takes precedence over use_shared_grad_init.
        # SHIPPED default for ours (matches the imagenet tree).
        self.s_histalign = s_histalign
        self.task = -1            # current adaptation-domain index
        self.prev_cov = None      # stashed per-layer first-batch covariance of the domain

        # push the contribution scales onto every injected layer (avg halving as upstream)
        sg, pg = slora_gamma, plora_gamma
        if use_slora and use_plora and avg:
            sg, pg = sg * 0.5, pg * 0.5
        for m in self._ours_modules(self.model):
            m.slora_gamma = sg
            m.plora_gamma = pg

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

    def config_fingerprint(self, extra=None):
        """Ordered record of every field that determines this run's method identity.

        Covers the code-level choices the cfg dump cannot show (A-design sources, S-merge
        gating, boundary source, and the forward-compatible renew/update-structure slots).
        ``extra`` carries run-level fields that live in cfg rather than on this object (seed,
        dataset, learning rates). Returns an ``OrderedDict`` whose first key is a short stable
        hash over all the others -- runs sharing a hash share a configuration.
        """
        mods = self._ours_modules(self.model)
        # effective gammas (after the `avg` halving) are pushed onto the injected layers in
        # __init__, so read them back from the model rather than from the ctor arguments
        sg = float(mods[0].slora_gamma) if mods else float("nan")
        pg = float(mods[0].plora_gamma) if mods else float("nan")

        fp = OrderedDict()
        fp["fp_ver"] = FINGERPRINT_VERSION
        fp["class"] = type(self).__name__
        fp["As_design"] = _resolve_as_design(self)
        fp["Ap_design"] = _resolve_ap_design(self)
        fp["proj_thr"] = getattr(self, "private_history_proj_thr", 0.0)
        fp["rank"] = self.rank
        fp["n_injected"] = len(mods)
        fp["use_slora"] = self.use_slora
        fp["use_plora"] = self.use_plora
        fp["slora_gamma"] = sg
        fp["plora_gamma"] = pg
        fp["merge_gamma"] = self.merge_gamma
        fp["s_merge_ungated"] = self.s_merge_ungated
        # forward-compatible slots: these read a constant until the boundary-mode (A1) and
        # update-structure (A2) ablation switches land, then start reporting the real variant
        # without any further change here
        fp["renew_mode"] = getattr(self, "renew_mode", "full")
        fp["gao_update"] = getattr(self, "gao_update", "within_batch_symmetric")
        fp["use_gao"] = self.use_gao
        fp["gao_conf_thr"] = self.gao_conf_thr
        fp["gao_rho"] = self.gao_rho
        fp["gao_min_samples"] = self.gao_min_samples
        fp["steps"] = self.steps
        fp["episodic"] = self.episodic
        fp["cov_batch"] = self.cov_batch_size
        # the online change-detector attributes only exist on the boundary-free subclass
        if hasattr(self, "switch_thr"):
            # the oracle-boundary control keeps every detector field (it still runs read-only)
            # but takes its boundaries from the known domain schedule; the extra key makes the
            # two configurations hash differently while leaving plain online hashes unchanged
            _op = getattr(self, "switch_oracle_period", 0)
            if _op > 0:
                fp["boundary"] = "oracle_schedule"
                fp["oracle_period"] = _op
            else:
                fp["boundary"] = "online_detector"
            fp["switch_metric"] = self.switch_metric
            fp["switch_thr"] = self.switch_thr
            fp["switch_gap"] = self.switch_gap
            fp["switch_layers"] = ",".join(str(i) for i in self.switch_layers)
            fp["switch_subspace_r"] = self.switch_subspace_r
            fp["proto_ema"] = self.proto_ema
        else:
            fp["boundary"] = "oracle"
        for k, v in (extra or {}).items():
            fp[str(k)] = v

        payload = ";".join("{}={}".format(k, v) for k, v in fp.items())
        fp["hash"] = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
        fp.move_to_end("hash", last=False)
        return fp

    def log_config_fingerprint(self, extra=None):
        """Emit the fingerprint as ONE greppable line, before adaptation starts.

        Recover a run's configuration with
        ``grep -h 'Ours-fingerprint' output/*.txt``; group identical configurations by ``hash=``.
        """
        fp = self.config_fingerprint(extra)
        logger.info("Ours-fingerprint %s",
                    " ".join("{}={}".format(k, v) for k, v in fp.items()))
        return fp

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x, self.model, self.optimizer)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        # clear Ours continual state (buffers were restored to zero by the strict load above;
        # cur_task / requires_grad are not in state_dict)
        self.task = -1
        self.prev_cov = None
        self._xb_prev = None
        for m in self._ours_modules(self.model):
            m.cur_task = -1
            m.ours_s_up.weight.requires_grad = False
            m.ours_p_up.weight.requires_grad = False

    # ------------------------------------------------------------------ helpers
    def _ours_modules(self, model):
        """Ours-injected layers in a fixed module order (shared by cov collection, down
        design and merging, so per-layer state lines up)."""
        return [m for m in model.modules() if isinstance(m, OursInjectedLinear)]

    @torch.enable_grad()
    def _estimate_query_value_grad(self, x):
        """Average softmax-entropy gradient of each injected layer's base ``qkv`` weight.

        This is measured over the domain inputs immediately after the merge, when the live
        up-projections are zero. Base weights receive gradients only transiently (they remain
        LR-0 in the optimizer); the up-projections are frozen for this pass
        and restored by _refresh_trainable afterwards. Returns per-layer [out, in] cpu grads."""
        mods = self._ours_modules(self.model)
        for m in mods:
            m.linear_ours.weight.requires_grad = True
            m.ours_s_up.weight.requires_grad = False
            m.ours_p_up.weight.requires_grad = False
        self.model.zero_grad(set_to_none=True)
        grad_sum = [None] * len(mods)
        num = 0
        bs = self.cov_batch_size
        was_training = self.model.training
        self.model.eval()
        for i in range(0, x.shape[0], bs):
            self.model.zero_grad(set_to_none=True)
            out = self.model(x[i:i + bs])
            loss = softmax_entropy(out).mean(0)
            loss.backward()
            for j, m in enumerate(mods):
                g = m.linear_ours.weight.grad
                if g is None:
                    continue
                grad_sum[j] = g.detach().float().cpu() if grad_sum[j] is None \
                    else grad_sum[j] + g.detach().float().cpu()
            num += 1
        if was_training:
            self.model.train()
        self.model.zero_grad(set_to_none=True)
        for m in mods:
            m.linear_ours.weight.requires_grad = False
        return [None if g is None else g / max(num, 1) for g in grad_sum]

    @torch.no_grad()
    def _collect_cov(self, x):
        """No-grad pre-pass over the new domain's first batch, accumulating each injected
        layer's input covariance (raw X^T X). All ups are zero at this point (just merged),
        so the forward equals the merged backbone. Returns the per-layer covariances."""
        mods = self._ours_modules(self.model)
        for m in mods:
            m.cur_matrix.zero_()
            m.collecting = True
        was_training = self.model.training
        self.model.eval()
        bs = min(self.cov_batch_size, x.shape[0])
        self.model(x[:bs])
        if was_training:
            self.model.train()
        cov = [m.cur_matrix.detach().clone() for m in mods]
        for m in mods:
            m.collecting = False
            m.cur_matrix.zero_()
        return cov

    def _refresh_trainable(self):
        """Make only the current domain's ups trainable (S always; P from domain 2 on);
        clear optimizer momentum for the swapped params so each domain starts fresh."""
        for m in self._ours_modules(self.model):
            m.ours_s_up.weight.requires_grad = self.use_slora
            m.ours_p_up.weight.requires_grad = self.use_plora and (self.task >= 1)
            self.optimizer.state.pop(m.ours_s_up.weight, None)
            self.optimizer.state.pop(m.ours_p_up.weight, None)

    def _trainable_ours_params(self, model):
        """Currently active Ours up-projections; GAO perturbations must not touch base/K."""
        return [p for name, p in model.named_parameters()
                if "ours_" in name and p.requires_grad]

    def _split_pseudo_label_groups(self, confidence, pseudo_labels):
        """Build two high-confidence pseudo-label-disjoint subsets for GAO."""
        keep = confidence >= self.gao_conf_thr
        min_samples = max(int(self.gao_min_samples), 1)
        if keep.sum().item() < 2 * min_samples:
            return None, None

        labels = torch.unique(pseudo_labels[keep], sorted=True)
        if labels.numel() < 2:
            return None, None

        mask_a = torch.zeros_like(keep, dtype=torch.bool)
        mask_b = torch.zeros_like(keep, dtype=torch.bool)
        for i, label in enumerate(labels):
            label_mask = keep & (pseudo_labels == label)
            if i % 2 == 0:
                mask_a = mask_a | label_mask
            else:
                mask_b = mask_b | label_mask

        if mask_a.sum().item() < min_samples or mask_b.sum().item() < min_samples:
            return None, None
        return mask_a.nonzero(as_tuple=False).flatten(), mask_b.nonzero(as_tuple=False).flatten()

    def _entropy_step(self, outputs, optimizer):
        loss = softmax_entropy(outputs).mean(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    def _gao_direction(self, x_src, y_src, x_tgt, y_tgt, model, optimizer, params,
                       src_logits=None):
        """ONE GAO direction, the primitive every update structure is built from.

        Ascent-perturb theta by the normalized CE gradient of the SOURCE subset, descend the
        TARGET subset's CE at the perturbed theta, restore theta, and take one optimizer step.
        ``src_logits`` lets the caller reuse logits already computed at the current theta;
        pass None whenever a previous direction has since stepped. Returns True if a
        perturbation existed (i.e. the direction actually ran)."""
        inner_logits = model(x_src) if src_logits is None else src_logits
        inner_loss = F.cross_entropy(inner_logits, y_src)
        grads = torch.autograd.grad(inner_loss, params, allow_unused=True)
        # Drop the inner graph before the outer forward. autograd.grad frees the backward
        # buffers, but these locals keep the ViT's activations alive, so without this the inner
        # and outer graphs coexist -- which on a 16GB card at batch 50 x 384px is enough to put
        # the caching allocator into thrash (measured: cross_batch_symmetric went 30s -> 6.5min
        # per 2-domain smoke run before this line existed).
        del inner_logits, inner_loss

        perturbations = []
        has_perturbation = False
        with torch.no_grad():
            for param, grad in zip(params, grads):
                if grad is None:
                    perturbations.append(None)
                    continue
                delta = self.gao_rho * grad / (grad.norm() + 1e-12)
                param.add_(delta)
                perturbations.append(delta)
                has_perturbation = True

        if not has_perturbation:
            return False

        try:
            optimizer.zero_grad()
            outer_loss = F.cross_entropy(model(x_tgt), y_tgt)
            outer_loss.backward()
        finally:
            with torch.no_grad():
                for param, delta in zip(params, perturbations):
                    if delta is not None:
                        param.sub_(delta)

        optimizer.step()
        optimizer.zero_grad()
        return True

    def _gao_step(self, x, outputs, pseudo_labels, idx_a, idx_b, model, optimizer):
        """Pseudo-label GAO over the two within-batch subsets, in whichever structure
        ``gao_update`` selects (see GAO_UPDATES).

        The shipped structure is symmetric A->B then B->A: each subset acts as an independent
        perturbation direction for the other's descent, which de-correlates the update and
        regularizes the noisy pseudo labels. The A<->B ALTERNATION is what helps, not the extra
        step -- ``within_batch_oneway_twice`` is the control that shows it. Returns True if at
        least one direction ran."""
        params = self._trainable_ours_params(model)
        if not params:
            return False

        xa, ya = x.index_select(0, idx_a), pseudo_labels.index_select(0, idx_a)
        xb, yb = x.index_select(0, idx_b), pseudo_labels.index_select(0, idx_b)
        # A's logits at the current (still unstepped) theta are already in `outputs`
        la = outputs.index_select(0, idx_a)
        mode = self.gao_update

        if mode == "within_batch_oneway":
            return self._gao_direction(xa, ya, xb, yb, model, optimizer, params, src_logits=la)

        if mode == "within_batch_oneway_twice":
            # the SAME direction twice: matches symmetric's step count and compute while keeping
            # the perturbation source fixed, so any gap to symmetric is the alternation alone
            ran1 = self._gao_direction(xa, ya, xb, yb, model, optimizer, params, src_logits=la)
            ran2 = self._gao_direction(xa, ya, xb, yb, model, optimizer, params)
            return ran1 or ran2

        if mode == "same_subset":
            # plain SAM on each subset: perturb and descend on the SAME samples. Same two steps
            # and the same total samples as symmetric -- only S != T is removed.
            ran1 = self._gao_direction(xa, ya, xa, ya, model, optimizer, params, src_logits=la)
            ran2 = self._gao_direction(xb, yb, xb, yb, model, optimizer, params)
            return ran1 or ran2

        # "within_batch_symmetric" (shipped): A->B, then B->A at the theta direction 1 produced
        ran_a = self._gao_direction(xa, ya, xb, yb, model, optimizer, params, src_logits=la)
        ran_b = self._gao_direction(xb, yb, xa, ya, model, optimizer, params)
        return ran_a or ran_b

    def _gao_cross_batch_step(self, x, outputs, confidence, pseudo_labels, model, optimizer):
        """symmetric x cross-batch GAO: the two subsets are two CONSECUTIVE BATCHES rather than
        two class-disjoint halves of one batch.

        Both perturbations are recomputed fresh at the current sequential theta (no stored
        epsilon). The previous batch's high-confidence samples are buffered; the buffer is
        dropped at every domain boundary, and a batch with no buffered predecessor (the first of
        a domain, or one whose predecessor had too few confident samples) falls back to Tent.
        This needs a one-batch data buffer, which is a mild deviation from strictly online CTTA
        and must be stated wherever this row is reported."""
        params = self._trainable_ours_params(model)
        min_samples = max(int(self.gao_min_samples), 1)
        keep = confidence >= self.gao_conf_thr

        cur = None
        if keep.sum().item() >= min_samples:
            idx = keep.nonzero(as_tuple=False).flatten()
            # Cap each subset at half the batch, keeping the most confident samples. Two
            # reasons, and the second is the important one:
            #  (a) memory -- uncapped, both directions run on ~the whole batch, so two ~50-image
            #      graphs coexist and a 16GB card thrashes;
            #  (b) COMPARABILITY -- the within-batch modes split the batch into two roughly
            #      half-sized class-disjoint subsets, so an uncapped cross-batch row would feed
            #      each direction ~2x the samples and confound "which subsets" (the thing this
            #      table is about) with "how many samples". The cap makes the subset SIZE match
            #      and leaves the subset CHOICE as the only difference.
            cap = max((x.shape[0] + 1) // 2, min_samples)
            if idx.numel() > cap:
                order = torch.argsort(confidence.index_select(0, idx), descending=True)
                idx = idx.index_select(0, order[:cap])
            cur = (x.index_select(0, idx).detach(), pseudo_labels.index_select(0, idx).detach())
        prev = self._xb_prev
        # hand the current batch to the next step regardless of whether this one can run
        self._xb_prev = cur

        if not params or cur is None or prev is None:
            return False
        x_prev, y_prev = prev
        x_cur, y_cur = cur
        # dir1: perturb from PREV, descend on CUR. src_logits is None because `outputs` covers
        # the current batch, not the previous one.
        ran1 = self._gao_direction(x_prev, y_prev, x_cur, y_cur, model, optimizer, params)
        # dir2: perturb from CUR (recomputed at the theta dir1 produced), descend on PREV
        ran2 = self._gao_direction(x_cur, y_cur, x_prev, y_prev, model, optimizer, params)
        return ran1 or ran2

    def prepare_domain(self, x):
        """Domain boundary (= new adaptation domain). Called by the evaluate loop for every domain
        (including the first) right after the domain's data is loaded.

        ``renew_mode`` decomposes the self-renewing cycle into its two independent halves
        (see RENEW_MODES): CONSOLIDATE (fold B@A into the frozen backbone) and REGENERATE
        (design a fresh A for the new domain). Clearing B is not separable -- every mode
        that runs this path at all starts the new domain from B=0."""
        mode = self.renew_mode
        # cross_batch_symmetric's one-batch buffer must never span a boundary: perturbing with
        # the OLD domain's samples is exactly the staleness the boundary exists to end. Dropped
        # before the early return so 'fixed' clears it too.
        self._xb_prev = None
        # 'fixed': one subspace, designed at the very first boundary, then never touched
        # again -- the boundary path becomes a no-op, so B accumulates across all 15 domains
        # and nothing is ever folded into the backbone. NOTE the task index never advances,
        # so cur_task stays 0 and the P branch never activates: this row is inherently
        # S-branch-only, and the paper text must say so rather than read it as "same model,
        # frozen subspace".
        if mode == "fixed" and self.task >= 0:
            return
        do_consolidate = mode in ("full", "consolidate_only")
        do_regenerate = mode in ("full", "reset_only")
        # The non-regenerating modes still have to design each branch ONCE, at its birth
        # (S at task 0, P at task 1) -- otherwise s_down/p_down stay all-zero and the branch
        # is dead, which would confound "no renewal" with "no branch". 'fixed' only ever
        # reaches this path at task 0 (it returns early afterwards), so for it "birth" is the
        # single design it gets.
        birth_only = mode in ("consolidate_only", "fixed")

        mods = self._ours_modules(self.model)
        # (1) finish the just-ended domain: energy-merge its update, roll its covariance
        eta_means = []
        if self.task >= 0:
            for i, m in enumerate(mods):
                if do_consolidate:
                    eta = energy_merge_into_base(m, self.task, self.prev_cov[i],
                                                 self.merge_gamma, self.lora_eps,
                                                 self.use_slora, self.use_plora,
                                                 force_s_add=self.s_merge_ungated)
                    if eta is not None:
                        eta_means.append(eta)
                else:
                    # 'reset_only': the finished domain's update is DISCARDED, not folded --
                    # the frozen backbone stays at its source weights for the whole stream
                    nn.init.zeros_(m.ours_s_up.weight)
                    nn.init.zeros_(m.ours_p_up.weight)
                # covariance history is part of REGENERATE's input, not of consolidation,
                # so it rolls in every mode that reaches here
                m.prev_matrix.add_(self.prev_cov[i].to(m.prev_matrix.device, m.prev_matrix.dtype))
        # (1b) Estimate the merged backbone's entropy gradient for private-subspace initialization.
        # (ups are 0 post-merge). Only when the NEW task will be >= 1 (self.task is still the OLD
        # index here; task 0 zeros p_down, so skip the wasted pass at the first boundary).
        need_grad = do_regenerate or (birth_only and self.task + 1 <= 1)
        grads = self._estimate_query_value_grad(x) if (need_grad and
                                              ((self.use_private_grad_init and self.task >= 0)
                                               or (self.use_shared_grad_init and self.task >= -1)
                                               or (self.s_histalign and self.task >= -1))) else None
        # (2) first-batch input covariance of the new domain
        cov = self._collect_cov(x)
        # (3) advance the task index
        self.task += 1
        # (4) design the frozen downs for the new domain; activate the adapter (ups still 0)
        # NOTE the init is SKIPPED entirely past the births in the non-regenerating modes --
        # not called-then-reverted. init_down_from_cov dereferences p_grad unconditionally in
        # the use_private_grad_init branch, and p_grad is None there (no gradient pass was run), so
        # calling it would raise. Skipping is also what makes those modes cheap.
        do_init = do_regenerate or (birth_only and self.task <= 1)
        for i, m in enumerate(mods):
            if do_init:
                # at P's birth (task 1) the call would also re-design an already-born s_down,
                # so snapshot and put it back
                s_keep = m.s_down.clone() if (birth_only and self.task >= 1) else None
                init_down_from_cov(m, self.task, cov[i], self.use_plora,
                                   use_private_grad_init=self.use_private_grad_init,
                                   p_grad=(grads[i] if grads is not None else None),
                                   private_history_proj_thr=self.private_history_proj_thr,
                                   use_shared_grad_init=self.use_shared_grad_init, s_weight=self.s_weight,
                                   s_histalign=self.s_histalign)
                if s_keep is not None:
                    m.s_down.copy_(s_keep)
            m.cur_task = self.task
        # (5) stash this domain's covariance for the next boundary's merge
        self.prev_cov = cov
        # (6) make only this domain's ups trainable
        self._refresh_trainable()
        # eta is only produced by the GATED S merge (domains >= 2); domain 0 has nothing to
        # merge and domain 1 merges domain-0's S ungated ('add'), so both report n/a.
        if eta_means:
            eta_str = "{:.3f}".format(sum(e.item() for e in eta_means) / len(eta_means))
        else:
            eta_str = "n/a"
        logger.info("Ours: entered domain %d (use_slora=%s use_plora=%s); prev-domain gated mean-eta=%s",
                    self.task, self.use_slora, self.use_plora, eta_str)

    @torch.enable_grad()  # ensure grads in possible no-grad context for testing
    def forward_and_adapt(self, x, model, optimizer):
            # Online update of the current domain's up-projections. GAO uses only
        # high-confidence pseudo labels; otherwise we keep the original Tent fallback.
        outputs = model(x)
        if not self.use_gao:
            self._entropy_step(outputs, optimizer)
            return outputs

        confidence, pseudo_labels = outputs.detach().softmax(1).max(1)

        if self.gao_update == "cross_batch_symmetric":
            # its two subsets are whole batches, so it never uses the within-batch A/B split --
            # and it never differentiates the current batch's own logits either, so the
            # full-batch graph built above is dead weight. Dropping it matters: it would
            # otherwise sit in VRAM alongside both per-direction graphs, and at batch 50 x 384px
            # on a 16GB card that combination thrashes the caching allocator (measured 9x
            # slower per batch). The caller only argmaxes this tensor.
            outputs = outputs.detach()
            if not self._gao_cross_batch_step(x, outputs, confidence, pseudo_labels,
                                              model, optimizer):
                self._entropy_step(model(x), optimizer)
            return outputs

        idx_a, idx_b = self._split_pseudo_label_groups(confidence, pseudo_labels)
        if idx_a is None:
            self._entropy_step(outputs, optimizer)
            return outputs

        if not self._gao_step(x, outputs, pseudo_labels, idx_a, idx_b, model, optimizer):
            self._entropy_step(model(x), optimizer)
        return outputs


@torch.jit.script
def softmax_entropy(x):  # -> torch.Tensor:
    """Standard Shannon entropy of the softmax (Tent objective)."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def collect_params(model):
    """Split params into (backbone group, ours group) by the 'ours_' name convention.
    Only the up-projections (ours_s_up / ours_p_up) carry the marker; the designed downs and
    covariances are buffers and so appear in neither group."""
    ours_params_list = []
    model_params_lst = []
    for name, param in model.named_parameters():
        if 'ours_' in name:
            ours_params_list.append(param)
        else:
            model_params_lst.append(param)
    return model_params_lst, ours_params_list


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model, cfg):
    """Inject Ours adapters into fused qkv projections and prepare the model for adaptation.

    The backbone is frozen through a base learning rate of zero.

    cifar convention (vs the imagenet original): the entry script has ALREADY wrapped the
    model in DataParallel and loaded the source checkpoint, so no wrap here, and the ckpt
    (vit_base_384_cifar10.t7) is {'model': state_dict-with-module.-prefixes}. The strict=False
    re-load refreshes base params after injection; the injected qkv.linear_ours reuses the
    already-loaded weight Parameter, so keys skipped by the rename stay correct."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.cpu()
    ours_params, ours_names = inject_trainable_ours(model=model,
            target_replace_module=["CrossAttention", "Attention"],
            r=cfg.TEST.ours_rank, n_tasks=len(cfg.CORRUPTION.TYPE))
    if cfg.TEST.ckpt != None:
        checkpoint = torch.load(cfg.TEST.ckpt)
        model.load_state_dict(checkpoint['model'], strict=False)
    model.to(device)
    model.train()
    return model


def check_model(model):
    """Check model for compatability."""
    is_training = model.training
    assert is_training, "needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    assert has_any_params, "needs params to update: check which require grad"


class _OursOnline(_OursBase):
    """Boundary-free Ours driven by a covariance change detector."""

    def __init__(self, *args, switch_thr=0.254, switch_gap=6,
                 switch_layers=(9, 10, 11), switch_metric="frobenius",
                 switch_subspace_r=64, proto_ema=0.2,
                 switch_oracle_period=0, **kwargs):
        super().__init__(*args, **kwargs)
        # At approximate online boundaries, fold the shared branch into the
        # backbone without the covariance-energy gate, just like the private branch.
        self.s_merge_ungated = True
        assert switch_metric in ("frobenius", "subspace"), switch_metric
        self.switch_thr = float(switch_thr)
        self.switch_gap = int(switch_gap)
        self.switch_metric = switch_metric
        self.switch_subspace_r = int(switch_subspace_r)
        self.proto_ema = float(proto_ema)
        self.switch_oracle_period = int(switch_oracle_period)

        mods = self._ours_modules(self.model)
        self.switch_layers = [i for i in switch_layers if i < len(mods)]
        self._fp_mods = [mods[i] for i in self.switch_layers]

        self._proto = None
        self._seg = 0
        self._last = -10 ** 9
        self._t = 0

    def prepare_domain(self, x):
        """Ignore oracle boundary callbacks from the evaluation loop."""
        pass

    def reset(self):
        super().reset()
        self._proto = None
        self._seg = 0
        self._last = -10 ** 9
        self._t = 0

    @torch.no_grad()
    def _fingerprint(self, x):
        """Collect trace-normalized qkv-input covariances."""
        bs = min(self.cov_batch_size, x.shape[0])
        for m in self._fp_mods:
            m.cur_matrix.zero_()
            m.collecting = True
        was_training = self.model.training
        self.model.eval()
        self.model(x[:bs])
        if was_training:
            self.model.train()
        fp = []
        for m in self._fp_mods:
            C = m.cur_matrix.detach().double()
            fp.append((C / C.diagonal().sum()).float().cpu())
            m.cur_matrix.zero_()
            m.collecting = False
        return fp

    def _top_r_subspace(self, C):
        r = min(self.switch_subspace_r, C.shape[0])
        _, V = torch.linalg.eigh(C.double())
        return V[:, -r:]

    def _distance(self, fp, proto):
        if self.switch_metric == "frobenius":
            return sum(
                torch.linalg.matrix_norm(a - b).item()
                for a, b in zip(fp, proto)
            )
        d = 0.0
        for a, b in zip(fp, proto):
            Ua, Ub = self._top_r_subspace(a), self._top_r_subspace(b)
            r = Ua.shape[1]
            overlap = (torch.linalg.matrix_norm(Ua.t() @ Ub).item() ** 2) / r
            d += 1.0 - overlap
        return d

    def forward(self, x):
        fp = self._fingerprint(x)
        d = None if self._proto is None else self._distance(fp, self._proto)
        if self.switch_oracle_period > 0:
            switched = (self._t % self.switch_oracle_period == 0)
            if d is not None:
                logger.info(
                    "Ours-oracle: batch %d dist=%.4f thr=%.4f would_fire=%d true=%d",
                    self._t, d, self.switch_thr, int(d > self.switch_thr),
                    int(switched),
                )
        elif self._t == 0 or self._proto is None:
            switched = True
        else:
            switched = (
                (d > self.switch_thr)
                and (self._t - self._last >= self.switch_gap)
            )

        if switched:
            super().prepare_domain(x)
            self._proto = [c.clone() for c in fp]
            self._seg = 1
            self._last = self._t
            logger.info(
                "Ours-%s: switch at batch %d (task=%d, metric=%s)",
                "oracle" if self.switch_oracle_period > 0 else "online",
                self._t, self.task, self.switch_metric,
            )
        else:
            a = self.proto_ema
            self._proto = [
                (1.0 - a) * p + a * c for p, c in zip(self._proto, fp)
            ]
            self._seg += 1
        self._t += 1
        return super().forward(x)


class Ours(_OursOnline):
    """Use history-aligned shared A and history-residual private A."""

    def __init__(self, *args, history_proj_thr=0.99, **kwargs):
        kwargs["use_private_grad_init"] = True
        kwargs["private_history_proj_thr"] = history_proj_thr
        kwargs["s_histalign"] = True
        super().__init__(*args, **kwargs)

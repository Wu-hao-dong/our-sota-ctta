from collections import OrderedDict
import logging
import math

import torch
import torch.optim as optim

from robustbench.data import load_imagenetc
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy as accuracy

import tent
import cotta
import ours

from conf import cfg, load_cfg_fom_args


logger = logging.getLogger(__name__)



def evaluate(description):
    args = load_cfg_fom_args(description)
    # configure model
    base_model = load_model(cfg.MODEL.ARCH, cfg.CKPT_DIR,
                       cfg.CORRUPTION.DATASET, ThreatModel.corruptions).cuda()
    if cfg.MODEL.ADAPTATION == "source":
        logger.info("test-time adaptation: NONE")
        model = setup_source(base_model)
    if cfg.MODEL.ADAPTATION == "tent":
        logger.info("test-time adaptation: TENT")
        model = setup_tent(base_model)
    if cfg.MODEL.ADAPTATION == "cotta":
        logger.info("test-time adaptation: CoTTA")
        model = setup_cotta(base_model)
    if cfg.MODEL.ADAPTATION == "ours":
        logger.info("test-time adaptation: Ours")
        model = setup_ours(args, base_model)

    # Repeat the corruption stream without resetting between rounds. ROUNDS=1 is the
    # standard single-pass CTTA protocol used by the released configuration.
    all_error = []
    round_domain_error = []
    for round_idx in range(cfg.CORRUPTION.ROUNDS):
        round_error = []
        for severity in cfg.CORRUPTION.SEVERITY:
            for corruption_idx, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
                stream_start = round_idx == 0 and corruption_idx == 0
                if stream_start:
                    try:
                        model.reset()
                        logger.info("resetting model (start of continual stream)")
                    except Exception:
                        logger.warning("not resetting model")
                else:
                    logger.warning("not resetting model")

                x_test, y_test = load_imagenetc(cfg.CORRUPTION.NUM_EX,
                                                severity, cfg.DATA_DIR, False,
                                                [corruption_type])
                x_test, y_test = x_test.cuda(), y_test.cuda()
                acc = accuracy(model, x_test, y_test, cfg.TEST.BATCH_SIZE)
                err = 1. - acc
                all_error.append(err)
                round_error.append(err)
                logger.info(
                    f"error % [round {round_idx + 1} {corruption_type}{severity}]: {err:.2%}"
                )

        round_domain_error.append(round_error)
        logger.info(
            f"[round {round_idx + 1}] mean error % : "
            f"{sum(round_error) / len(round_error):.2%}"
        )

    logger.info(
        f"mean error % (all {cfg.CORRUPTION.ROUNDS} rounds): "
        f"{sum(all_error) / len(all_error):.2%}"
    )
    if cfg.CORRUPTION.ROUNDS > 1:
        logger.info("==== summary (mean error per round) ====")
        for round_idx, domain_error in enumerate(round_domain_error):
            logger.info(
                f"round {round_idx + 1:2d}: "
                f"mean={sum(domain_error) / len(domain_error):.2%}"
            )


def setup_source(model):
    """Set up the baseline source model without adaptation."""
    model.eval()
    logger.info(f"model for evaluation: %s", model)
    return model

def setup_tent(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = tent.configure_model(model)
    params, param_names = tent.collect_params(model)
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model


def setup_optimizer(params):
    """Set up optimizer for tent adaptation.

    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.

    For best results, try tuning the learning rate and batch size.
    """
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam(params,
                    lr=cfg.OPTIM.LR,
                    betas=(cfg.OPTIM.BETA, 0.999),
                    weight_decay=cfg.OPTIM.WD)
    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD(params,
                   lr=cfg.OPTIM.LR,
                   momentum=0.9,
                   dampening=0,
                   weight_decay=cfg.OPTIM.WD,
                   nesterov=True)
    else:
        raise NotImplementedError

def setup_cotta(model):
    """Set up tent adaptation.

    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = cotta.configure_model(model)
    params, param_names = cotta.collect_params(model)
    optimizer = setup_optimizer(params)
    cotta_model = cotta.CoTTA(model, optimizer,
                           steps=cfg.OPTIM.STEPS,
                           episodic=cfg.MODEL.EPISODIC)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return cotta_model

def _ours_run_extra():
    """Run-level fields included in the method's reproducibility fingerprint."""
    return OrderedDict([
        ("dataset", cfg.CORRUPTION.DATASET),
        ("arch", cfg.MODEL.ARCH),
        ("seed", cfg.RNG_SEED),
        ("num_ex", cfg.CORRUPTION.NUM_EX),
        ("batch", cfg.TEST.BATCH_SIZE),
        ("rounds", cfg.CORRUPTION.ROUNDS),
        ("lr", cfg.OPTIM.LR),
        ("ours_lr", cfg.OPTIM.OursLR),
    ])


def _oracle_period():
    """Return batches per domain for the optional oracle-boundary control."""
    if not cfg.OPTIM.Ours_switch_oracle:
        return 0
    return math.ceil(cfg.CORRUPTION.NUM_EX / cfg.TEST.BATCH_SIZE)


def setup_ours(args, model):
    """Set up the released boundary-free Ours method."""
    model = ours.configure_model(model, cfg)
    model_param, ours_param = ours.collect_params(model)
    optimizer = setup_optimizer_ours(
        model_param, ours_param, cfg.OPTIM.LR, cfg.OPTIM.OursLR
    )
    ours_model = ours.Ours(
        model,
        optimizer,
        steps=cfg.OPTIM.STEPS,
        episodic=cfg.MODEL.EPISODIC,
        rank=cfg.TEST.ours_rank,
        slora_gamma=cfg.OPTIM.Ours_slora_gamma,
        plora_gamma=cfg.OPTIM.Ours_plora_gamma,
        merge_gamma=cfg.OPTIM.Ours_merge_gamma,
        lora_eps=cfg.OPTIM.Ours_lora_eps,
        use_slora=cfg.OPTIM.Ours_use_slora,
        use_plora=cfg.OPTIM.Ours_use_plora,
        avg=cfg.OPTIM.Ours_avg,
        cov_batch_size=cfg.TEST.BATCH_SIZE,
        use_gao=cfg.OPTIM.Ours_use_gao,
        gao_conf_thr=cfg.OPTIM.Ours_GAO_conf_thr,
        gao_rho=cfg.OPTIM.Ours_GAO_rho,
        gao_min_samples=cfg.OPTIM.Ours_GAO_min_samples,
        gao_update=cfg.OPTIM.Ours_gao_update,
        switch_thr=cfg.OPTIM.Ours_switch_thr,
        switch_gap=cfg.OPTIM.Ours_switch_gap,
        switch_layers=tuple(cfg.TEST.ours_switch_layers),
        switch_metric=cfg.OPTIM.Ours_switch_metric,
        switch_subspace_r=cfg.TEST.ours_switch_subspace_r,
        proto_ema=cfg.OPTIM.Ours_proto_ema,
        history_proj_thr=cfg.OPTIM.Ours_history_proj_thr,
        switch_oracle_period=_oracle_period(),
        renew_mode=cfg.OPTIM.Ours_renew_mode,
    )
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    ours_model.log_config_fingerprint(_ours_run_extra())
    return ours_model


def setup_optimizer_ours(params, adapter_params, model_lr, adapter_lr):
    if cfg.OPTIM.METHOD == 'Adam':
        return optim.Adam([{"params": params, "lr": model_lr},
                                  {"params": adapter_params, "lr": adapter_lr}],
                                 lr=1e-5, betas=(cfg.OPTIM.BETA, 0.999),weight_decay=cfg.OPTIM.WD)

    elif cfg.OPTIM.METHOD == 'SGD':
        return optim.SGD([{"params": params, "lr": model_lr},
                                  {"params": adapter_params, "lr": adapter_lr}],
                                    momentum=cfg.OPTIM.MOMENTUM,dampening=cfg.OPTIM.DAMPENING,
                                    nesterov=cfg.OPTIM.NESTEROV,
                                 lr=1e-5,weight_decay=cfg.OPTIM.WD)
    else:
        raise NotImplementedError
if __name__ == '__main__':
    evaluate('"Imagenet-C evaluation.')

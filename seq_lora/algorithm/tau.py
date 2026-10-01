from __future__ import annotations

from typing import Callable, Dict, List, Tuple
import math


MetricDict = Dict[str, float]
TauEvalFn = Callable[[float, int, str], MetricDict]


def _kl_in_window(metrics: MetricDict, *, kl_low: float, kl_high: float) -> bool:
    kl = float(metrics["kl_map_to_bayes"])
    return float(kl_low) <= kl <= float(kl_high)


def _kl_window_distance(metrics: MetricDict, *, kl_low: float, kl_high: float) -> float:
    kl = float(metrics["kl_map_to_bayes"])
    if kl < float(kl_low):
        return float(kl_low) - kl
    if kl > float(kl_high):
        return kl - float(kl_high)
    return 0.0


def fit_posterior_scale_from_anchor_kl(
    eval_tau: TauEvalFn,
    *,
    tau_max: float,
    anchor_n_samples: int,
    kl_target_low: float,
    kl_target_high: float,
) -> Dict[str, float]:
    """Select Seq-LoRA's posterior-scale factor from train-anchor KL.

    The caller supplies ``eval_tau(tau, n_samples, desc)``, which evaluates the
    posterior predictive distribution at a candidate scale and returns metrics
    containing ``acc_bayes``, ``nll_bayes``, and ``kl_map_to_bayes``.
    """
    tau_max_arg = float(tau_max)
    if tau_max_arg <= 0.0:
        raise ValueError(f"tau_search_max must be positive, got {tau_max_arg}")
    tau_max = tau_max_arg

    kl_target_low = float(kl_target_low)
    kl_target_high = float(kl_target_high)
    if not (0.0 <= kl_target_low <= kl_target_high):
        raise ValueError(
            f"Expected 0 <= tau_kl_target_low <= tau_kl_target_high, "
            f"got {kl_target_low}, {kl_target_high}"
        )

    baseline = eval_tau(0.0, 1, "TAU-AUTO ref tau=0")
    acc0 = float(baseline["acc_bayes"])
    nll0 = float(baseline["nll_bayes"])
    records: List[Tuple[float, MetricDict, str]] = [(0.0, baseline, "baseline")]
    kl_target = kl_target_low
    probe_tau = min(0.5, tau_max)
    eps = 1e-12

    print(
        "\n=== Auto-selecting posterior_tau by direct anchor KL estimate ===\n"
        f"[Tau auto] baseline tau=0.000000 "
        f"Acc0={acc0*100:.2f}% NLL0={nll0:.6f} KL0={baseline['kl_map_to_bayes']:.6f}\n"
        f"[Tau auto] target: "
        f"{kl_target_low:.6f} <= KL_tau <= {kl_target_high:.6f} "
        f"(direct_target={kl_target:.6f}) "
        f"using KL(tau) ~= C * tau^2 within [0, {tau_max:.6f}]",
        flush=True,
    )

    def _select(tau: float, metrics: MetricDict, source: str) -> Dict[str, float]:
        print(
            f"[Tau auto] selected posterior_tau={float(tau):.6f} "
            f"source={source} "
            f"anchor_acc={metrics['acc_bayes']*100:.2f}% "
            f"anchor_nll={metrics['nll_bayes']:.6f} "
            f"anchor_kl={metrics['kl_map_to_bayes']:.6f} "
            f"kl_window=[{kl_target_low:.6f}, {kl_target_high:.6f}]",
            flush=True,
        )
        return {
            "optimal_posterior_tau": float(tau),
            "baseline_acc": float(acc0),
            "baseline_nll": float(nll0),
            "selected_acc": float(metrics["acc_bayes"]),
            "selected_nll": float(metrics["nll_bayes"]),
            "selected_kl_map_to_bayes": float(metrics["kl_map_to_bayes"]),
            "kl_target": float(kl_target),
            "kl_target_low": float(kl_target_low),
            "kl_target_high": float(kl_target_high),
            "tau_max": float(tau_max),
            "anchor_n_samples": int(anchor_n_samples),
        }

    probe_metrics = eval_tau(
        probe_tau,
        int(anchor_n_samples),
        f"TAU-AUTO probe tau={probe_tau:.4f}",
    )
    probe_kl = float(probe_metrics["kl_map_to_bayes"])
    probe_in_window = _kl_in_window(
        probe_metrics,
        kl_low=kl_target_low,
        kl_high=kl_target_high,
    )
    records.append((float(probe_tau), probe_metrics, "probe"))
    print(
        f"[Tau auto] probe tau={probe_tau:.6f} "
        f"acc={probe_metrics['acc_bayes']*100:.2f}% "
        f"nll={probe_metrics['nll_bayes']:.6f} "
        f"kl={probe_kl:.6f} "
        f"kl_in_window={int(probe_in_window)}",
        flush=True,
    )
    if probe_in_window:
        return _select(probe_tau, probe_metrics, "probe_kl_window")

    if probe_kl <= eps:
        tau_hat = tau_max
    else:
        tau_hat = probe_tau * math.sqrt(max(kl_target, 0.0) / max(probe_kl, eps))
        tau_hat = min(max(float(tau_hat), 0.0), tau_max)
    print(
        f"[Tau auto] direct estimate from probe: "
        f"tau={tau_hat:.6f} = {probe_tau:.6f} * sqrt({kl_target:.6f} / {max(probe_kl, eps):.6f})",
        flush=True,
    )

    if abs(float(tau_hat) - float(probe_tau)) <= 1e-8:
        tau_hat_metrics = probe_metrics
    else:
        tau_hat_metrics = eval_tau(
            tau_hat,
            int(anchor_n_samples),
            f"TAU-AUTO direct tau={tau_hat:.4f}",
        )
        records.append((float(tau_hat), tau_hat_metrics, "direct"))
    tau_hat_in_window = _kl_in_window(
        tau_hat_metrics,
        kl_low=kl_target_low,
        kl_high=kl_target_high,
    )
    print(
        f"[Tau auto] direct tau={tau_hat:.6f} "
        f"acc={tau_hat_metrics['acc_bayes']*100:.2f}% "
        f"nll={tau_hat_metrics['nll_bayes']:.6f} "
        f"kl={tau_hat_metrics['kl_map_to_bayes']:.6f} "
        f"kl_in_window={int(tau_hat_in_window)}",
        flush=True,
    )
    if tau_hat_in_window:
        return _select(tau_hat, tau_hat_metrics, "direct_kl_window")

    tau_hat_kl = float(tau_hat_metrics["kl_map_to_bayes"])
    if tau_hat_kl > eps and tau_hat > 0.0:
        tau_refined = tau_hat * math.sqrt(max(kl_target, 0.0) / max(tau_hat_kl, eps))
        tau_refined = min(max(float(tau_refined), 0.0), tau_max)
        print(
            f"[Tau auto] one-step KL refinement: "
            f"tau={tau_refined:.6f} = {tau_hat:.6f} * sqrt({kl_target:.6f} / {tau_hat_kl:.6f})",
            flush=True,
        )
        if abs(float(tau_refined) - float(tau_hat)) > 1e-8:
            refined_metrics = eval_tau(
                tau_refined,
                int(anchor_n_samples),
                f"TAU-AUTO refined tau={tau_refined:.4f}",
            )
            records.append((float(tau_refined), refined_metrics, "refined"))
            refined_in_window = _kl_in_window(
                refined_metrics,
                kl_low=kl_target_low,
                kl_high=kl_target_high,
            )
            print(
                f"[Tau auto] refined tau={tau_refined:.6f} "
                f"acc={refined_metrics['acc_bayes']*100:.2f}% "
                f"nll={refined_metrics['nll_bayes']:.6f} "
                f"kl={refined_metrics['kl_map_to_bayes']:.6f} "
                f"kl_in_window={int(refined_in_window)}",
                flush=True,
            )
            if refined_in_window:
                return _select(tau_refined, refined_metrics, "refined_kl_window")

            refined_kl = float(refined_metrics["kl_map_to_bayes"])
            direct_kl = float(tau_hat_metrics["kl_map_to_bayes"])
            if (
                min(direct_kl, refined_kl) <= kl_target_low
                and max(direct_kl, refined_kl) >= kl_target_high
                and abs(direct_kl - refined_kl) > eps
            ):
                if direct_kl <= refined_kl:
                    tau_low, kl_low_actual = float(tau_hat), direct_kl
                    tau_high, kl_high_actual = float(tau_refined), refined_kl
                else:
                    tau_low, kl_low_actual = float(tau_refined), refined_kl
                    tau_high, kl_high_actual = float(tau_hat), direct_kl
                tau_interp = tau_low + (
                    (kl_target_low - kl_low_actual)
                    / max(kl_high_actual - kl_low_actual, eps)
                    * (tau_high - tau_low)
                )
                tau_interp = min(max(float(tau_interp), 0.0), tau_max)
                print(
                    f"[Tau auto] linear KL interpolation: tau={tau_interp:.6f} "
                    f"from ({tau_low:.6f}, kl={kl_low_actual:.6f}) "
                    f"to ({tau_high:.6f}, kl={kl_high_actual:.6f}) "
                    f"target_kl={kl_target_low:.6f}; selecting without extra eval",
                    flush=True,
                )
                interp_metrics = dict(refined_metrics if refined_kl < direct_kl else tau_hat_metrics)
                interp_metrics["kl_map_to_bayes"] = float(kl_target_low)
                return _select(tau_interp, interp_metrics, "interp_linear_kl_no_eval")

    selected_tau, selected_metrics, selected_source = min(
        records,
        key=lambda item: _kl_window_distance(
            item[1],
            kl_low=kl_target_low,
            kl_high=kl_target_high,
        ),
    )
    return _select(float(selected_tau), selected_metrics, f"{selected_source}_closest_kl")

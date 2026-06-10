import html
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .common import plot_line_with_band, sanitize_name
from .stats import add_stats_dataframe, add_stats_value

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCHAEFER_ORDER_DIR = Path(
    "/TODO_SET_PATH/order"
)


def load_roi_labels(atlas_names, roi_counts):
    if len(atlas_names) == 1 and atlas_names[0].startswith("schaefer"):
        parcel_count = int(atlas_names[0].replace("schaefer", ""))
        order_fp = SCHAEFER_ORDER_DIR / f"Schaefer2018_{parcel_count}Parcels_7Networks_order.txt"
        crosswalk_fp = SCHAEFER_ORDER_DIR / f"atlas-Schaefer2018v0143_desc-{parcel_count}ParcelsAllNetworks_dseg.tsv"
        if order_fp.exists() and crosswalk_fp.exists():
            order = pd.read_csv(order_fp, sep="\t", header=None).iloc[:roi_counts[0], 1]
            crosswalk = pd.read_csv(crosswalk_fp, sep="\t")
            label_map = dict(zip(crosswalk["label_7network"], crosswalk["label_17network"]))
            labels = order.map(label_map)
            assert not labels.isnull().any()
            return labels.str.replace("^17Networks_", "", regex=True).tolist()

    labels = []
    for atlas_name, count in zip(atlas_names, roi_counts):
        labels.extend([f"{atlas_name}:{i:03d}" for i in range(int(count))])
    return labels


def _window_psd(x, tr_seconds):
    x = np.asarray(x, dtype=np.float32)
    x = np.transpose(x, (0, 2, 1))
    x = x - x.mean(axis=-1, keepdims=True)
    spec = np.fft.rfft(x, axis=-1)
    power = (spec.real ** 2 + spec.imag ** 2) / x.shape[-1]
    freq = np.fft.rfftfreq(x.shape[-1], d=float(tr_seconds))
    return freq, power


def _autocorr_lags(windows, max_lag=5):
    out = []
    for lag in range(1, max_lag + 1):
        vals = []
        for window in windows:
            a = window[:, :-lag]
            b = window[:, lag:]
            a = a - a.mean(axis=1, keepdims=True)
            b = b - b.mean(axis=1, keepdims=True)
            num = np.sum(a * b, axis=1)
            den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1) + 1e-8)
            vals.append(np.mean(num / den))
        out.append(vals)
    return np.asarray(out, dtype=np.float32)


def _fc_upper(window):
    fc = np.corrcoef(window.T)
    iu = np.triu_indices(fc.shape[0], k=1)
    return fc[iu]


def _mmd_rbf(x, y):
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    max_n = 2048
    if x.shape[0] > max_n:
        rng = np.random.default_rng(0)
        x = x[rng.choice(x.shape[0], size=max_n, replace=False)]
    if y.shape[0] > max_n:
        rng = np.random.default_rng(1)
        y = y[rng.choice(y.shape[0], size=max_n, replace=False)]

    z = np.concatenate([x, y], axis=0)
    z_norm = np.sum(z * z, axis=1, keepdims=True)
    d2 = z_norm + z_norm.T - 2.0 * (z @ z.T)
    d2 = np.maximum(d2, 0.0)
    sigma2 = np.median(d2[d2 > 0])
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = 1.0

    x_norm = np.sum(x * x, axis=1, keepdims=True)
    y_norm = np.sum(y * y, axis=1, keepdims=True)
    d_xx = np.maximum(x_norm + x_norm.T - 2.0 * (x @ x.T), 0.0)
    d_yy = np.maximum(y_norm + y_norm.T - 2.0 * (y @ y.T), 0.0)
    d_xy = np.maximum(x_norm + y_norm.T - 2.0 * (x @ y.T), 0.0)

    k_xx = np.exp(-d_xx / sigma2)
    k_yy = np.exp(-d_yy / sigma2)
    k_xy = np.exp(-d_xy / sigma2)
    return float(k_xx.mean() + k_yy.mean() - 2 * k_xy.mean())


def _corr_1d(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    if den <= 0:
        return 0.0
    return float(np.sum(a * b) / den)


def _r2_1d(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    den = np.sum((y_true - y_true.mean()) ** 2)
    if den <= 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / den)


def _condition_groups(condition_names):
    groups = {}
    for idx, name in enumerate(condition_names):
        groups.setdefault(str(name), []).append(int(idx))
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in groups.items()}


def _mean_fc_summary(windows):
    fc_vecs = np.stack([_fc_upper(window) for window in windows], axis=0).astype(np.float32)
    return float(fc_vecs.mean()), float(fc_vecs.std()), fc_vecs.mean(axis=0)


def compute_window_stats(windows, tr_seconds):
    freq, psd = _window_psd(windows, tr_seconds)
    roi_var = windows.var(axis=1)
    ac = _autocorr_lags(windows.transpose(0, 2, 1), max_lag=5)
    return {
        "freq": freq.astype(np.float32),
        "psd_mean": psd.mean(axis=(0, 1)).astype(np.float32),
        "roi_var_mean": roi_var.mean(axis=0).astype(np.float32),
        "roi_var_std": roi_var.std(axis=0).astype(np.float32),
        "ac_mean": ac.mean(axis=1).astype(np.float32),
        "ac_std": ac.std(axis=1).astype(np.float32),
    }


def _compute_level1_metrics_core(real_future, synth_future, tr_seconds):
    real_stats = compute_window_stats(real_future, tr_seconds)
    synth_stats = compute_window_stats(synth_future, tr_seconds)

    ks_vals = []
    for roi_idx in range(real_future.shape[-1]):
        real_roi = real_future[:, :, roi_idx].reshape(-1)
        synth_roi = synth_future[:, :, roi_idx].reshape(-1)
        ks_vals.append(float(ks_2samp(real_roi, synth_roi).statistic))
    ks_vals = np.asarray(ks_vals, dtype=np.float32)

    fc_corr = []
    for real_window, synth_window in zip(real_future, synth_future):
        real_vec = _fc_upper(real_window)
        synth_vec = _fc_upper(synth_window)
        fc_corr.append(float(np.corrcoef(real_vec, synth_vec)[0, 1]))
    fc_corr = np.asarray(fc_corr, dtype=np.float32)
    real_fc_mean, real_fc_std, real_fc_vec = _mean_fc_summary(real_future)
    synth_fc_mean, synth_fc_std, synth_fc_vec = _mean_fc_summary(synth_future)

    return {
        "freq": real_stats["freq"],
        "real_psd_mean": real_stats["psd_mean"],
        "synth_psd_mean": synth_stats["psd_mean"],
        "real_var_mean": real_stats["roi_var_mean"],
        "real_var_std": real_stats["roi_var_std"],
        "synth_var_mean": synth_stats["roi_var_mean"],
        "synth_var_std": synth_stats["roi_var_std"],
        "real_ac_mean": real_stats["ac_mean"],
        "real_ac_std": real_stats["ac_std"],
        "synth_ac_mean": synth_stats["ac_mean"],
        "synth_ac_std": synth_stats["ac_std"],
        "ks_per_roi": ks_vals,
        "mean_ks": float(ks_vals.mean()),
        "mmd": _mmd_rbf(real_future.reshape(-1, real_future.shape[-1]), synth_future.reshape(-1, synth_future.shape[-1])),
        "paired_fc_corr_mean": float(fc_corr.mean()),
        "paired_fc_corr_std": float(fc_corr.std()),
        "paired_fc_corr_all": fc_corr,
        "real_mean_fc_edge_weight": real_fc_mean,
        "real_std_fc_edge_weight": real_fc_std,
        "real_fc_mean_vec": real_fc_vec.astype(np.float32),
        "synth_mean_fc_edge_weight": synth_fc_mean,
        "synth_std_fc_edge_weight": synth_fc_std,
        "synth_fc_mean_vec": synth_fc_vec.astype(np.float32),
    }


def compute_level1_metrics(real_future, synth_future, tr_seconds, condition_names):
    by_condition = {}
    for name, indices in _condition_groups(condition_names).items():
        by_condition[name] = _compute_level1_metrics_core(real_future[indices], synth_future[indices], tr_seconds)
    return {"overall": _compute_level1_metrics_core(real_future, synth_future, tr_seconds), "by_condition": by_condition}


def _compute_averaged_metrics_core(real_future, rollout_bank, counts):
    counts = sorted(int(x) for x in counts)
    curves = []
    ranked = {}
    for n in counts:
        mean_rollout = rollout_bank[:n].mean(axis=0)
        mse = float(np.mean((mean_rollout - real_future) ** 2))
        r_vals = []
        r2_vals = []
        for roi_idx in range(real_future.shape[-1]):
            real_roi = real_future[:, :, roi_idx].reshape(-1)
            pred_roi = mean_rollout[:, :, roi_idx].reshape(-1)
            r_vals.append(_corr_1d(real_roi, pred_roi))
            r2_vals.append(_r2_1d(real_roi, pred_roi))
        r_vals = np.asarray(r_vals, dtype=np.float32)
        r2_vals = np.asarray(r2_vals, dtype=np.float32)
        ranked[n] = {"r": r_vals, "r2": r2_vals}
        curves.append({"n": n, "mse": mse, "mean_r": float(r_vals.mean()), "mean_r2": float(r2_vals.mean())})

    final_n = counts[-1]
    mean_rollout = rollout_bank[:final_n].mean(axis=0)
    mse_by_t = ((mean_rollout - real_future) ** 2).mean(axis=2)
    r_by_t = []
    for t in range(real_future.shape[1]):
        vals = []
        for roi_idx in range(real_future.shape[2]):
            vals.append(_corr_1d(real_future[:, t, roi_idx], mean_rollout[:, t, roi_idx]))
        r_by_t.append(np.asarray(vals, dtype=np.float32))
    r_by_t = np.stack(r_by_t, axis=0)
    return {
        "curves": pd.DataFrame(curves),
        "ranked": ranked,
        "by_t": pd.DataFrame({
            "t": np.arange(real_future.shape[1], dtype=np.int64),
            "mse_mean": mse_by_t.mean(axis=0).astype(np.float32),
            "mse_std": mse_by_t.std(axis=0).astype(np.float32),
            "mean_r": r_by_t.mean(axis=1).astype(np.float32),
            "std_r": r_by_t.std(axis=1).astype(np.float32),
        }),
    }


def compute_averaged_metrics(real_future, rollout_bank, counts, condition_names):
    by_condition = {}
    for name, indices in _condition_groups(condition_names).items():
        by_condition[name] = _compute_averaged_metrics_core(real_future[indices], rollout_bank[:, indices], counts)
    return {"overall": _compute_averaged_metrics_core(real_future, rollout_bank, counts), "by_condition": by_condition}


def save_grid_timeseries_stats(bundle, stats_prefix, real_future, synth_future):
    real_roi_var = real_future.var(axis=1)
    synth_roi_var = synth_future.var(axis=1)
    r_by_roi = np.asarray([
        _corr_1d(real_future[:, :, roi_idx].reshape(-1), synth_future[:, :, roi_idx].reshape(-1))
        for roi_idx in range(real_future.shape[2])
    ], dtype=np.float32)
    r_by_t = np.asarray([
        [
            _corr_1d(real_future[:, t, roi_idx], synth_future[:, t, roi_idx])
            for roi_idx in range(real_future.shape[2])
        ]
        for t in range(real_future.shape[1])
    ], dtype=np.float32)
    summary = {
        "real_roi_time_var_global_mean": float(real_roi_var.mean()),
        "synth_roi_time_var_global_mean": float(synth_roi_var.mean()),
        "forecast_mean_r": float(r_by_roi.mean()),
        "forecast_std_r": float(r_by_roi.std()),
    }
    add_stats_value(bundle, stats_prefix, "timeseries", "real_roi_time_var_mean", value=real_roi_var.mean(axis=0).astype(np.float32))
    add_stats_value(bundle, stats_prefix, "timeseries", "real_roi_time_var_std", value=real_roi_var.std(axis=0).astype(np.float32))
    add_stats_value(bundle, stats_prefix, "timeseries", "synth_roi_time_var_mean", value=synth_roi_var.mean(axis=0).astype(np.float32))
    add_stats_value(bundle, stats_prefix, "timeseries", "synth_roi_time_var_std", value=synth_roi_var.std(axis=0).astype(np.float32))
    add_stats_value(bundle, stats_prefix, "timeseries", "forecast_r_by_roi", value=r_by_roi)
    add_stats_value(bundle, stats_prefix, "timeseries", "forecast_mean_r_by_t", value=r_by_t.mean(axis=1).astype(np.float32))
    add_stats_value(bundle, stats_prefix, "timeseries", "forecast_std_r_by_t", value=r_by_t.std(axis=1).astype(np.float32))
    for key, value in summary.items():
        add_stats_value(bundle, stats_prefix, "timeseries", key, value=value)
    return summary


def _store_level1_stats(bundle, stats_prefix, metrics, roi_labels):
    overall = metrics["overall"]
    for key, value in overall.items():
        add_stats_value(bundle, stats_prefix, "overall", key, value=value)
    top_idx = np.argsort(overall["ks_per_roi"])[-25:][::-1]
    add_stats_value(bundle, stats_prefix, "overall", "top_ks_roi", value=top_idx.astype(np.int64))
    add_stats_value(bundle, stats_prefix, "overall", "top_ks", value=overall["ks_per_roi"][top_idx].astype(np.float32))
    add_stats_value(bundle, stats_prefix, "overall", "top_ks_label", value=np.asarray([roi_labels[i] for i in top_idx], dtype=str))

    condition_names = []
    for name, curr in metrics["by_condition"].items():
        condition_key = sanitize_name(name)
        condition_names.append(str(name))
        add_stats_value(bundle, stats_prefix, "by_condition", condition_key, "name", value=str(name))
        for key, value in curr.items():
            add_stats_value(bundle, stats_prefix, "by_condition", condition_key, key, value=value)
        top_idx = np.argsort(curr["ks_per_roi"])[-25:][::-1]
        add_stats_value(bundle, stats_prefix, "by_condition", condition_key, "top_ks_roi", value=top_idx.astype(np.int64))
        add_stats_value(bundle, stats_prefix, "by_condition", condition_key, "top_ks", value=curr["ks_per_roi"][top_idx].astype(np.float32))
        add_stats_value(
            bundle,
            stats_prefix,
            "by_condition",
            condition_key,
            "top_ks_label",
            value=np.asarray([roi_labels[i] for i in top_idx], dtype=str),
        )
    add_stats_value(bundle, stats_prefix, "condition_names", value=np.asarray(condition_names, dtype=str))


def save_level1(task_dir, task, metrics, roi_labels, bundle=None, stats_prefix=None):
    task_dir.mkdir(parents=True, exist_ok=True)
    overall = metrics["overall"]
    if bundle is not None and stats_prefix is not None:
        _store_level1_stats(bundle, stats_prefix, metrics, roi_labels)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    axes[0, 0].plot(overall["freq"], overall["real_psd_mean"], label="real")
    axes[0, 0].plot(overall["freq"], overall["synth_psd_mean"], label="synth")
    axes[0, 0].set_title(f"{task} PSD")
    axes[0, 0].set_xlim(0.0, 0.25)
    axes[0, 0].legend()

    x = np.arange(len(overall["real_var_mean"]))
    plot_line_with_band(axes[0, 1], x, overall["real_var_mean"], overall["real_var_std"], "real")
    plot_line_with_band(axes[0, 1], x, overall["synth_var_mean"], overall["synth_var_std"], "synth")
    axes[0, 1].set_title(f"{task} ROI variance")
    axes[0, 1].legend()

    lags = np.arange(1, 6)
    plot_line_with_band(axes[1, 0], lags, overall["real_ac_mean"], overall["real_ac_std"], "real")
    plot_line_with_band(axes[1, 0], lags, overall["synth_ac_mean"], overall["synth_ac_std"], "synth")
    axes[1, 0].set_title(f"{task} autocorrelation")
    axes[1, 0].legend()

    top_idx = np.argsort(overall["ks_per_roi"])[-20:]
    axes[1, 1].barh([roi_labels[i] for i in top_idx], overall["ks_per_roi"][top_idx])
    axes[1, 1].set_title(f"{task} pooled top KS parcels")
    fig.savefig(task_dir / "level1_overview.png")
    plt.close(fig)


def _store_ranked_stats(bundle, stats_prefix, ranked, final_n):
    add_stats_value(bundle, stats_prefix, "final_n", value=int(final_n))
    for metric_name in ("r", "r2"):
        vals = ranked[final_n][metric_name]
        order = np.argsort(vals)
        add_stats_value(bundle, stats_prefix, metric_name, value=vals.astype(np.float32))
        add_stats_value(bundle, stats_prefix, f"bottom25_{metric_name}_idx", value=order[:25].astype(np.int64))
        add_stats_value(bundle, stats_prefix, f"top25_{metric_name}_idx", value=order[-25:][::-1].astype(np.int64))


def _store_averaged_stats(bundle, stats_prefix, averaged):
    overall = averaged["overall"]
    add_stats_dataframe(bundle, stats_prefix, "overall", "curves", frame=overall["curves"])
    add_stats_dataframe(bundle, stats_prefix, "overall", "by_t", frame=overall["by_t"])
    final_n = int(overall["curves"]["n"].iloc[-1])
    _store_ranked_stats(bundle, make_stats_prefix(stats_prefix, "overall", "ranked"), overall["ranked"], final_n)

    condition_names = []
    for name, curr in averaged["by_condition"].items():
        condition_key = sanitize_name(name)
        condition_names.append(str(name))
        add_stats_value(bundle, stats_prefix, "by_condition", condition_key, "name", value=str(name))
        add_stats_dataframe(bundle, stats_prefix, "by_condition", condition_key, "curves", frame=curr["curves"])
        add_stats_dataframe(bundle, stats_prefix, "by_condition", condition_key, "by_t", frame=curr["by_t"])
        final_n = int(curr["curves"]["n"].iloc[-1])
        _store_ranked_stats(bundle, make_stats_prefix(stats_prefix, "by_condition", condition_key, "ranked"), curr["ranked"], final_n)
    add_stats_value(bundle, stats_prefix, "condition_names", value=np.asarray(condition_names, dtype=str))


def make_stats_prefix(*parts):
    return "__".join(str(part) for part in parts if part not in (None, ""))


def save_averaged(task_dir, averaged, roi_labels, bundle=None, stats_prefix=None):
    task_dir.mkdir(parents=True, exist_ok=True)
    overall = averaged["overall"]
    if bundle is not None and stats_prefix is not None:
        _store_averaged_stats(bundle, stats_prefix, averaged)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.reshape(-1)
    axes[0].plot(overall["curves"]["n"], overall["curves"]["mse"], marker="o")
    axes[0].set_title("MSE vs N")
    axes[1].plot(overall["curves"]["n"], overall["curves"]["mean_r"], marker="o")
    axes[1].set_title("Mean r vs N")
    plot_line_with_band(axes[2], overall["by_t"]["t"], overall["by_t"]["mse_mean"], overall["by_t"]["mse_std"], f"N={int(overall['curves']['n'].iloc[-1])}")
    axes[2].set_title("MSE vs t")
    axes[2].legend()
    plot_line_with_band(axes[3], overall["by_t"]["t"], overall["by_t"]["mean_r"], overall["by_t"]["std_r"], f"N={int(overall['curves']['n'].iloc[-1])}")
    axes[3].set_title("Mean r vs t")
    axes[3].legend()
    fig.savefig(task_dir / "curves.png")
    plt.close(fig)


def save_subject_id(task_dir, results, context_frames, bundle=None, stats_prefix=None):
    task_dir.mkdir(parents=True, exist_ok=True)
    if bundle is not None and stats_prefix is not None:
        add_stats_value(bundle, stats_prefix, "positions", value=results["positions"])
        add_stats_value(bundle, stats_prefix, "real_mean", value=results["real_mean"])
        add_stats_value(bundle, stats_prefix, "real_std", value=results["real_std"])
        add_stats_value(bundle, stats_prefix, "synth_names", value=np.asarray(list(results["synth"].keys()), dtype=str))
        for name, synth in results["synth"].items():
            synth_key = sanitize_name(name)
            add_stats_value(bundle, stats_prefix, "synth", synth_key, "name", value=str(name))
            add_stats_value(bundle, stats_prefix, "synth", synth_key, "mean", value=synth["mean"])
            add_stats_value(bundle, stats_prefix, "synth", synth_key, "std", value=synth["std"])

    keep = results["positions"] >= int(context_frames)
    x = results["positions"][keep] - int(context_frames)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    plot_line_with_band(ax, x, results["real_mean"][keep], results["real_std"][keep], "real")
    for name, synth in results["synth"].items():
        plot_line_with_band(ax, x, synth["mean"][keep], synth["std"][keep], name)
    ax.set_title("Subject ID accuracy")
    ax.set_ylim(0.0, 0.5)
    ax.legend()
    fig.savefig(task_dir / "subject_id_accuracy.png")
    plt.close(fig)


def write_report(out_dir, summary_rows):
    lines = [
        "<html><head><meta charset='utf-8'><title>HCP Eval</title></head><body>",
        "<h1>HCP Eval</h1>",
    ]
    for section in summary_rows:
        lines.append(f"<h2>{html.escape(section['title'])}</h2>")
        if section.get("table") is not None:
            lines.append(section["table"].to_html(index=False))
        for image in section.get("images", []):
            if isinstance(image, dict):
                rel = image["path"].relative_to(out_dir)
                lines.append(f"<h3>{html.escape(image['caption'])}</h3>")
                lines.append(f"<div><img src='{html.escape(str(rel))}' style='max-width:1200px'></div>")
                continue
            rel = image.relative_to(out_dir)
            lines.append(f"<div><img src='{html.escape(str(rel))}' style='max-width:1200px'></div>")
    lines.append("</body></html>")
    with open(out_dir / "index.html", "w") as f:
        f.write("\n".join(lines))

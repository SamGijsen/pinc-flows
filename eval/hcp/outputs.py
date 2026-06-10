import numpy as np
import pandas as pd

from .common import sanitize_name, write_json
from .report import write_report
from .stats import add_stats_value


def level1_summary(level1):
    overall = level1["overall"]
    return {
        "mean_ks": overall["mean_ks"],
        "mmd": overall["mmd"],
        "paired_fc_corr_mean": overall["paired_fc_corr_mean"],
        "paired_fc_corr_std": overall["paired_fc_corr_std"],
    }


def glm_report_images(base_dir, contrast_name, caption_suffix, prefix="GLM"):
    return [
        {
            "path": base_dir / contrast_name / "beta_profile.png",
            "caption": f"{prefix} beta profile, {caption_suffix}",
        },
        {
            "path": base_dir / contrast_name / "overview.png",
            "caption": f"{prefix} overview, {caption_suffix}",
        },
    ]


def hcp_task_summary_display(task_summary):
    out = dict(task_summary)
    for key in (
        "glm_guidance_grid",
        "glm_piece_drop",
        "glm_phrase_variants",
        "ar_fixation",
        "relevance_input_compare",
        "glm_phrase_variants_relevance",
    ):
        out.pop(key, None)
    return out


def add_hcp_task_summary_stats(
    stats_bundle,
    task_prefix,
    task_summary_display,
    default_relevance_mode,
    relevance_grid,
    relevance_compare,
):
    for summary_key, summary_value in task_summary_display.items():
        add_stats_value(stats_bundle, task_prefix, "summary", summary_key, value=summary_value)
    add_stats_value(
        stats_bundle,
        task_prefix,
        "relevance_compare",
        "modes",
        value=np.asarray([default_relevance_mode] + relevance_grid, dtype=str),
    )
    for mode, mode_summary in relevance_compare.items():
        mode_key = sanitize_name(mode)
        add_stats_value(stats_bundle, task_prefix, "relevance_compare", mode_key, "mode", value=str(mode))
        for metric_name, metric_value in mode_summary.items():
            add_stats_value(stats_bundle, task_prefix, "relevance_compare", mode_key, metric_name, value=metric_value)


def append_hcp_report_rows(
    summary_rows,
    task,
    task_dir,
    primary_contrast,
    task_summary_display,
    guidance_scale,
    guidance_scale_grid,
    context_guidance_scale_grid,
    glm_guidance_grid_records,
    glm_guidance_grid_images,
    relevance_grid,
    default_relevance_mode,
    relevance_compare,
    relevance_images,
    glm_piece_drop_records,
    glm_piece_drop_images,
):
    glm_images = glm_report_images(task_dir / "glm", primary_contrast, f"guidance={guidance_scale}")
    for curr_guidance_scale in guidance_scale_grid:
        glm_images.extend(glm_report_images(
            task_dir / "glm_guidance" / f"scale_{str(curr_guidance_scale).replace('.', 'p')}",
            primary_contrast,
            f"guidance={curr_guidance_scale}",
        ))
    for curr_context_guidance_scale in context_guidance_scale_grid:
        glm_images.extend(glm_report_images(
            task_dir / "glm_context_guidance" / f"scale_{str(curr_context_guidance_scale).replace('.', 'p')}",
            primary_contrast,
            f"context guidance={curr_context_guidance_scale}",
        ))

    images = [
        {"path": task_dir / "level1" / "level1_overview.png", "caption": "Timeseries fidelity: Level 1"},
        {"path": task_dir / "averaged" / "curves.png", "caption": "Timeseries fidelity: averaged rollouts"},
    ] + glm_images
    if (task_dir / "subject_id" / "subject_id_accuracy.png").exists():
        images.append({"path": task_dir / "subject_id" / "subject_id_accuracy.png", "caption": "Timeseries fidelity: subject ID"})

    summary_rows.append({
        "title": task,
        "table": pd.DataFrame([task_summary_display]),
        "images": images,
    })
    if glm_guidance_grid_records:
        summary_rows.append({
            "title": f"{task} GLM Guidance Grid",
            "table": pd.DataFrame(glm_guidance_grid_records),
            "images": glm_guidance_grid_images,
        })
    if relevance_grid:
        relevance_rows = [{"mode": mode, **relevance_compare[mode]} for mode in [default_relevance_mode] + relevance_grid]
        summary_rows.append({
            "title": f"{task} Relevance Input Compare",
            "table": pd.DataFrame(relevance_rows),
            "images": relevance_images,
        })
    if glm_piece_drop_records:
        summary_rows.append({
            "title": f"{task} GLM Piece Drop",
            "table": pd.DataFrame(glm_piece_drop_records),
            "images": glm_piece_drop_images,
        })


def append_ibc_report_rows(
    summary_rows,
    task,
    task_dir,
    contrast_name,
    glm_summary,
    guidance_scale,
    glm_piece_drop_records,
    glm_piece_drop_images,
    glm_guidance_grid_records,
    glm_guidance_grid_images,
):
    summary_rows.append({
        "title": f"IBC {task}",
        "table": glm_summary,
        "images": glm_report_images(
            task_dir / "glm",
            contrast_name,
            f"guidance={guidance_scale}",
            prefix="IBC GLM",
        ),
    })
    if glm_piece_drop_records:
        summary_rows.append({
            "title": f"IBC {task} GLM Piece Drop",
            "table": pd.DataFrame(glm_piece_drop_records),
            "images": glm_piece_drop_images,
        })
    if glm_guidance_grid_records:
        summary_rows.append({
            "title": f"IBC {task} GLM Guidance Grid",
            "table": pd.DataFrame(glm_guidance_grid_records),
            "images": glm_guidance_grid_images,
        })


def append_ibc_relevance_report_row(summary_rows, task, relevance_rows, relevance_images):
    summary_rows.append({
        "title": f"IBC {task} Relevance Input Compare",
        "table": pd.DataFrame(relevance_rows),
        "images": relevance_images,
    })


def write_fc_only_outputs(out_dir, fc_cfg, tasks):
    write_json(out_dir / "summary.json", {
        "tasks": {},
        "ibc": {},
        "fc_eval": {
            "tasks": [str(task).upper() for task in fc_cfg.get("tasks", tasks)],
            "phrase_variant_indices": [int(x) for x in fc_cfg.get("phrase_variant_indices", [0, 1, 2, 3, 4, 5])],
        },
    })
    np.savez_compressed(out_dir / "stats.npz")
    write_report(out_dir, [])


def write_eval_outputs(out_dir, summary_json, stats_bundle, summary_rows):
    write_json(out_dir / "summary.json", summary_json)
    np.savez_compressed(out_dir / "stats.npz", **stats_bundle)
    write_report(out_dir, summary_rows)

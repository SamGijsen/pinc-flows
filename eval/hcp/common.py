import json


def base_subject_id(run_id):
    return str(run_id).rsplit("_", 1)[0]


def pick(data_cfg, split, key, default=None):
    split_key = f"{split}_{key}"
    if split_key in data_cfg:
        return data_cfg[split_key]
    return data_cfg.get(key, default)


def sanitize_name(name):
    return str(name).lower().replace("/", "_").replace(" ", "_")


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def plot_line_with_band(ax, x, mean, std, label):
    ax.plot(x, mean, label=label)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2)

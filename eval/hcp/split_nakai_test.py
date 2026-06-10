"""Split nakai_test_v2.h5 into test12 (reps 1-2) and test34 (reps 3-4)."""

import h5py

SRC = "/TODO_SET_PATH/nakai_test_v2.h5"
OUT12 = "/TODO_SET_PATH/nakai_test12_v2.h5"
OUT34 = "/TODO_SET_PATH/nakai_test34_v2.h5"


def get_split_indices(subject_ids):
    """Return indices for rep-01/02 and rep-03/04."""
    ids = [s.decode() if isinstance(s, bytes) else s for s in subject_ids]
    idx12 = [i for i, s in enumerate(ids) if s.endswith("rep-01") or s.endswith("rep-02")]
    idx34 = [i for i, s in enumerate(ids) if s.endswith("rep-03") or s.endswith("rep-04")]
    return idx12, idx34


def copy_dataset(src_ds, dst_grp, name, indices=None):
    """Copy a dataset, optionally selecting along the first axis."""
    if src_ds.shape == ():
        # Scalar dataset
        dst_grp.create_dataset(name, data=src_ds[()])
        return

    data = src_ds[:]
    if indices is not None:
        data = data[indices]

    if src_ds.dtype.kind == 'O':
        # Variable-length / object dtype
        dst_grp.create_dataset(name, data=data, dtype=h5py.special_dtype(vlen=str))
    else:
        dst_grp.create_dataset(name, data=data)


def copy_group(src_grp, dst_grp, indices12, indices34, dst_is_12):
    indices = indices12 if dst_is_12 else indices34
    for key in src_grp.keys():
        item = src_grp[key]
        if isinstance(item, h5py.Group):
            sub = dst_grp.require_group(key)
            copy_group(item, sub, indices12, indices34, dst_is_12)
        else:
            # Split only datasets whose first dimension matches n_sequences (24)
            if item.shape and item.shape[0] == 24:
                copy_dataset(item, dst_grp, key, indices=indices)
            else:
                copy_dataset(item, dst_grp, key, indices=None)


def main():
    with h5py.File(SRC, "r") as src:
        subject_ids = src["long_subject_id"][:]
        idx12, idx34 = get_split_indices(subject_ids)
        print(f"test12 indices ({len(idx12)}): {idx12}")
        print(f"test34 indices ({len(idx34)}): {idx34}")

        for out_path, is_12 in [(OUT12, True), (OUT34, False)]:
            label = "test12" if is_12 else "test34"
            print(f"\nWriting {label} -> {out_path}")
            with h5py.File(out_path, "w") as dst:
                copy_group(src, dst, idx12, idx34, is_12)
            print(f"  Done.")

    # Verify
    print("\nVerification:")
    for path, label in [(OUT12, "test12"), (OUT34, "test34")]:
        with h5py.File(path, "r") as f:
            ids = f["long_subject_id"][:]
            ts_shape = f["timeseries/schaefer400"].shape
            print(f"  {label}: long_subject_id={[s.decode() for s in ids]}")
            print(f"  {label}: timeseries/schaefer400 shape={ts_shape}")


if __name__ == "__main__":
    main()

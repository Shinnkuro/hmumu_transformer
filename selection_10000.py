from pathlib import Path
import numpy as np
import pyarrow.dataset as ds
import pyarrow.parquet as pq


# =========================
# 配置区
# =========================
INPUT_FILES = {
    "ggH": "/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/ggh_powhegPS_merged.parquet",
    "VBF": "/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/dy_VBF_filter_merged.parquet",
    "DY_M50": "/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/dy_M-50_MiNNLO_merged.parquet",
    "DY_M100To200": "/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/dy_M-100To200_MiNNLO_merged.parquet",
}

OUTPUT_DIR = Path("/depot/cms/hu1027/hmm_ntuples/skimmed_for_dnn_AK8jets/merged/2018/filtered_hpeak_10k")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASS_LOW = 115.0
MASS_HIGH = 135.0
N_KEEP = 10000
SEED = 42


# =========================
# 主逻辑
# =========================
def filter_and_sample_parquet(input_path: str, output_path: Path, n_keep: int, seed: int) -> None:
    dataset = ds.dataset(input_path, format="parquet")

    # 在读取阶段直接做过滤，避免先把全部事件读进内存
    filtered_table = dataset.to_table(
        filter=(ds.field("dimuon_mass") >= MASS_LOW) & (ds.field("dimuon_mass") < MASS_HIGH)
    )

    n_filtered = filtered_table.num_rows
    print(f"[INFO] {input_path}")
    print(f"       events after h_peak filter: {n_filtered}")

    if n_filtered == 0:
        print("       no events pass the filter, writing empty parquet.")
        pq.write_table(filtered_table, output_path)
        return

    rng = np.random.default_rng(seed)

    if n_filtered > n_keep:
        indices = rng.choice(n_filtered, size=n_keep, replace=False)
        indices = np.sort(indices)  # 不是必须，但这样输出更稳定
        sampled_table = filtered_table.take(indices)
        print(f"       randomly kept: {n_keep}")
    else:
        sampled_table = filtered_table
        print(f"       fewer than {n_keep} events, kept all: {n_filtered}")

    pq.write_table(sampled_table, output_path)
    print(f"       output written to: {output_path}\n")


def main():
    for sample_name, input_path in INPUT_FILES.items():
        input_file = Path(input_path)
        output_name = input_file.stem + "_hpeak_10k.parquet"
        output_path = OUTPUT_DIR / output_name

        filter_and_sample_parquet(
            input_path=input_path,
            output_path=output_path,
            n_keep=N_KEEP,
            seed=SEED,
        )


if __name__ == "__main__":
    main()
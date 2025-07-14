from pathlib import Path
import pandas as pd

import os
import math
from multiprocessing import Pool
from datasets.utils.py_utils import convert_file_size_to_int
from PIL import Image
from datasets import load_dataset

def save_shard(shard_info):
    """
    shard_info is a tuple of (index, shard, shard_path).
    We call to_parquet on the shard, writing it out to shard_path.
    """
    index, shard, shard_path = shard_info
    shard.to_parquet(shard_path)


def to_parquet(ds, output_dir, num_proc):
    os.makedirs(output_dir, exist_ok=True)

    dataset_nbytes = ds._estimate_nbytes()
    max_shard_size = convert_file_size_to_int("500MB")
    num_shards = max(math.ceil(dataset_nbytes / max_shard_size), 1)

    # Build all shard subsets in a list
    shard_infos = []
    for i in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=i, contiguous=True)
        shard_path = os.path.join(
            output_dir, f"{i:05d}-of-{(num_shards - 1):05d}.parquet"
        )
        shard_infos.append((i, shard, shard_path))

    # Start a worker pool and map the shards
    with Pool(processes=num_proc) as pool:
        pool.map(save_shard, shard_infos)


# Load the dataset from a local directory containing .parquet files
dataset = load_dataset(
    "/gpfs/ZuFS1/proj/deep-search/mao/datasets/FinTabNet_OTSL_v1.2_doclingdocuments/gt_dataset",
    split="test",
)
print(dataset)
for i, row in enumerate(dataset):
    if i == 1:
        break
    print(row)


# df = pd.read_csv(
#     "/gpfs/ZuFS1/proj/deep-search/mao/datasets/im2latex230k_ood_test_set/test.csv"
# )
# df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
# df = df.iloc[:10000]  # select first 10k rows
# print(df)
# images_path = Path(
#     "/gpfs/ZuFS1/proj/deep-search/mao/datasets/im2latex230k_ood_test_set/images"
# )

# rows = []
# for i, row in df.iterrows():
#     formula = row["formula"]

#     texts = [
#         {"assistant": f"<formula><loc_0><loc_0><loc_500><loc_500>{formula}</formula>",
#          "user": "Convert formula to LaTeX."
#         }
#     ]
#     filename = row["filename"]
#     images = [Image.open(images_path / filename).convert("RGB")]

#     rows.append(
#         {
#             "texts": texts,
#             "images": images,
#             "id": filename.replace(".png", "")
#         }
#     )

# from datasets import Dataset
# from datasets.features import Features, Value, Sequence, Image as HFImage

# # Convert rows to Hugging Face dataset
# ds = Dataset.from_list(rows)

# # Save as sharded parquet
# to_parquet(
#     ds,
#     "/gpfs/ZuFS1/proj/deep-search/mao/datasets/im2latex230k_ood_test_set/test",
#     num_proc=128,
# )

import time
import argparse
import os
from typing import Dict
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from PIL import Image
from io import BytesIO

from datasets import load_dataset, load_from_disk, DatasetDict


def load_hf_dataset(dataset_path, num_workers, split="test"):
    try:
        dataset = load_dataset(dataset_path, split=split, num_proc=num_workers)
    except:
        dataset = load_from_disk(dataset_path)
        if isinstance(dataset, DatasetDict) and split in dataset.keys():
            dataset = dataset[split]
        elif isinstance(dataset, DatasetDict) and "train" in dataset.keys():
            dataset = dataset["train"]
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Batch process images using LLM.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Dataset path (Hugging Face repo or disk path).",
    )
    parser.add_argument("--split", type=str, default="test", help="Split to load.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Convert this page to docling.",
        help="Prompt for the LLM.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=160, help="Batch size for inference."
    )
    parser.add_argument(
        "--log_interval", type=int, default=100, help="Logging interval."
    )
    parser.add_argument(
        "--num_workers", type=int, default=16, help="Number of dataset loading workers."
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = load_hf_dataset(args.dataset_path, args.num_workers, args.split)
    print(dataset)

    if "document_id" in dataset.features:
        ids = [id for id in dataset["document_id"]]
    elif "id" in dataset.features:
        ids = [id for id in dataset["id"]]
    else:
        print(
            "🚨 Could not find id column in dataset. Falling back to simple enumeration."
        )
        ids = list(range(len(dataset)))

    if "GroundTruthPageImages" in dataset.features:
        images = [imgs[0] for imgs in dataset["GroundTruthPageImages"]]
    elif "images" in dataset.features:
        images = [imgs[0] for imgs in dataset["images"]]
    elif "image" in dataset.features:
        images = [img for img in dataset["image"]]
    else:
        raise Exception("Image(s) column not found in dataset")

    pil_images = []
    for image in images:
        if isinstance(image, Dict) and "bytes" in image:
            pil_images.append(Image.open(BytesIO(image["bytes"])))
        elif isinstance(image, Image.Image):
            pil_images.append(image)
        else:
            raise Exception(f"Unsupported image type: {type(image)}")

    assert len(pil_images) == len(ids)
    images = pil_images

    llm = LLM(model=args.model_path, limit_mm_per_prompt={"image": 1}, seed=42)

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        skip_special_tokens=False,
        stop_token_ids=[100338],  # change
    )

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": args.prompt}],
        },
    ]

    processor = AutoProcessor.from_pretrained(args.model_path)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    start_time = time.time()

    for i in range(0, len(images), args.batch_size):
        batch_images = images[i : i + args.batch_size]
        batch_ids = ids[i : i + args.batch_size]

        llm_inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": img}}
            for img in batch_images
        ]

        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

        for id_, output in zip(batch_ids, outputs):
            output_text = output.outputs[0].text
            output_path = os.path.join(args.output_dir, f"{id_}.dt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(output_text)

        del llm_inputs, outputs

        if (i + len(batch_images)) % args.log_interval == 0 or i + len(
            batch_images
        ) == len(images):
            print(f"Processed | {i + len(batch_images)} / {len(images)} images")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f} sec")


if __name__ == "__main__":
    main()


# bsub -q normal -n 1 -R "span[hosts=1]" -M 200G -gpu "num=1:mode=exclusive_process" -oo ~/.lsbatch/batch_inference_vllm.stdout -eo ~/.lsbatch/bath_inference_vllm.stderr python3 batch_inference_vllm.py

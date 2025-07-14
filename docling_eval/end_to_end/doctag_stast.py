from pathlib import Path

folder = Path(
    "/gpfs/ZuFS1/proj/deep-search/mao/repos/docling-eval/output_eval_no_rl/model_output/ocr"
)  # change this
target_string = "</doctag>"

total_files = 0
matching_files = 0

for file in folder.iterdir():
    if file.is_file():
        total_files += 1
        try:
            content = file.read_text(encoding="utf-8", errors="ignore")
            if target_string in content:
                matching_files += 1
        except Exception as e:
            print(f"Error reading {file}: {e}")

print("📊 Stats:")
print(f"Total files checked:     {total_files}")
print(f"Files containing tag:    {matching_files}")
print(
    f"Percentage:              {matching_files / total_files * 100:.2f}%"
    if total_files
    else "N/A"
)

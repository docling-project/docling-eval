#!/usr/bin/env bash
set -euo pipefail 

source .venv/bin/activate

timestamp=$(date +"%Y%m%d-%H%M%S")
uuid=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c1-8)

# ------------------------------
# Default values
# ------------------------------
model_path="" # if not passed only evaluations will be performed, if inference data already present
# don't pass any of these params to avoid evaluating for this task
layout_dataset_path=""
ocr_dataset_path=""
table_dataset_path=""
equation_dataset_path=""
code_dataset_path=""

output_dir=""
num_workers=64

# ------------------------------
# Parse args
# ------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) model_path="$2"; shift 2 ;;
        --layout_dataset_path) layout_dataset_path="$2"; shift 2 ;;
        --ocr_dataset_path) ocr_dataset_path="$2"; shift 2 ;;
        --table_dataset_path) table_dataset_path="$2"; shift 2 ;;
        --equation_dataset_path) equation_dataset_path="$2"; shift 2 ;;
        --code_dataset_path) code_dataset_path="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --num_workers) num_workers="$2"; shift 2 ;;
        --help |-h)
            echo "Usage: $0 [--model_path path] [--layout_dataset_path path] ... [--output_dir dir] [--num_workers N]"
            exit 0
            ;;
        *) echo "❌ Unknown option $1"; exit 1 ;;
    esac
done

printf '%s\n' "-----------------------------------------------"
printf '| %-51s \n' "✅ Parsed args:"
printf '|   👉 %-25s = %s \n' "model_path" "$model_path"
printf '|   👉 %-25s = %s \n' "layout_dataset_path" "$layout_dataset_path"
printf '|   👉 %-25s = %s \n' "ocr_dataset_path" "$ocr_dataset_path"
printf '|   👉 %-25s = %s \n' "table_dataset_path" "$table_dataset_path"
printf '|   👉 %-25s = %s \n' "equation_dataset_path" "$equation_dataset_path"
printf '|   👉 %-25s = %s \n' "code_dataset_path" "$code_dataset_path"
printf '|   👉 %-25s = %s \n' "output_dir" "$output_dir"
printf '|   👉 %-25s = %s \n' "num_workers" "$num_workers"
printf '%s\n' "-----------------------------------------------"

if [ -z "$output_dir" ]; then
    if [ -n "$model_path" ]; then
        # Extract basename from model path, stripping trailing slash
        model_name=$(basename "${model_path%/}")
        output_dir="./outputs_evals/output_eval_${model_name}_${timestamp}_${uuid}"
    else
        output_dir="./outputs_evals/output_eval_${timestamp}_${uuid}"
    fi
fi


skip_layout=false
if [ -z "$layout_dataset_path" ]; then
    echo "🚨 layout_dataset_path not passed, skipping the evaluation for this task."
    skip_layout=true
fi

skip_ocr=false
if [ -z "$ocr_dataset_path" ]; then
    echo "🚨 ocr_dataset_path not passed, skipping the evaluation for this task."
    skip_ocr=true
fi

skip_table=false
if [ -z "$table_dataset_path" ]; then
    echo "🚨 table_dataset_path not passed, skipping the evaluation for this task."
    skip_table=true
fi

skip_equation=false
if [ -z "$equation_dataset_path" ]; then
    echo "🚨 equation_dataset_path not passed, skipping the evaluation for this task."
    skip_equation=true
fi

skip_code=false
if [ -z "$code_dataset_path" ]; then
    echo "🚨 code_dataset_path not passed, skipping the evaluation for this task."
    skip_code=true
fi

mkdir -p "$output_dir"

skip_inference=false
if [ -z "$model_path" ]; then
    skip_inference=true
    required=(layout ocr table equation code)
    echo "🚨 Model path has not been passed. This means that model output should already be present. Skipping Inference."
    for name in "${required[@]}"; do
        var="skip_${name}"
        # if we’re skipping this task, we must have the folder under inferred_path
        if [ "${!var}" = "false" ]; then
            if [ ! -d "${output_dir}/model_output/${name}" ]; then
                echo "🚨 Missing required folder '${name}' under '${output_dir}/model_output' for skipped task."
                exit 1
            fi
        fi
    done
    echo "✅ All required inferred sub-folders are present."
fi

printf '%s\n' "-----------------------------------------------"
printf '| %-18s %s\n' "⏭️ Skip Inference?" "$skip_inference"
printf '| %-16s %s\n' "⏭️ Skip Layout?" "$skip_layout"
printf '| %-16s %s\n' "⏭️ Skip OCR?" "$skip_ocr"
printf '| %-16s %s\n' "⏭️ Skip Table?" "$skip_table"
printf '| %-16s %s\n' "⏭️ Skip Equation?" "$skip_equation"
printf '| %-16s %s\n' "⏭️ Skip Code?" "$skip_code"
printf '%s\n' "-----------------------------------------------"



# ---------------------------------------------------------
# Inference (run only if we’re *not* skipping inference)
# ---------------------------------------------------------
if [ "$skip_inference" = false ]; then
    declare -A data_paths=(
        [layout]="$layout_dataset_path"
        [ocr]="$ocr_dataset_path"
        [table]="$table_dataset_path"
        [equation]="$equation_dataset_path"
        [code]="$code_dataset_path"
    )

    declare -A prompts=(
        [layout]="Convert this page to docling."
        [ocr]="Convert this page to docling."
        [table]="Convert table to OTSL."
        [equation]="Convert formula to LaTeX."
        [code]="Convert code to text."
    )

    declare -A splits=(
        [layout]="test"
        [ocr]="test"
        [table]="test"
        [equation]="test"
        [code]="test"
    )

    declare -A dataset_to_task

    for task in layout ocr table equation code; do
        skip_var="skip_${task}"
        if [ "${!skip_var}" = false ]; then
            dataset="${data_paths[$task]}"
            if [[ -v 'dataset_to_task["'"$dataset"'"]' ]]; then
                # This dataset has already been assigned to another task
                original_task="${dataset_to_task[$dataset]}"
                echo "🔁 Detected identical dataset for $task and $original_task."
                echo "🔗 Creating symlink to reuse output from $original_task for $task."
                mkdir -p "${output_dir}/model_output"
                ln -sfn "${original_task}" "${output_dir}/model_output/${task}"
            else
                dataset_to_task[$dataset]="$task"
            fi
        fi
    done

    for task in layout ocr table equation code; do
        skip_var="skip_${task}"
        if [ "${!skip_var}" = false ]; then
            if [ ! -L "${output_dir}/model_output/${task}" ]; then
                echo "🚀 Performing inference for **${task^^}** on dataset ${data_paths[$task]}"
                mkdir -p "${output_dir}/model_output/${task}"
                uv run python3 docling_eval/end_to_end/batch_inference_vllm.py \
                    --model_path "$model_path" \
                    --dataset_path "${data_paths[$task]}" \
                    --split "${splits[$task]}" \
                    --output_dir "${output_dir}/model_output/${task}" \
                    --num_workers "$num_workers" \
                    --prompt "${prompts[$task]}"
            else
                echo "⏭️ Skipping inference for ${task} (already linked to another task)."
            fi
        fi
    done
fi

# normalize equations
# if [ "$skip_equation" = false ]; then
#     echo "🧮 Normalizing equations..."
#     mkdir -p "${output_dir}/model_output/equation_normalized"
#     uv run python3 docling_eval/end_to_end/normalize_equations.py  \
#         --input_dir "${output_dir}/model_output/equation" \
#         --output_dir "${output_dir}/model_output/equation_normalized" \
#         --num_workers "$num_workers"
#     printf '%s\n' "-----------------------------------------------"
# fi

# ---------------------------------------------------------
# Evaluation & visualisation with `docling_eval`
# ---------------------------------------------------------
declare -A modality=(
    [layout]="layout"
    [ocr]="markdown_text"
    [table]="table_structure"
)

declare -A dataset_paths=(
    [layout]="$layout_dataset_path"
    [ocr]="$ocr_dataset_path"
    [table]="$table_dataset_path"
    [equation]="$equation_dataset_path"
    [code]="$code_dataset_path"
)

declare -A output_dirs=(
    [layout]="${output_dir}/evals/layout/"
    [ocr]="${output_dir}/evals/ocr/"
    [table]="${output_dir}/evals/table/"
    [equation]="${output_dir}/evals/equation/"
    [code]="${output_dir}/evals/code/"
)

declare -A splits=(
    [layout]="test"
    [ocr]="test"
    [table]="test"
    [equation]="test"
    [code]="test"
)

declare -A benchmarks=(
    [layout]="DocLayNetV2"
    [ocr]="DocLayNetV2"
    [table]="FinTabNet"
)

declare -A file_source_paths=(
    [layout]="${output_dir}/model_output/layout"
    [ocr]="${output_dir}/model_output/ocr"
    [table]="${output_dir}/model_output/table"
    [equation]="${output_dir}/model_output/equation"
    [code]="${output_dir}/model_output/code"

)
# evaluate tasks for which you can use docling_eval
for task in layout ocr table; do
    skip_var="skip_${task}"
    if [ "${!skip_var}" = false ]; then
        echo "📝 Evaluating **${task}** (modality: ${modality[$task]}) …"

        echo "  👉 Running create-eval …"
        uv run docling-eval \
            create-eval \
            --benchmark ${benchmarks[$task]} \
            --gt-dir  ${dataset_paths[$task]}\
            --prediction-provider File \
            --file-prediction-format doctags \
            --file-source-path ${file_source_paths[$task]} \
            --output-dir "${output_dirs[$task]}"

        echo "  👉 Running evaluate …"
        uv run docling-eval evaluate \
            --modality "${modality[$task]}" \
            --benchmark "${benchmarks[$task]}" \
            --output-dir "${output_dirs[$task]}" \
            --split "${splits[$task]}"

        echo "  👉 Running visualize …"
        uv run docling-eval visualize \
            --modality "${modality[$task]}" \
            --benchmark "${benchmarks[$task]}" \
            --output-dir "${output_dirs[$task]}" \
            --split "${splits[$task]}"
        printf '%s\n' "-----------------------------------------------"
    else
        echo "⏭️  Skipping evaluation for ${task} (skip_${task}=true)"
        printf '%s\n' "-----------------------------------------------"
    fi
done


# evaluate tasks for which you CAN'T use docling_eval
for task in code equation; do
    skip_var="skip_${task}"
    if [ "${!skip_var}" = false ]; then
        echo "📝 Evaluating **${task}** …"

        uv run python3 docling_eval/end_to_end/evaluate_code_equations.py \
            --file_source_path ${file_source_paths[$task]} \
            --gt_dir  ${dataset_paths[$task]} \
            --output_dir "${output_dirs[$task]}" \
            --num_workers "$num_workers" \
            --split "${splits[$task]}"
    else
        echo "⏭️  Skipping evaluation for ${task} (skip_${task}=true)"
    fi
done

# bsub -q normal -n 1 -R "span[hosts=1]" -M 200G -gpu "num=1:mode=exclusive_process" -oo ~/.lsbatch/evaluate_2.stdout -eo ~/.lsbatch/evaluate_2.stderr docling_eval/end_to_end/evaluate_smoldocling_end_to_end.sh --model_path /gpfs/ZuFS1/proj/deep-search/mao/repos/docling-eval/checkpoints/granitedocling_v06.5r_stg_4_comon_debug_1_checkpoint-4808 --layout_dataset_path /gpfs/ZuFS1/proj/deep-search/datasets/doclaynet-v2-docling-GT/gt_dataset --ocr_dataset_path /gpfs/ZuFS1/proj/deep-search/datasets/doclaynet-v2-docling-GT/gt_dataset --code_dataset_path /gpfs/ZuFS1/proj/deep-search/mao/datasets/synth_code_net_test_set_5k --equation_dataset_path /gpfs/ZuFS1/proj/deep-search/mao/datasets/im2latex230k_ood_test_set/test_hf_format --table_dataset_path /gpfs/ZuFS1/proj/deep-search/mao/datasets/FinTabNet_OTSL_v1.2_doclingdocuments/gt_dataset

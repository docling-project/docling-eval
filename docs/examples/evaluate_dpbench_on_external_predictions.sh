#!/bin/bash

###########################################################################################
# Invariants
#

readonly GT_DIR=scratch/DPBench/gt_dataset

readonly MODALITIES=(
layout
table_structure
document_structure
reading_order
markdown_text
bboxes_text
key_value
timings
)


###########################################################################################
# Functions
#

evaluate() {
    local pred_dir save_dir modality
    pred_dir="$1"
    save_dir="$2"

    # Check if the GT/preds dirs exist
    if [ ! -d "${GT_DIR}" ]; then
        echo "Missing GT dir: ${GT_DIR}"
        exit 1
    fi
    if [ ! -d "${pred_dir}" ]; then
        echo "Missing predictions dir: ${pred_dir}"
        exit 2
    fi

    for modality in "${MODALITIES[@]}"; do
        echo "Evaluate: modality: ${modality}: predictions: ${pred_dir}"
        uv run docling-eval evaluate \
            --benchmark DPBench \
            --modality "${modality}" \
            --input-dir "${GT_DIR}" \
            --external-predictions-path "${pred_dir}" \
            --output-dir "${save_dir}"
    done
}


visualize_predictions() {
    local pred_dir save_dir modality
    pred_dir="$1"
    save_dir="$2"

    # Check if the GT/preds dirs exist
    if [ ! -d "${GT_DIR}" ]; then
        echo "Missing GT dir: ${GT_DIR}"
        exit 1
    fi
    if [ ! -d "${pred_dir}" ]; then
        echo "Missing predictions dir: ${pred_dir}"
        exit 2
    fi

    echo "Visualize predictions: ${pred_dir}"
    uv run docling-eval create_viz \
        --dataset-dir "${GT_DIR}" \
        --external-predictions-path "${pred_dir}" \
        --output-dir "${save_dir}"
}


visualize_evaluations() {
    local pred_dir eval_root modality
    pred_dir="$1"
    eval_root="$2"

    for modality in "${MODALITIES[@]}"; do
        echo "Evaluate: modality: ${modality} for evaluations: ${eval_root}"
        uv run docling-eval visualize \
            --benchmark DPBench \
            --modality "${modality}" \
            --input-dir "${pred_dir}" \
            --output-dir "${eval_root}"
    done
}

###########################################################################################
# Main
#

#########################################
# Predictions

# json predictions
evaluate \
    scratch/DPBench/predicted_documents/json \
    scratch/DPBench/external_predictions_jsons


# doctags predictions
evaluate \
    scratch/DPBench/predicted_documents/doctag \
    scratch/DPBench/external_predictions_doctags


# yaml predictions
evaluate \
    scratch/DPBench/predicted_documents/yaml \
    scratch/DPBench/external_predictions_yaml


#########################################
# Visualisations
visualize_predictions \
    scratch/DPBench/predicted_documents/json \
    scratch/DPBench/external_predictions_visualisations

visualize_evaluations \
    scratch/DPBench/predicted_documents/doctag \
    scratch/DPBench/external_predictions_doctags


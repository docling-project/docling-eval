# Review Bundle Workflow

Follow these steps whenever you need to review a submission’s visualization outputs.

## 1. Generate the Bundle

You need to start from a submission evaluation directory, e.g. `submission-4_10-11-2025_v0`, as created through the `cvat_evaluation_pipeline.py`.
It is required to contain a `combined_evaluation.csv` (export it from `combined_evaluation.xlsx` if missing).

Run the bundle builder from the submission root. This reads `combined_evaluation.csv` and writes a timestamped bundle folder next to it.

```bash
uv run python -m docling_eval.campaign_tools.review_bundle_builder \
  /path/to/submission-root
```

The command prints the bundle path, e.g. `submission-4_10-11-2025_v0/review_bundle_20251110_162408`.

## 2. Host the Submission Root

Serve the submission directory so the bundle and original visualization HTMLs share the same origin:

```bash
cd /path/to/submission-root
python -m http.server 8765
```

Open `http://localhost:8765/review_bundle_*/index.html` in your browser.

## 3. Connect to `review_state.json` database

At the top of the page click **Connect bundle for saving**. Choose the same `review_bundle_*` folder. **This is critical** to ensure the browser has permissions to save into the `review_state.json`, which acts as a database. Once connected, the button switches to “Change bundle connection” and the status line on the bottom states that saves go directly to `review_state.json`. Decisions are still mirrored in browser storage as a safety net.

You can import prior logs via **Import log** (JSON or CSV). Imports do not change the save target — always reconnect if you move to another bundle.

## 4. Review and Save Decisions

The left sidebar lists documents sorted by the configured column. Selecting an entry loads its visualization(s). You will find the tab buttons for key-value and layout if both are available. The decision panel shows separate controls for **User A** and **User B**:

1. Click **Correct** or **Need changes** for each user as needed (buttons stay lit when active).
2. Optionally add a shared comment.
3. Press **Save decision**. The status line confirms whether you saved user verdicts, comments only, or cleared the entry.

Edits remain locked to the current sample until you save or discard them; this prevents accidental navigation losses. Use **Export CSV** at any time for an external report.


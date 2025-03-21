### OCR PixParse Evaluation

Prerequisites for evaluating OCR on the PixParse dataset.

## Setup Instructions

1. Clone the repository:
```bash
git clone https://huggingface.co/datasets/samiuc/pixparse-idl
```


2. Environment Configuration

Set the following environment variables for cloud service providers:

#### AWS
```bash
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
```

#### Google Cloud
```bash
export GOOGLE_PROJECT_ID="your_google_project_id"
export GOOGLE_PROCESSOR_ID="your_google_processor_id"
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

#### Azure
```bash
export AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT="your_azure_endpoint"
export AZURE_DOCUMENT_INTELLIGENCE_KEY="your_azure_key"
```

3. Running Evaluation

```
poetry run python create_pixparse.py 
```

**Additional Notes**

- Set `reprocess=False` to use cached results if available
- Other hyperscaler/OCR engines can be selected by changing the `ocr_engine` and `hyperscaler` parameter
- Results will be saved to the specified output directory
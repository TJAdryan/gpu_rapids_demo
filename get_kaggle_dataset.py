import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Initialize the API client (authenticates using your kaggle.json file)
api = KaggleApi()
api.authenticate()

# Download files to a specific path and unzip them
api.dataset_download_files(
    dataset='albedox/cms-open-payment-dataset-2025', 
    path='./data', 
    unzip=True
)
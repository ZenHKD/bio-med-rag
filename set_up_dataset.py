import json
import os
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT_DIR = Path(__file__).resolve().parent

data_dir = os.path.join(ROOT_DIR, "data")

os.makedirs(data_dir, exist_ok=True)

external_data_dir = os.path.join(data_dir, "external")
os.makedirs(external_data_dir, exist_ok=True)

vectorstore_data_dir = os.path.join(data_dir, "vectorstore")
os.makedirs(vectorstore_data_dir, exist_ok=True)

"""
snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="bc5cdr/**",
    local_dir=external_data_dir
)

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="ChemDisGene/**"
    local_dir=external_data_dir
)

"""

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="bioasq/**",
    local_dir=external_data_dir
)

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="medqa/**",
    local_dir=external_data_dir
)

snapshot_download(
    repo_id="zinzinmit/MedNLPCombined",
    repo_type="dataset",
    allow_patterns="pubmedqa/**",
    local_dir=external_data_dir
)

snapshot_download(
    repo_id="ngocnvh/medmcqa",
    repo_type="dataset",
    allow_patterns="medmcqa/**",
    local_dir=external_data_dir,
)
# Use the environment variable if the user doesn't provide Project ID.
import os

import vertexai

PROJECT_ID = ""  # @param {type: "string", placeholder: "[your-project-id]", isTemplate: true}

if not PROJECT_ID:
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")

REGION = ""  # @param {type: "string", placeholder: "[your-region]", isTemplate: true}

if not REGION:
    REGION = os.environ.get("GOOGLE_CLOUD_REGION", "us-central1")

vertexai.init(project=PROJECT_ID, location=REGION)

print(f"Project: {PROJECT_ID}\nLocation: {REGION}")
     

from vertexai import model_garden



all_model_versions = model_garden.list_deployable_models(
    model_filter="gemma3", list_hf_models=False
)
     

print(all_model_versions)
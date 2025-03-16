#!/bin/bash
set -e

# Colors for console output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Configuration variables
PROJECT_ID="koalitionskompass"
REGION="europe-west1"
SERVICE_NAME="koalitionskompass"
SECRETS_FILE=".streamlit/secrets.toml"

# Print banner
echo -e "${GREEN}=======================================================${NC}"
echo -e "${GREEN}     Koalitionskompass GCP Deployment Script          ${NC}"
echo -e "${GREEN}=======================================================${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}gcloud CLI is not installed. Please install it first.${NC}"
    echo "Visit: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if secrets file exists
if [ ! -f "$SECRETS_FILE" ]; then
    echo -e "${RED}Error: Secrets file not found at $SECRETS_FILE${NC}"
    exit 1
fi

# Function to extract a value from secrets.toml file
extract_secret() {
    local section=$1
    local key=$2
    local value=$(grep -A 20 "^\[$section\]" "$SECRETS_FILE" | grep "^$key =" | head -1 | cut -d '=' -f2- | tr -d ' "')
    echo "$value"
}

# Extract secrets from .streamlit/secrets.toml
echo -e "${YELLOW}Extracting secrets from $SECRETS_FILE...${NC}"
OPENAI_API_KEY=$(extract_secret "openai" "api_key")
PINECONE_API_KEY=$(extract_secret "pinecone" "api_key")
PINECONE_ENVIRONMENT=$(extract_secret "pinecone" "environment")
PINECONE_INDEX_NAME=$(extract_secret "pinecone" "index_name")
PINECONE_NAMESPACE=$(extract_secret "pinecone" "namespace")
APP_ENVIRONMENT=$(extract_secret "app" "environment")

# Validate required secrets
if [[ -z "$OPENAI_API_KEY" || -z "$PINECONE_API_KEY" || -z "$PINECONE_ENVIRONMENT" ]]; then
    echo -e "${RED}Error: Required secrets are missing.${NC}"
    echo "Please check your $SECRETS_FILE file."
    exit 1
fi

# Prompt for GCP Project ID if not set
if [[ -z "$PROJECT_ID" ]]; then
    # Get default project from gcloud config
    DEFAULT_PROJECT=$(gcloud config get-value project 2>/dev/null)
    
    read -p "Enter your GCP Project ID [$DEFAULT_PROJECT]: " INPUT_PROJECT_ID
    PROJECT_ID=${INPUT_PROJECT_ID:-$DEFAULT_PROJECT}
    
    if [[ -z "$PROJECT_ID" ]]; then
        echo -e "${RED}Error: No GCP Project ID provided.${NC}"
        exit 1
    fi
fi

# Prompt for region if desired
read -p "Enter GCP region [$REGION]: " INPUT_REGION
REGION=${INPUT_REGION:-$REGION}

# Set the project
echo -e "${YELLOW}Setting GCP project to: $PROJECT_ID${NC}"
gcloud config set project "$PROJECT_ID"

# Enable required services if not already enabled
echo -e "${YELLOW}Enabling required GCP services...${NC}"
gcloud services enable cloudbuild.googleapis.com run.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com --quiet

# Build the container image using Cloud Build
echo -e "${YELLOW}Building container using Cloud Build...${NC}"
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$SERVICE_NAME" .

# Deploy to Cloud Run with secrets as environment variables
echo -e "${YELLOW}Deploying to Cloud Run...${NC}"
gcloud run deploy "$SERVICE_NAME" \
    --image "gcr.io/$PROJECT_ID/$SERVICE_NAME" \
    --platform managed \
    --region "$REGION" \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY" \
    --set-env-vars "PINECONE_API_KEY=$PINECONE_API_KEY" \
    --set-env-vars "PINECONE_ENVIRONMENT=$PINECONE_ENVIRONMENT" \
    --set-env-vars "PINECONE_INDEX_NAME=$PINECONE_INDEX_NAME" \
    --set-env-vars "PINECONE_NAMESPACE=$PINECONE_NAMESPACE" \
    --set-env-vars "APP_ENVIRONMENT=$APP_ENVIRONMENT"

# Output the service URL
echo -e "${GREEN}Deployment complete!${NC}"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" --region "$REGION" --format='value(status.url)')
echo -e "${GREEN}Your service is available at: $SERVICE_URL${NC}"
echo -e "${YELLOW}Note: Environment variables have been set from your secrets.toml file.${NC}" 
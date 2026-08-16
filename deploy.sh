#!/usr/bin/env bash
# ===========================================================================
# MedRax — Google Cloud Deployment Script
# ===========================================================================
# This script automates deploying MedRax to Google Cloud.
#
# Prerequisites:
#   1. gcloud CLI installed and authenticated
#   2. Docker installed
#   3. A GCP project with billing enabled
#   4. Artifact Registry API enabled
#   5. A .env file with valid HF_TOKEN
#
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh                    # Deploy with defaults
#   ./deploy.sh --gpu              # Deploy with GPU support
#   ./deploy.sh --vm               # Deploy to a Compute Engine VM instead
#
# ===========================================================================

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────
# Override these via environment variables or edit directly

PROJECT_ID="${GCP_PROJECT_ID:?Error: GCP_PROJECT_ID is not set. Export it or add to .env}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-medrax}"
REPO_NAME="${REPO_NAME:-medrax-repo}"
IMAGE_NAME="${IMAGE_NAME:-medrax}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-nvidia-tesla-t4}"
GPU_COUNT="${GPU_COUNT:-1}"
ZONE="${ZONE:-us-central1-a}"

# Full image path in Artifact Registry
FULL_IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}"

# ── Parse arguments ───────────────────────────────────────────────────────
DEPLOY_MODE="cloudrun"   # default: Cloud Run
USE_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            USE_GPU=true
            shift
            ;;
        --vm)
            DEPLOY_MODE="vm"
            shift
            ;;
        --help)
            echo "Usage: ./deploy.sh [--gpu] [--vm]"
            echo ""
            echo "  --gpu    Enable GPU support (Cloud Run GPU or GCE with GPU)"
            echo "  --vm     Deploy to a Compute Engine VM instead of Cloud Run"
            echo ""
            echo "Environment variables:"
            echo "  GCP_PROJECT_ID   (required)  Google Cloud project ID"
            echo "  GCP_REGION       (optional)  Region (default: us-central1)"
            echo "  SERVICE_NAME     (optional)  Service name (default: medrax)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Helper functions ──────────────────────────────────────────────────────
info()    { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
success() { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error()   { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# ── Step 1: Verify prerequisites ─────────────────────────────────────────
info "Verifying prerequisites..."

command -v gcloud >/dev/null 2>&1 || error "gcloud CLI is not installed."
command -v docker >/dev/null 2>&1 || error "Docker is not installed."

gcloud config set project "${PROJECT_ID}" --quiet
success "GCP project: ${PROJECT_ID}"

# ── Step 2: Enable required APIs ─────────────────────────────────────────
info "Enabling required GCP APIs..."

gcloud services enable \
    artifactregistry.googleapis.com \
    run.googleapis.com \
    compute.googleapis.com \
    cloudbuild.googleapis.com \
    --quiet

success "APIs enabled."

# ── Step 3: Create Artifact Registry repository ──────────────────────────
info "Creating Artifact Registry repository (if not exists)..."

gcloud artifacts repositories create "${REPO_NAME}" \
    --repository-format=docker \
    --location="${REGION}" \
    --description="MedRax Docker images" \
    --quiet 2>/dev/null || true

# Configure Docker authentication for Artifact Registry
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
success "Artifact Registry configured."

# ── Step 4: Build and push Docker image ──────────────────────────────────
info "Building Docker image: ${FULL_IMAGE}"
docker build -t "${FULL_IMAGE}" .

info "Pushing image to Artifact Registry..."
docker push "${FULL_IMAGE}"
success "Image pushed: ${FULL_IMAGE}"

# ── Step 5: Deploy ───────────────────────────────────────────────────────
if [[ "${DEPLOY_MODE}" == "cloudrun" ]]; then
    info "Deploying to Cloud Run..."

    DEPLOY_CMD=(
        gcloud run deploy "${SERVICE_NAME}"
        --image "${FULL_IMAGE}"
        --region "${REGION}"
        --platform managed
        --port 7860
        --memory 16Gi
        --cpu 4
        --timeout 600
        --max-instances 3
        --min-instances 0
        --allow-unauthenticated
        --set-env-vars "DEVICE=auto,TORCH_DTYPE=bfloat16"
        --quiet
    )

    if [[ "${USE_GPU}" == true ]]; then
        DEPLOY_CMD+=(--gpu 1 --gpu-type "${GPU_TYPE}")
        info "GPU enabled: ${GPU_TYPE}"
    fi

    "${DEPLOY_CMD[@]}"

    # Get the service URL
    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region "${REGION}" \
        --format "value(status.url)")

    success "Deployed to Cloud Run!"
    info "Service URL: ${SERVICE_URL}"

elif [[ "${DEPLOY_MODE}" == "vm" ]]; then
    info "Deploying to Compute Engine VM..."

    VM_NAME="${SERVICE_NAME}-vm"

    # Create the VM with GPU
    CREATE_CMD=(
        gcloud compute instances create "${VM_NAME}"
        --zone "${ZONE}"
        --machine-type "${MACHINE_TYPE}"
        --image-family "pytorch-latest-gpu"
        --image-project "deeplearning-platform-release"
        --boot-disk-size 100GB
        --scopes "cloud-platform"
        --metadata "install-nvidia-driver=True"
        --quiet
    )

    if [[ "${USE_GPU}" == true ]]; then
        CREATE_CMD+=(--accelerator "type=${GPU_TYPE},count=${GPU_COUNT}")
        info "GPU enabled: ${GPU_TYPE} x${GPU_COUNT}"
    fi

    "${CREATE_CMD[@]}" 2>/dev/null || warn "VM may already exist."

    # Generate startup script
    info "Generating startup commands..."
    cat <<EOF

═══════════════════════════════════════════════════════════
  VM CREATED: ${VM_NAME}
═══════════════════════════════════════════════════════════

  SSH into the VM and run:

  gcloud compute ssh ${VM_NAME} --zone ${ZONE}

  # On the VM:
  sudo docker pull ${FULL_IMAGE}
  sudo docker run -d \\
      -p 7860:7860 \\
      --gpus all \\
      -e HF_TOKEN=<your-token> \\
      -e DEVICE=auto \\
      -e TORCH_DTYPE=bfloat16 \\
      ${FULL_IMAGE}

  # Allow firewall traffic on port 7860:
  gcloud compute firewall-rules create allow-medrax \\
      --allow tcp:7860 \\
      --target-tags medrax \\
      --quiet

═══════════════════════════════════════════════════════════
EOF

    success "VM deployment prepared."
fi

echo ""
success "MedRax deployment complete! 🏥"

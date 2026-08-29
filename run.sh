#!/bin/bash

# Ensure the script stops on any error
set -e

# Make sure the data directories exist before mounting them in Docker
mkdir -p data/indexes
mkdir -p data/papers

echo "🚀 Starting SciGuru via Docker Compose..."
docker compose up -d --build

echo "============================================================"
echo "✅ SciGuru is successfully deployed locally!"
echo "🌐 Access the application at: http://localhost:8501"
echo ""
echo "To share the app publicly using Ngrok during your interview,"
echo "open a NEW terminal and run:"
echo "    ngrok http 8501"
echo "============================================================"

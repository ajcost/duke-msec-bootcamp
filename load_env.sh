#!/bin/bash

# Check if .env file exists
if [ -f .env ]; then
  # Source the .env file to load environment variables
  set -a
  source .env
  set +a
  echo "Environment variables loaded from .env file."
else
  echo ".env file not found."
fi
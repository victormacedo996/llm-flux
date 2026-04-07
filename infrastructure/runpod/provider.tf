terraform {
  required_providers {
    runpod = {
      source = "decentralized-infrastructure/runpod"
    }
  }
}

provider "runpod" {
  api_key = var.runpod_api_key
}
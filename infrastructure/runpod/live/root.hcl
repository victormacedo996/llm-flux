# Root Terragrunt configuration.
# Shared across all child modules in live/.
#
# Usage:
#   cp infrastructure/runpod/live/secrets.hcl.example infrastructure/runpod/live/secrets.hcl
#   # Fill in your RunPod API key in secrets.hcl, then:
#   cd live/gpu_pod && terragrunt apply

locals {
  # Read the API key from the gitignored secrets.hcl file in this directory.
  # secrets.hcl must contain:
  #   locals { runpod_api_key = "rpa_..." }
  secrets        = read_terragrunt_config(find_in_parent_folders("runpod_api_key.secrets.hcl"))
  runpod_api_key = local.secrets.locals.runpod_api_key
}

# Inject provider + required_providers into every child module automatically.
generate "provider" {
  path      = "provider_override.tf"
  if_exists = "overwrite_terragrunt"
  contents  = <<-EOF
    terraform {
      required_version = ">= 1.5"
      required_providers {
        runpod = {
          source  = "decentralized-infrastructure/runpod"
          version = "~> 1.0"
        }
      }
    }

    provider "runpod" {
      api_key = "${local.runpod_api_key}"
    }
  EOF
}

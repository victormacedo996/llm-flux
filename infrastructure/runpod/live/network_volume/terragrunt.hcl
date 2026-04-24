# Network Volume deployment.
# Inherits provider config from the root terragrunt.hcl.
#
# Deploy:  cd live/network_volume && terragrunt apply
# Destroy: cd live/network_volume && terragrunt destroy
#
# Once deployed, pass the output `network_volume_id` to a gpu_pod's
# `network_volume_id` input to attach persistent storage.

include "root" {
  path = find_in_parent_folders("root.hcl")
}

terraform {
  source = "../../modules/runpod_network_volume"
}

inputs = {
  name           = "ag-test-network-volume"
  size           = 150              # GB — between 0 and 4000
  data_center_id = "EU-CZ-1"
}

include "root" {
  path = find_in_parent_folders("root.hcl")
}

terraform {
  # Path is relative to this file's directory.
  source = "../../modules/runpod_pod"
}

dependency "network_volume" {
  config_path = "../network_volume"
  mock_outputs = {
    network_volume_id = "mock-network-volume-id"
  }
}

inputs = {
  name  = "ag-test-gpu-pod"

  # ── Image ────────────────────────────────────────────────────────────────────
  image_name = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"

  # ── Hardware ─────────────────────────────────────────────────────────────────
  cloud_type      = "SECURE"   # "SECURE" for Secure Cloud
  gpu_type_ids    = ["NVIDIA GeForce RTX 4090"]
  gpu_count       = 1
  data_center_ids = ["EUR-NO-1"]

#   | value must be one of 'NVIDIA GeForce RTX 4090', 'NVIDIA A40', 'NVIDIA RTX A5000', 
#   | 'NVIDIA GeForce RTX
#   │ 5090', 'NVIDIA H100 80GB HBM3', 'NVIDIA GeForce RTX 3090', 'NVIDIA RTX
#   │ A4500', 'NVIDIA L40S', 'NVIDIA H200', 'NVIDIA L4', 'NVIDIA RTX 6000 Ada
#   │ Generation', 'NVIDIA A100-SXM4-80GB', 'NVIDIA RTX 4000 Ada Generation',
#   │ 'NVIDIA RTX A6000', 'NVIDIA A100 80GB PCIe', 'NVIDIA RTX 2000 Ada
#   │ Generation', 'NVIDIA RTX A4000', 'NVIDIA RTX PRO 6000 Blackwell Server
#   │ Edition', 'NVIDIA H100 PCIe', 'NVIDIA H100 NVL', 'NVIDIA L40', 'NVIDIA
#   │ B200', 'NVIDIA GeForce RTX 3080 Ti', 'NVIDIA RTX PRO 6000 Blackwell
#   │ Workstation Edition', 'NVIDIA GeForce RTX 3080', 'NVIDIA GeForce RTX 3070',
#   │ 'AMD Instinct MI300X OAM', 'NVIDIA GeForce RTX 4080 SUPER', 'Tesla
#   │ V100-PCIE-16GB', 'Tesla V100-SXM2-32GB', 'NVIDIA RTX 5000 Ada Generation',
#   │ 'NVIDIA GeForce RTX 4070 Ti', 'NVIDIA RTX 4000 SFF Ada Generation', 'NVIDIA
#   │ GeForce RTX 3090 Ti', 'NVIDIA RTX A2000', 'NVIDIA GeForce RTX 4080',
#   │ 'NVIDIA A30', 'NVIDIA GeForce RTX 5080', 'Tesla V100-FHHL-16GB', 'NVIDIA
#   │ H200 NVL', 'Tesla V100-SXM2-16GB', 'NVIDIA RTX PRO 6000 Blackwell Max-Q
#   │ Workstation Edition', 'NVIDIA A5000 Ada', 'Tesla V100-PCIE-32GB', 'NVIDIA
#   │ RTX A4500', 'NVIDIA  A30', 'NVIDIA GeForce RTX 3080TI', 'Tesla T4', 'NVIDIA
#   │ RTX A30'


  # ── Storage ───────────────────────────────────────────────────────────────────
  # volume_in_gb         = 20
  volume_mount_path    = "/workspace"
  # container_disk_in_gb = 10

  # Attach an existing network volume (leave empty to skip).
  network_volume_id = dependency.network_volume.outputs.network_volume_id

  # ── Networking ────────────────────────────────────────────────────────────────
  support_public_ip = true
  ports = [
    "22/tcp",     # SSH
    # "8888/http",  # Jupyter
  ]

  # ── Environment ───────────────────────────────────────────────────────────────
  # RunPod reads PUBLIC_KEY to authorize SSH access.
#   env = {
#     PUBLIC_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAs3wvdKsWeWZtq6YY9ielQG9irNpEsjqtRQO8uLmsps victormacedo996@gmail.com"
#   }

  # ── Optional start command ────────────────────────────────────────────────────
  # docker_start_cmd = ["bash", "-c", "jupyter lab --ip=0.0.0.0 --no-browser"]
}

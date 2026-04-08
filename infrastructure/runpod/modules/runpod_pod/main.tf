resource "runpod_pod" "gpu_instance" {
  name            = var.name
  image_name      = var.image_name
  gpu_type_ids    = var.gpu_type_ids
  data_center_ids = var.data_center_ids
  vcpu_count      = var.vcpu_count

  gpu_count         = var.gpu_count
  cloud_type        = var.cloud_type
  support_public_ip = var.support_public_ip
  network_volume_id = var.network_volume_id

  volume_in_gb         = var.volume_in_gb
  volume_mount_path    = var.volume_mount_path
  container_disk_in_gb = var.container_disk_in_gb

  ports            = var.ports
  env              = var.env
  docker_start_cmd = length(var.docker_start_cmd) > 0 ? var.docker_start_cmd : null
}
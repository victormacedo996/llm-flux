resource "runpod_network_volume" "this" {
  name           = var.name
  size           = var.size
  data_center_id = var.data_center_id
}

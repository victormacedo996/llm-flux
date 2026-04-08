variable "name" {
  type        = string
  default     = ""
  description = "name of the instance"
}

variable "image_name" {
  type        = string
  default     = ""
  description = "image name to be used in the instance"
}

variable "gpu_type_ids" {
  type        = list(string)
  default     = []
  description = "gpu type ids to be used in the instance"
}

variable "data_center_ids" {
  type        = list(string)
  default     = []
  description = "data center ids to be used in the instance"
}

variable "gpu_count" {
  type        = number
  default     = 1
  description = "number of GPUs to be used in the instance"
}

variable "cloud_type" {
  type        = string
  default     = "COMMUNITY"
  description = "type of cloud to be used in the instance"
}

variable "support_public_ip" {
  type        = bool
  default     = true
  description = "whether the instance should have a public IP"
}

variable "network_volume_id" {
  type        = string
  default     = ""
  description = "ID of the network volume to be used in the instance"
}

variable "volume_in_gb" {
  type        = number
  default     = 20
  description = "the amount of disk space, in gigabytes (GB), to allocate on the Pod volume. Data is persisted across Pod restarts."
}

variable "container_disk_in_gb" {
  type        = number
  default     = 20
  description = "size of the container disk in GB"
}

variable "vcpu_count" {
  type        = number
  default     = null
  description = "number of vCPUs to allocate (CPU Pods only)"
}

variable "env" {
  type        = map(string)
  default     = {}
  description = "environment variables to set on the Pod (e.g. { PUBLIC_KEY = \"ssh-rsa ...\" })"
}

variable "volume_mount_path" {
  type        = string
  default     = "/workspace"
  description = "absolute path where the network volume will be mounted"
}

variable "docker_start_cmd" {
  type        = list(string)
  default     = []
  description = "overrides the Docker image start CMD when non-empty"
}

variable "ports" {
  type        = list(string)
  default     = ["8888/http", "22/tcp"]
  description = "list of ports in the format <PORT_NUMBER>/<PROTOCOL> to be exposed on the instance"
}
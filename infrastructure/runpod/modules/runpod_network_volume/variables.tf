variable "name" {
  type        = string
  description = "A user-defined name for the Network Volume. The name does not need to be unique."
}

variable "size" {
  type        = number
  description = "The amount of disk space, in gigabytes (GB), allocated to the Network Volume. Must be between 0 and 4000."

  validation {
    condition     = var.size >= 0 && var.size <= 4000
    error_message = "size must be between 0 and 4000 GB."
  }
}

variable "data_center_id" {
  type        = string
  description = "The RunPod data center ID where the Network Volume is located (e.g. \"US-TX-3\")."
}

output "pod_id" {
  description = "The unique identifier of the Pod."
  value       = runpod_pod.gpu_instance.id
}

output "public_ip" {
  description = "The public IP address of the Pod."
  value       = runpod_pod.gpu_instance.public_ip
}

output "actual_data_center" {
  description = "The actual data center where the Pod was deployed."
  value       = runpod_pod.gpu_instance.actual_data_center
}

output "cost_per_hr" {
  description = "The cost in RunPod credits per hour of running the Pod."
  value       = runpod_pod.gpu_instance.cost_per_hr
}

output "desired_status" {
  description = "The current expected status of the Pod."
  value       = runpod_pod.gpu_instance.desired_status
}

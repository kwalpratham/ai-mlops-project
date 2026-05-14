output "ecr_repository_url" {
  description = "ECR repository URL for the API image"
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "ECS service name"
  value       = aws_ecs_service.api.name
}

output "s3_artifacts_bucket" {
  description = "S3 bucket for model artifacts"
  value       = aws_s3_bucket.artifacts.id
}

output "cloudwatch_log_group" {
  description = "CloudWatch log group for API logs"
  value       = aws_cloudwatch_log_group.api.name
}

# GCP Configuration
variable "gcp_project_id" {
  description = "The GCP project ID."
  type        = string
}

variable "gcp_region" {
  description = "The GCP region for all resources."
  type        = string
  default     = "europe-west4"
}

# GKE Configuration
variable "gke_cluster_name" {
  description = "The name of the GKE cluster."
  type        = string
  default     = "mlflow-cluster"
}

variable "gke_node_count" {
  description = "The number of nodes in the GKE cluster."
  type        = number
  default     = 1
}

variable "gke_machine_type" {
  description = "The machine type for GKE nodes."
  type        = string
  default     = "e2-medium"
}

# Cloud SQL (PostgreSQL) Configuration
variable "db_instance_name" {
  description = "The name of the Cloud SQL instance."
  type        = string
  default     = "mlflow-postgres-db"
}

variable "db_name" {
  description = "The name of the database."
  type        = string
  default     = "mlflow_db"
}

variable "db_user" {
  description = "The database username."
  type        = string
  default     = "mlflow_user"
}

variable "db_password" {
  description = "The database password."
  type        = string
  sensitive   = true
}

# GCS Configuration
variable "gcs_bucket_name" {
  description = "The name of the GCS bucket for artifacts."
  type        = string
}
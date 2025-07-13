terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = ">= 4.0.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = ">= 2.0.0"
    }
  }
}

provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# 1. GCS Bucket for MLflow Artifacts
resource "google_storage_bucket" "mlflow_artifacts" {
  name          = var.gcs_bucket_name
  location      = var.gcp_region
  force_destroy = true # Set to false in production
  uniform_bucket_level_access = true
}

# 2. Cloud SQL for PostgreSQL Instance
resource "google_sql_database_instance" "postgres" {
  name             = var.db_instance_name
  database_version = "POSTGRES_13"
  region           = var.gcp_region

  deletion_protection = false

  settings {
    tier = "db-g1-small"
  }
}

resource "google_sql_database" "mlflow_db" {
  name     = var.db_name
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "mlflow_user" {
  name     = var.db_user
  instance = google_sql_database_instance.postgres.name
  password = var.db_password
}

# 3. GKE Cluster
resource "google_container_cluster" "primary" {
  name     = var.gke_cluster_name
  location = var.gcp_region

  deletion_protection = false

  remove_default_node_pool = true
  initial_node_count       = 1

  network    = "default"
  subnetwork = "default"
}

resource "google_container_node_pool" "primary_nodes" {
  name       = "${google_container_cluster.primary.name}-node-pool"
  location   = var.gcp_region
  cluster    = google_container_cluster.primary.name
  node_count = var.gke_node_count

  node_config {
    machine_type = var.gke_machine_type
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
}

# 4. Kubernetes Provider Configuration
data "google_client_config" "default" {}

provider "kubernetes" {
  host                   = "https://${google_container_cluster.primary.endpoint}"
  token                  = data.google_client_config.default.access_token
  cluster_ca_certificate = base64decode(google_container_cluster.primary.master_auth.0.cluster_ca_certificate)
}

# 5. Kubernetes Deployment and Service for MLflow
resource "kubernetes_secret" "db_secret" {
  metadata {
    name = "mlflow-db-secret"
  }
  data = {
    DB_USER = var.db_user
    DB_PASS = var.db_password
  }
}

resource "kubernetes_deployment" "mlflow" {
  metadata {
    name = "mlflow-server"
    labels = {
      app = "mlflow"
    }
  }

  spec {
    replicas = 1

    selector {
      match_labels = {
        app = "mlflow"
      }
    }

    template {
      metadata {
        labels = {
          app = "mlflow"
        }
      }

      spec {
        container {
          image = "ghcr.io/mlflow/mlflow:v3.1.1"
          name  = "mlflow"

          args = [
            "--backend-store-uri",
            "postgresql://${kubernetes_secret.db_secret.data.DB_USER}:${kubernetes_secret.db_secret.data.DB_PASS}@${google_sql_database_instance.postgres.ip_address.0.ip_address}/${google_sql_database.mlflow_db.name}",
            "--default-artifact-root",
            "gs://${google_storage_bucket.mlflow_artifacts.name}",
            "--host",
            "0.0.0.0"
          ]

          port {
            container_port = 5000
          }
        }
      }
    }
  }
}

resource "kubernetes_service" "mlflow" {
  metadata {
    name = "mlflow-service"
  }
  spec {
    selector = {
      app = kubernetes_deployment.mlflow.spec.0.template.0.metadata.0.labels.app
    }
    port {
      port        = 80
      target_port = 5000
    }

    type = "LoadBalancer"
  }
}
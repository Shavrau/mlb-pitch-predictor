-- =============================================================================
-- Unity Catalog Environment Setup Script for Baseball ML Pitch Predictor
-- Idempotent DDL for reproducibility across environments (Dev / Staging / Prod)
-- =============================================================================

-- Step 1: Create Catalog (if user has metastore admin privileges)
-- Note: In shared environments or Free Edition, 'main' is commonly used.
CREATE CATALOG IF NOT EXISTS ${catalog};
USE CATALOG ${catalog};

-- Step 2: Create Schema for Baseball ML
CREATE SCHEMA IF NOT EXISTS ${catalog}.${schema}
COMMENT 'Schema dedicated to MLB Statcast pitch predictor pipelines and ML models';

-- Step 3: Create Managed Volume for raw Statcast Parquet landing
CREATE VOLUME IF NOT EXISTS ${catalog}.${schema}.${volume}
COMMENT 'Managed volume for staging raw Statcast Parquet batches before ingestion into Bronze @dp pipeline';

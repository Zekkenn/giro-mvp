-- =============================================================================
-- Giro Agent - Database Initialization Script
-- =============================================================================
-- This script runs when the PostgreSQL container is first created
-- =============================================================================

-- Create extensions if needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- For UUID generation
CREATE EXTENSION IF NOT EXISTS "pg_trgm";     -- For text search optimization

-- Set timezone to UTC
SET timezone = 'UTC';

-- Create database if it doesn't exist (this runs before Alembic)
-- Note: The database is actually created by Docker's POSTGRES_DB env var
-- This is just for reference and additional setup

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE '✅ Giro Agent database initialized successfully';
    RAISE NOTICE 'Database: %', current_database();
    RAISE NOTICE 'Timezone: UTC';
    RAISE NOTICE 'Extensions: uuid-ossp, pg_trgm';
END $$;
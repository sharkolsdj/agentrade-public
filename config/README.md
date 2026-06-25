# config/

This directory holds production configuration for the system (container
orchestration, service wiring, and environment composition).

The production configuration is intentionally not included in this public
repository. The application reads all credentials and connection parameters
from environment variables; see `.env.example` in the repository root for the
full list of required variables.

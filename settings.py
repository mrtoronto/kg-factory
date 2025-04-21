from local_settings import OPENAI_API_KEY, YOUR_EMAIL

# Output Settings
OUTPUT_FORMAT = "json"  # Options: "json", "neo4j"
JSON_OUTPUT_PATH = "knowledge_graph.json"  # Where to save the JSON output

# Neo4j Database Settings (only needed if OUTPUT_FORMAT = "neo4j")
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your_password"  # Change this to your Neo4j password

# OpenAI Settings
OPENAI_API_KEY = OPENAI_API_KEY
OPENAI_MODEL = "gpt-4o-mini"  # Model to use for entity extraction

# Current MeSH term being processed
CURRENT_MESH_TERM = "Athletic Performance"

# Import mesh configuration loader
from generate_mesh_config import load_mesh_config, generate_mesh_config

def get_mesh_config(mesh_term: str):
    """Get or generate mesh configuration for the given term."""
    try:
        return load_mesh_config(mesh_term)
    except FileNotFoundError:
        print(f"No configuration found for {mesh_term}, generating new configuration...")
        generate_mesh_config(mesh_term)
        return load_mesh_config(mesh_term)

# Load configuration for current MeSH term
mesh_config = get_mesh_config(CURRENT_MESH_TERM)

# Node Types, Relationships, and Properties are now dynamically set based on MeSH term
ALLOWED_NODE_TYPES = mesh_config.allowed_nodes
NODE_PROPERTIES = mesh_config.node_properties

DEFAULT_EMAIL = YOUR_EMAIL
DEFAULT_API_KEY = None  # Set your API key here or use environment variable
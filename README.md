# Molecular Biology Knowledge Graph Converter

This script converts unstructured molecular biology text documents into a knowledge graph. The graph can be saved either as a JSON file or stored in a Neo4j database.

## Prerequisites

1. Python 3.8 or higher
2. OpenAI API key
3. Neo4j database (optional - only if using Neo4j output)

## Installation

1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Configure your settings:
   - Open `settings.py`
   - Update the following values:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `OUTPUT_FORMAT`: Choose "json" or "neo4j"
     - If using Neo4j:
       - `NEO4J_PASSWORD`: Your Neo4j database password
       - Other Neo4j settings as needed (URI, username)
     - If using JSON:
       - `JSON_OUTPUT_PATH`: Where to save the JSON file (default: "knowledge_graph.json")

## Usage

Run the script with the following command:

```bash
python add_to_kg.py input_file.txt
```

## Output Formats

### JSON Output
When `OUTPUT_FORMAT = "json"`, the script will save the knowledge graph as a JSON file with the following structure:

```json
{
  "nodes": [
    {
      "id": "node_id",
      "type": "node_type",
      "properties": {
        "name": "...",
        "description": "...",
        ...
      }
    }
  ],
  "relationships": [
    {
      "source": {
        "id": "source_node_id",
        "type": "source_node_type"
      },
      "target": {
        "id": "target_node_id",
        "type": "target_node_type"
      },
      "type": "relationship_type",
      "properties": {}
    }
  ]
}
```

### Neo4j Output
When `OUTPUT_FORMAT = "neo4j"`, the script will store the knowledge graph in a Neo4j database.

## Configuration

All configuration is done through `settings.py`:

### Output Settings
- `OUTPUT_FORMAT`: Choose between "json" or "neo4j"
- `JSON_OUTPUT_PATH`: Path where to save the JSON file (if using JSON output)

### Database Settings (only needed for Neo4j output)
- `NEO4J_URI`: Neo4j database URI (default: bolt://localhost:7687)
- `NEO4J_USERNAME`: Neo4j username (default: neo4j)
- `NEO4J_PASSWORD`: Neo4j password

### OpenAI Settings
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_MODEL`: Model to use for entity extraction (default: gpt-4-turbo)

### Knowledge Graph Schema
The biological schema is fully configurable in settings.py:

Node Types:
- Gene
- Protein
- Disease
- Pathway
- Drug
- CellType
- Organism
- Phenotype
- MolecularFunction
- BiologicalProcess

Key Relationships:
- ENCODES (Gene → Protein)
- INTERACTS_WITH (Protein → Protein)
- PARTICIPATES_IN (Protein → Pathway)
- ASSOCIATED_WITH (Gene/Protein → Disease)
- TARGETS (Drug → Protein)
- TREATS (Drug → Disease)
- EXPRESSED_IN (Gene → CellType)
- REGULATES (Protein → BiologicalProcess)
- And more...

Node Properties:
- id (e.g., HGNC, UniProt identifiers)
- name
- description
- synonyms
- organism
- location
- evidence_level
- database_refs

You can modify any of these settings in `settings.py` to customize the knowledge graph structure for your specific needs. 
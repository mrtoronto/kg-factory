# Molecular Biology Knowledge Graph Converter

This project converts PubMed articles into a structured knowledge graph. The workflow involves first scraping relevant articles from PubMed based on MeSH keywords, then converting them into a knowledge graph that can be saved as JSON or stored in a Neo4j database.

## Prerequisites

1. Python 3.8 or higher
2. Poetry (Python dependency management)
3. OpenAI API key
4. Neo4j database (optional - only if using Neo4j output)

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Generate the lock file and install project dependencies:
```bash
poetry lock
poetry install
```

3. Configure your settings:
   - Make a local_settings.py file with the following:
     - `touch local_settings.py`
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `YOUR_EMAIL`: Your email
   - Open `settings.py`
   - Update the following values:
     - `OUTPUT_FORMAT`: Choose "json" or "neo4j"
     - If using Neo4j:
       - `NEO4J_PASSWORD`: Your Neo4j database password
       - Other Neo4j settings as needed (URI, username)
     - If using JSON:
       - `JSON_OUTPUT_PATH`: Where to save the JSON file (default: "knowledge_graph.json")

## Usage

The workflow consists of two main steps:

### 1. Scraping PubMed Articles

Use the PubMed scraper to search for articles based on MeSH keywords:

```bash
poetry run python pubmed_scraper.py "your_mesh_keyword" --max-results 10000
```

For example:
```bash
poetry run python pubmed_scraper.py "running" --max-results 10000
```

This will create a JSON file containing the scraped articles. The filename will be displayed in the console logs, following the format: `pubmed_results_YYYYMMDD_HHMMSS.json`

### 2. Creating the Knowledge Graph

Once you have the PubMed results file, you can create the knowledge graph:

```bash
poetry run python add_to_kg.py your_pubmed_results_file.json
```

For example:
```bash
poetry run python add_to_kg.py pubmed_results_20250420_145233.json
```

## Performance Considerations

### PubMed Rate Limits
- Without a PubMed API key: Limited to 3 articles per second
- With a PubMed API key: Limited to 10 articles per second

### Processing Times
- Knowledge graph generation: Approximately 20 seconds per article (using GPT to extract entities and relationships)

Please consider these limitations when planning your queries. For example, processing 1,000 articles would take:
- Download time: ~5.5 minutes with API key, ~5.5 minutes without
- Processing time: ~5.5 hours for knowledge graph generation

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
import os
import json
import argparse
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import settings

class RelationshipType(BaseModel):
    source: str = Field(description="Source node type")
    type: str = Field(description="Relationship type")
    target: str = Field(description="Target node type")

class MeshConfig(BaseModel):
    mesh_term: str = Field(description="The MeSH term this configuration is for")
    allowed_nodes: List[str] = Field(description="List of allowed node types in the knowledge graph")
    node_properties: List[str] = Field(description="List of properties that can be extracted for nodes")
    extraction_prompt: str = Field(description="Custom prompt template for extracting information for this MeSH term")
    description: str = Field(description="Description of what this configuration aims to capture")
    examples: List[Dict[str, Any]] = Field(default=[], description="Example extractions to guide the LLM")

def create_mesh_config_prompt() -> ChatPromptTemplate:
    """Create a prompt template for generating mesh configurations."""
    template = """You are an expert in biomedical knowledge graphs and information extraction.
Given the MeSH term "{mesh_term}", create a comprehensive configuration for extracting relevant information from scientific articles.

The configuration should include:
1. A list of relevant node types (entities) that would be important to track
2. Important properties that should be extracted for the nodes
3. A specialized prompt template that will guide the LLM in extracting this information, including:
   - What kinds of relationships to look for
   - What specific aspects of {mesh_term} to focus on
   - How to handle measurements, conditions, and temporal information
4. A clear description of what this configuration aims to capture
5. Optional: 1-2 brief examples of the kind of information we want to extract

Consider the domain carefully and focus on entities that are most relevant to {mesh_term}.
The prompt template should be specific and guide the LLM to extract precisely the kind of information we want,
while allowing flexibility in the types of relationships that can be discovered.

Format the response according to the following schema:
{format_instructions}
"""
    
    parser = PydanticOutputParser(pydantic_object=MeshConfig)
    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    return prompt, parser

def generate_mesh_config(mesh_term: str, output_dir: str = "mesh_configs") -> None:
    """Generate a new mesh configuration for the given MeSH term."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ChatOpenAI
    llm = ChatOpenAI(
        temperature=0.7,  # Some creativity is good for this task
        model_name=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
    
    # Create prompt and parser
    prompt, parser = create_mesh_config_prompt()
    
    # Generate configuration
    chain = prompt | llm | parser
    
    # Call the chain
    config = chain.invoke({"mesh_term": mesh_term})
    
    # Save configuration
    output_file = os.path.join(output_dir, f"{mesh_term.lower().replace(' ', '_')}.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config.model_dump(), f, indent=2)
    
    print(f"Generated configuration for '{mesh_term}' saved to {output_file}")

def load_mesh_config(mesh_term: str, config_dir: str = "mesh_configs") -> MeshConfig:
    """Load a mesh configuration from file."""
    config_file = os.path.join(config_dir, f"{mesh_term.lower().replace(' ', '_')}.json")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"No configuration found for MeSH term: {mesh_term}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = json.load(f)
    
    return MeshConfig(**config_data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MeSH term configuration")
    parser.add_argument("mesh_term", help="MeSH term to generate configuration for")
    parser.add_argument("--output-dir", default="mesh_configs", help="Directory to save configurations")
    
    args = parser.parse_args()
    generate_mesh_config(args.mesh_term, args.output_dir) 
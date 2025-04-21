import os
import json
import time
import logging
from typing import List, Optional, Tuple, Dict, Generator, Any
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph
import settings
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from itertools import islice
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import signal
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from settings import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OUTPUT_FORMAT,
    JSON_OUTPUT_PATH,
    CURRENT_MESH_TERM
)
from generate_mesh_config import load_mesh_config, MeshConfig
from langchain.prompts import ChatPromptTemplate
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Processing timed out")

# Set up timeout handler
signal.signal(signal.SIGALRM, timeout_handler)

def setup_environment():
    """Set up environment variables for OpenAI and optionally Neo4j from settings."""
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    if settings.OUTPUT_FORMAT == "neo4j":
        os.environ["NEO4J_URI"] = settings.NEO4J_URI
        os.environ["NEO4J_USERNAME"] = settings.NEO4J_USERNAME
        os.environ["NEO4J_PASSWORD"] = settings.NEO4J_PASSWORD

def create_graph_transformer(mesh_config: MeshConfig) -> LLMGraphTransformer:
    """Create a graph transformer with the appropriate configuration."""
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY
    )
    
    # Create a prompt that includes the mesh term and extraction prompt
    prompt = ChatPromptTemplate.from_template(
        f"""Given the following text about {mesh_config.mesh_term}, extract a knowledge graph following these specifications:

{mesh_config.extraction_prompt}

Text: {{input}}

Respond with ONLY a JSON object containing 'nodes' and 'relationships' lists. Each node should have 'id', 'type', and 'properties'. Each relationship should have 'source', 'target', 'type', and 'properties'.
"""
    )
    
    return LLMGraphTransformer(
        llm=llm,
        prompt=prompt,
        allowed_nodes=mesh_config.allowed_nodes,
        node_properties=mesh_config.node_properties,
        strict_mode=True
    )

def create_document_from_article(article: Dict[str, Any]) -> Document:
    """Create a Document object from an article dictionary."""
    # Combine title and abstract for processing
    content = f"Title: {article.get('title', '')}\nAbstract: {article.get('abstract', '')}"
    
    # Add full text if available
    if article.get('full_text'):
        content += f"\nFull Text: {article['full_text']}"
    
    # Create metadata
    metadata = {
        "pmid": article.get("pmid"),
        "pmc_id": article.get("pmc_id"),
        "doi": article.get("doi"),
        "authors": article.get("authors", []),
        "journal": article.get("journal"),
        "mesh_term": CURRENT_MESH_TERM
    }
    
    return Document(page_content=content, metadata=metadata)

def node_to_dict(node) -> Dict[str, Any]:
    """Convert a Node object to a dictionary."""
    return {
        "type": node.type,
        "properties": dict(node.properties)
    }

def relationship_to_dict(rel) -> Dict[str, Any]:
    """Convert a Relationship object to a dictionary."""
    return {
        "source": node_to_dict(rel.source),
        "target": node_to_dict(rel.target),
        "type": rel.type,
        "properties": dict(rel.properties)
    }

def process_article(article: Dict[str, Any], mesh_config: MeshConfig) -> Dict[str, Any]:
    """Process a single article and return the extracted graph data."""
    try:
        document = create_document_from_article(article)
        
        # Create transformer with mesh config
        transformer = create_graph_transformer(mesh_config)
        
        # Split text into smaller chunks (4000 chars instead of 8000)
        text = document.page_content
        chunk_size = 4000
        overlap = 200
        
        # Split on sentence boundaries where possible
        chunks = []
        start = 0
        while start < len(text):
            # Find the end of the chunk
            end = start + chunk_size
            if end < len(text):
                # Try to find a sentence boundary
                next_period = text.find('. ', end - 100, end + 100)
                if next_period != -1:
                    end = next_period + 1
            
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        # Process each chunk
        nodes = []
        relationships = []
        
        for i, chunk in enumerate(chunks):
            try:
                chunk_doc = Document(page_content=chunk, metadata=document.metadata)
                graph_doc = transformer.process_response(chunk_doc)
                
                if graph_doc:
                    nodes.extend(graph_doc.nodes)
                    relationships.extend(graph_doc.relationships)
                    
            except Exception as e:
                if "token limit" in str(e).lower():
                    logging.warning(f"Token limit reached for chunk {i+1}/{len(chunks)} of article {document.metadata.get('pmid', 'unknown')}")
                    # Try splitting the chunk further if it's too large
                    subchunks = [chunk[:len(chunk)//2], chunk[len(chunk)//2:]]
                    for subchunk in subchunks:
                        try:
                            subchunk_doc = Document(page_content=subchunk, metadata=document.metadata)
                            sub_graph_doc = transformer.process_response(subchunk_doc)
                            if sub_graph_doc:
                                nodes.extend(sub_graph_doc.nodes)
                                relationships.extend(sub_graph_doc.relationships)
                        except Exception as sub_e:
                            logging.error(f"Error processing subchunk: {sub_e}")
                else:
                    logging.error(f"Error processing chunk: {e}")
        
        # Return the extracted data
        return {
            'nodes': [node.dict() for node in nodes],
            'relationships': [rel.dict() for rel in relationships],
            'metadata': document.metadata
        }
        
    except Exception as e:
        logging.error(f"Error processing article: {e}")
        return {
            'nodes': [],
            'relationships': [],
            'metadata': article.get('metadata', {}),
            'error': str(e)
        }

def save_intermediate_results(results: List[Dict], output_path: str, suffix: str = ""):
    """Save intermediate results to avoid losing progress."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate_path = f"{output_path}.{timestamp}{suffix}"
    logging.info(f"Saving intermediate results to {intermediate_path}")
    
    merged_data = merge_graph_data(results)
    with open(intermediate_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2)
    logging.info(f"Saved intermediate results with {len(results)} articles processed")

def merge_graph_data(results: List[Dict[str, Any]], mesh_config: MeshConfig) -> Dict[str, Any]:
    """Merge results from multiple processed articles."""
    all_nodes = []
    all_relationships = []
    processed_articles = []
    failed_articles = []
    
    for result in results:
        if not result.get('error'):
            processed_articles.append({
                'pmid': result.get('metadata', {}).get('pmid'),
                'nodes': len(result.get('nodes', [])),
                'relationships': len(result.get('relationships', []))
            })
            all_nodes.extend(result.get('nodes', []))
            all_relationships.extend(result.get('relationships', []))
        else:
            failed_articles.append({
                'pmid': result.get('metadata', {}).get('pmid'),
                'error': result.get('error')
            })
    
    return {
        "nodes": all_nodes,
        "relationships": all_relationships,
        "mesh_term": mesh_config.mesh_term,
        "config": {
            "allowed_nodes": mesh_config.allowed_nodes,
            "node_properties": mesh_config.node_properties,
            "extraction_prompt": mesh_config.extraction_prompt,
            "description": mesh_config.description
        },
        "stats": {
            "total_articles": len(results),
            "successful_articles": len(processed_articles),
            "failed_articles": len(failed_articles),
            "processed_articles": processed_articles,
            "failed_article_details": failed_articles
        }
    }

def create_knowledge_graph(input_file: str, num_processes: int = 10) -> None:
    """Create a knowledge graph from the input articles using multiprocessing."""
    try:
        # Load mesh configuration
        mesh_config = load_mesh_config(CURRENT_MESH_TERM)
        logging.info(f"Loaded mesh configuration for term: {mesh_config.mesh_term}")
        
        # Load articles
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Ensure we have the correct structure - articles should be a list of dictionaries
        if isinstance(data, dict) and 'articles' in data:
            articles = data['articles']
        elif isinstance(data, list):
            articles = data
        else:
            raise ValueError("Input file must contain either a list of articles or a dictionary with 'articles' key")
            
        # Limit to 25 articles for testing
        articles = articles[:25]
        
        total_articles = len(articles)
        logging.info(f"Total articles to process: {total_articles}")
        
        # Create a partial function with mesh_config already set
        process_article_with_config = partial(process_article, mesh_config=mesh_config)
        
        # Process articles in parallel
        with mp.Pool(num_processes) as pool:
            with tqdm(total=total_articles, desc="Processing articles") as pbar:
                results = []
                for result in pool.imap_unordered(process_article_with_config, articles):
                    results.append(result)
                    pbar.update()
        
        # Merge all results
        final_graph = merge_graph_data(results, mesh_config)
        
        # Save final results
        output_file = f"data/knowledge_graph_{mesh_config.mesh_term}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(final_graph, f, indent=2)
        
        logging.info(f"\nProcessing complete!")
        logging.info(f"Total nodes: {len(final_graph['nodes'])}")
        logging.info(f"Total relationships: {len(final_graph['relationships'])}")
        logging.info(f"Results saved to: {output_file}")
        
    except Exception as e:
        logging.error(f"Error in create_knowledge_graph: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a knowledge graph from PubMed articles")
    parser.add_argument("input_file", help="Path to the JSON file containing PubMed articles")
    parser.add_argument("--processes", type=int, default=10, help="Number of parallel processes to use")
    args = parser.parse_args()
    
    create_knowledge_graph(args.input_file, args.processes)

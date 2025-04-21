import json
from generate_mesh_config import generate_mesh_config, load_mesh_config
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate

def test_mesh_config(mesh_term: str, test_text: str):
    """Test the mesh configuration generation and extraction for a given term."""
    print(f"\nTesting configuration for MeSH term: {mesh_term}")
    print("-" * 50)
    
    # Generate configuration if it doesn't exist
    try:
        config = load_mesh_config(mesh_term)
        print("Using existing configuration")
    except FileNotFoundError:
        print("Generating new configuration...")
        generate_mesh_config(mesh_term)
        config = load_mesh_config(mesh_term)
    
    # Print configuration details
    print("\nConfiguration Details:")
    print(f"Description: {config.description}")
    print(f"\nAllowed Node Types: {', '.join(config.allowed_nodes)}")
    print(f"\nNode Properties: {', '.join(config.node_properties)}")
    print("\nExtraction Prompt:")
    print(config.extraction_prompt)
    
    # Create custom prompt with citation instructions
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert in biomedical knowledge graph extraction. 
Extract entities and relationships from the provided text, focusing on {mesh_term} related information.

When extracting information:
1. Always include citation information using the document's metadata (PMID, DOI, etc.)
2. For each entity and relationship, specify which part of the text supports it
3. Include relevant properties like measurement values, units, conditions, etc.
4. Pay attention to temporal relationships and study conditions

{extraction_prompt}"""),
        ("human", "{text}"),
    ])
    
    # Create transformer with this configuration
    llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=config.allowed_nodes,
        node_properties=True,
        relationship_properties=True,
        allowed_relationships=[],
        prompt=prompt,
        strict_mode=False
    )
    
    # Create test document with metadata
    doc = Document(
        page_content=test_text,
        metadata={
            "pmid": "TEST123",
            "doi": "10.1234/test.2024",
            "mesh_term": mesh_term
        }
    )
    
    # Extract information
    print("\nExtracting information from test text...")
    graph_docs = transformer.convert_to_graph_documents([doc])
    
    # Print results
    print("\nExtracted Information:")
    for graph_doc in graph_docs:
        print("\nNodes:")
        for node in graph_doc.nodes:
            print(f"  Type: {node.type}")
            print(f"  Properties: {dict(node.properties)}")
            if "citation" in node.properties:
                print(f"  Citation: {node.properties['citation']}")
            if "text_evidence" in node.properties:
                print(f"  Evidence: {node.properties['text_evidence']}")
        
        print("\nRelationships:")
        for rel in graph_doc.relationships:
            print(f"  {rel.source.type} --[{rel.type}]--> {rel.target.type}")
            if rel.properties:
                print(f"  Properties: {dict(rel.properties)}")
                if "citation" in rel.properties:
                    print(f"  Citation: {rel.properties['citation']}")
                if "text_evidence" in rel.properties:
                    print(f"  Evidence: {rel.properties['text_evidence']}")

if __name__ == "__main__":
    # Test with a sample MeSH term and text
    test_mesh_term = "Athletic Performance"
    test_text = """
    A recent study investigated the effects of caffeine on athletic performance in endurance runners. 
    The research found that consuming 3-6mg of caffeine per kilogram of body weight improved running 
    time by an average of 2-3%. The caffeine supplementation also showed increased muscle endurance 
    and reduced perceived exertion during high-intensity exercise. However, some athletes experienced 
    side effects such as increased heart rate and mild anxiety.
    """
    
    test_mesh_config(test_mesh_term, test_text)
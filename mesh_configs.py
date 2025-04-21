"""
Configuration management for different MeSH terms.
Each configuration defines the relevant node types, relationships, and properties
for a specific medical/scientific domain.
"""

from typing import Dict, List, Tuple

class MeshConfig:
    def __init__(
        self,
        allowed_nodes: List[str],
        allowed_relationships: List[Tuple[str, str, str]],
        node_properties: List[str]
    ):
        self.allowed_nodes = allowed_nodes
        self.allowed_relationships = allowed_relationships
        self.node_properties = node_properties

# Base configuration that applies to all domains
BASE_NODE_PROPERTIES = [
    "_id",              # Standard identifiers
    "name",            # Common name
    "description",     # Brief description
    "synonyms",        # Alternative names
    "evidence_level",  # Confidence in the annotation
    "database_refs"    # References to other databases
]

# Configuration for Athletic Performance domain
ATHLETIC_PERFORMANCE_CONFIG = MeshConfig(
    allowed_nodes=[
        "Athlete",
        "Exercise",
        "PhysicalMetric",
        "TrainingProgram",
        "Supplement",
        "PhysiologicalState",
        "Equipment",
        "Injury",
        "RecoveryProtocol",
        "PerformanceMetric",
        "MuscleGroup",
        "MetabolicPathway",
        "NutritionalFactor"
    ],
    allowed_relationships=[
        ("Athlete", "PERFORMS", "Exercise"),
        ("Athlete", "FOLLOWS", "TrainingProgram"),
        ("Athlete", "USES", "Equipment"),
        ("Athlete", "EXPERIENCES", "Injury"),
        ("Athlete", "MAINTAINS", "PhysiologicalState"),
        ("Exercise", "TARGETS", "MuscleGroup"),
        ("Exercise", "IMPROVES", "PerformanceMetric"),
        ("TrainingProgram", "INCLUDES", "Exercise"),
        ("Supplement", "ENHANCES", "PerformanceMetric"),
        ("Supplement", "AFFECTS", "MetabolicPathway"),
        ("NutritionalFactor", "INFLUENCES", "PhysiologicalState"),
        ("RecoveryProtocol", "TREATS", "Injury"),
        ("Exercise", "ACTIVATES", "MetabolicPathway"),
        ("PhysiologicalState", "AFFECTS", "PerformanceMetric")
    ],
    node_properties=BASE_NODE_PROPERTIES + [
        "measurement_unit",     # For metrics
        "intensity_level",      # For exercises
        "duration",            # For protocols
        "frequency",           # For training programs
        "contraindications",   # For exercises and supplements
        "certification_level"  # For athletes
    ]
)

# Configuration for Molecular Biology domain (your original configuration)
MOLECULAR_BIOLOGY_CONFIG = MeshConfig(
    allowed_nodes=[
        "Gene",
        "Protein",
        "Disease",
        "Pathway",
        "Drug",
        "CellType",
        "Organism",
        "Phenotype",
        "MolecularFunction",
        "BiologicalProcess"
    ],
    allowed_relationships=[
        ("Gene", "ENCODES", "Protein"),
        ("Protein", "INTERACTS_WITH", "Protein"),
        ("Protein", "PARTICIPATES_IN", "Pathway"),
        ("Gene", "ASSOCIATED_WITH", "Disease"),
        ("Protein", "ASSOCIATED_WITH", "Disease"),
        ("Drug", "TARGETS", "Protein"),
        ("Drug", "TREATS", "Disease"),
        ("Gene", "EXPRESSED_IN", "CellType"),
        ("Protein", "REGULATES", "BiologicalProcess"),
        ("Gene", "FOUND_IN", "Organism"),
        ("Disease", "HAS_PHENOTYPE", "Phenotype"),
        ("Protein", "HAS_FUNCTION", "MolecularFunction"),
        ("Gene", "REGULATES", "Gene")
    ],
    node_properties=BASE_NODE_PROPERTIES + [
        "organism",        # Source organism
        "location",       # Cellular/chromosomal location
    ]
)

# Dictionary mapping MeSH terms to their configurations
MESH_CONFIGS: Dict[str, MeshConfig] = {
    "Athletic Performance": ATHLETIC_PERFORMANCE_CONFIG,
    "Molecular Biology": MOLECULAR_BIOLOGY_CONFIG,
    # Add more configurations as needed
}

def get_config_for_mesh_term(mesh_term: str) -> MeshConfig:
    """
    Get the configuration for a specific MeSH term.
    Falls back to Molecular Biology config if the term is not found.
    
    Args:
        mesh_term: The MeSH term to get configuration for
        
    Returns:
        MeshConfig object containing the configuration
    """
    return MESH_CONFIGS.get(mesh_term, MOLECULAR_BIOLOGY_CONFIG) 
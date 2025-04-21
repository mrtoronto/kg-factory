import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from collections import Counter
import logging
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()

def validate_article(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate a single article and return a dict of validation results.
    """
    results = {
        'has_pmid': bool(article.get('pmid')),
        'has_title': bool(article.get('title')),
        'has_abstract': bool(article.get('abstract')),
        'has_authors': bool(article.get('authors')),
        'has_journal': bool(article.get('journal')),
        'has_full_text': bool(article.get('full_text')),
        'has_pmc_id': bool(article.get('pmc_id')),
        'has_doi': bool(article.get('doi')),
        'text_lengths': {
            'title': len(str(article.get('title', ''))),
            'abstract': len(str(article.get('abstract', ''))),
            'full_text': len(str(article.get('full_text', '')))
        },
        'num_authors': len(article.get('authors', [])),
    }
    
    # Check if journal info is complete
    if results['has_journal']:
        journal_info = article['journal']
        results['has_journal_title'] = bool(journal_info.get('title'))
        results['has_pub_date'] = bool(journal_info.get('pub_date'))
    else:
        results['has_journal_title'] = False
        results['has_pub_date'] = False
    
    return results

def analyze_dataset(file_path: str) -> None:
    """
    Analyze the PubMed dataset and print comprehensive statistics.
    """
    try:
        # Load the JSON file
        console.print(f"\n[bold blue]Loading data from:[/bold blue] {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic dataset info
        mesh_term = data.get('mesh_term', 'N/A')
        total_results = data.get('total_results', 0)
        articles = data.get('articles', [])
        
        console.print(f"\n[bold green]Dataset Overview[/bold green]")
        console.print(f"MeSH Term: {mesh_term}")
        console.print(f"Total Articles: {total_results}")
        console.print(f"Articles in file: {len(articles)}")
        
        # Validate each article
        validation_results = []
        for article in track(articles, description="Analyzing articles..."):
            validation_results.append(validate_article(article))
        
        # Compute statistics
        stats = {
            'total_articles': len(validation_results),
            'with_pmid': sum(1 for r in validation_results if r['has_pmid']),
            'with_title': sum(1 for r in validation_results if r['has_title']),
            'with_abstract': sum(1 for r in validation_results if r['has_abstract']),
            'with_full_text': sum(1 for r in validation_results if r['has_full_text']),
            'with_pmc_id': sum(1 for r in validation_results if r['has_pmc_id']),
            'with_doi': sum(1 for r in validation_results if r['has_doi']),
            'with_authors': sum(1 for r in validation_results if r['has_authors']),
            'with_journal_title': sum(1 for r in validation_results if r['has_journal_title']),
            'with_pub_date': sum(1 for r in validation_results if r['has_pub_date']),
        }
        
        # Calculate text length statistics
        text_lengths = {
            'title': [],
            'abstract': [],
            'full_text': []
        }
        for result in validation_results:
            for field, length in result['text_lengths'].items():
                text_lengths[field].append(length)
        
        # Create results table
        table = Table(title="Data Completeness Analysis")
        table.add_column("Field", style="cyan")
        table.add_column("Count", justify="right", style="green")
        table.add_column("Percentage", justify="right", style="yellow")
        
        for field, count in stats.items():
            if field != 'total_articles':
                percentage = (count / stats['total_articles']) * 100
                table.add_row(
                    field.replace('_', ' ').title(),
                    str(count),
                    f"{percentage:.1f}%"
                )
        
        console.print("\n")
        console.print(table)
        
        # Print text length statistics
        console.print("\n[bold green]Text Length Statistics[/bold green]")
        for field, lengths in text_lengths.items():
            if lengths:
                avg_len = sum(lengths) / len(lengths)
                max_len = max(lengths)
                min_len = min(lengths)
                console.print(f"\n{field.title()}:")
                console.print(f"  Average length: {avg_len:.0f} characters")
                console.print(f"  Maximum length: {max_len:,} characters")
                console.print(f"  Minimum length: {min_len:,} characters")
        
        # Check for any potential issues
        console.print("\n[bold yellow]Potential Issues:[/bold yellow]")
        issues_found = False
        
        if stats['with_full_text'] == 0:
            console.print("⚠️  No articles have full text content")
            issues_found = True
        
        if stats['with_abstract'] < stats['total_articles']:
            console.print(f"⚠️  {stats['total_articles'] - stats['with_abstract']} articles are missing abstracts")
            issues_found = True
            
        if not issues_found:
            console.print("[green]No major issues found![/green]")
        
    except json.JSONDecodeError as e:
        console.print(f"[bold red]Error:[/bold red] Invalid JSON file: {str(e)}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "/Users/matttoronto/med-kg/pubmed_results_20250420_115807.json"
    
    analyze_dataset(file_path) 
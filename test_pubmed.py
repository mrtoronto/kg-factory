from Bio import Entrez
import json
from pprint import pprint
import xml.etree.ElementTree as ET

# Set up Entrez
Entrez.email = "matt.toronto97@gmail.com"

def fetch_and_examine_article(pmid: str):
    """Fetch a single article and examine its structure"""
    print(f"\n{'='*80}\nExamining PMID: {pmid}\n{'='*80}")
    
    # Fetch the article
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    xml_content = handle.read()
    handle.close()
    
    # Parse XML directly first
    root = ET.fromstring(xml_content)
    print("\nSearching for PMC references in XML:")
    for elem in root.findall(".//*"):
        if any(attr.lower().find('pmc') >= 0 for attr in elem.attrib.values()):
            print(f"Found PMC reference in element: {elem.tag}")
            print(f"Attributes: {elem.attrib}")
            print(f"Text: {elem.text}")
    
    # Now use Entrez parser
    handle = Entrez.efetch(db="pubmed", id=pmid, retmode="xml")
    records = Entrez.read(handle)
    handle.close()
    
    if not records['PubmedArticle']:
        print("No article found")
        return
    
    article = records['PubmedArticle'][0]
    
    # Check PubmedData section
    if 'PubmedData' in article:
        print("\nPubmedData section:")
        pprint(article['PubmedData'])
    
    # Check ArticleIdList
    if 'ArticleIdList' in article:
        print("\nArticle IDs:")
        for article_id in article['ArticleIdList']:
            print(f"Type: {article_id.attributes.get('IdType', '')}, Value: {str(article_id)}")
    
    # Save raw XML for inspection
    with open(f"pubmed_raw_{pmid}.xml", 'wb') as f:
        f.write(xml_content)
    print(f"\nRaw XML saved to pubmed_raw_{pmid}.xml")

def main():
    # Let's examine some recent PLoS One articles
    test_pmids = [
        "38099084",  # Recent PLoS One article
        "38099083",  # Recent PLoS One article
        "38099082",  # Recent PLoS One article
    ]
    
    for pmid in test_pmids:
        fetch_and_examine_article(pmid)

if __name__ == "__main__":
    main() 
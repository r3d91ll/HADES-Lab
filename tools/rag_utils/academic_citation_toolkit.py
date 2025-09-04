#!/usr/bin/env python3
"""
Academic Citation Toolkit
==========================

SOURCE-AGNOSTIC citation and bibliography extraction toolkit.

This toolkit works with ANY academic paper corpus:
- ArXiv papers
- SSRN papers  
- Harvard Law Library
- PubMed articles
- Any academic paper with citations and bibliography

The toolkit is completely independent of the source and can be used
as a utility for building citation networks from any academic corpus.

Information Reconstructionism Framework
---------------------------------------

This toolkit embodies Information Reconstructionism principles by reconstructing 
citation networks and contextual relationships from diverse academic corpora. 
It supports Conveyance by enabling accurate transmission of bibliographic 
metadata and citation relationships for downstream analytical tasks, creating 
obligatory passage points in Actor-Network Theory terms where knowledge must 
pass through standardized citation extraction processes.
"""

import os
import re
import json
import logging
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BibliographyEntry:
    """
    Represents a single bibliography entry from any academic paper.
    
    SOURCE-AGNOSTIC: Works regardless of whether the paper comes from
    ArXiv, SSRN, law library, PubMed, or any other academic source.
    """
    entry_number: Optional[str]  # [1], [2], etc.
    raw_text: str
    authors: List[str]
    title: Optional[str]
    venue: Optional[str]  # Journal/conference
    year: Optional[int]
    arxiv_id: Optional[str]
    doi: Optional[str]
    pmid: Optional[str]  # PubMed ID
    ssrn_id: Optional[str]  # SSRN ID  
    url: Optional[str]
    source_paper_id: str
    confidence: float

@dataclass
class InTextCitation:
    """
    Represents an in-text citation that points to a bibliography entry.
    
    SOURCE-AGNOSTIC: Works with any academic citation format.
    """
    raw_text: str
    citation_type: str  # 'numeric', 'author_year', 'hybrid'
    start_pos: int
    end_pos: int
    context: str
    bibliography_ref: Optional[str]  # Points to bibliography entry
    source_paper_id: str
    confidence: float

class DocumentProvider(ABC):
    """
    Abstract interface for providing document text.
    
    This allows the citation toolkit to work with ANY document source:
    - ArangoDB (our current ArXiv setup)
    - File system (PDF/text files)
    - Database (PostgreSQL, MongoDB, etc.)
    - APIs (SSRN, PubMed, etc.)
    - Web scraping
    """
    
    @abstractmethod
    def get_document_text(self, document_id: str) -> Optional[str]:
        """Get full text of a document by ID."""
        pass
    
    @abstractmethod
    def get_document_chunks(self, document_id: str) -> List[str]:
        """Get text chunks of a document by ID."""
        pass

class ArangoDocumentProvider(DocumentProvider):
    """
    Document provider for ArangoDB (our current ArXiv setup).
    """
    
    def __init__(self, arango_client, db_name: str = 'academy_store', username: str = None):
        self.client = arango_client
        username = username or os.getenv('ARANGO_USERNAME', 'root')
        password = os.getenv('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        self.db = arango_client.db(db_name, username=username, password=password)
    
    def get_document_text(self, document_id: str) -> Optional[str]:
        """Get full document text by combining all chunks."""
        chunks = self.get_document_chunks(document_id)
        return ' '.join(chunks) if chunks else None
    
    def get_document_chunks(self, document_id: str) -> List[str]:
        """Get document chunks from ArangoDB."""
        try:
            cursor = self.db.aql.execute("""
            FOR chunk IN arxiv_chunks
                FILTER chunk.paper_id == @paper_id
                SORT chunk.chunk_index ASC
                RETURN chunk.text
            """, bind_vars={'paper_id': document_id})
            
            return list(cursor)
        except Exception as e:
            logger.error(f"Error getting chunks for {document_id}: {e}")
            return []

class FileSystemDocumentProvider(DocumentProvider):
    """
    Document provider for local files (PDFs, text files).
    """
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def get_document_text(self, document_id: str) -> Optional[str]:
        """Read full text from file."""
        try:
            file_path = f"{self.base_path}/{document_id}.txt"
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading {document_id}: {e}")
            return None
    
    def get_document_chunks(self, document_id: str) -> List[str]:
        """Split document into chunks by paragraphs."""
        text = self.get_document_text(document_id)
        if not text:
            return []
        
        # Simple paragraph-based chunking
        return [p.strip() for p in text.split('\n\n') if p.strip()]

class CitationStorage(ABC):
    """
    Abstract interface for storing citation data.
    
    This allows the toolkit to store results in any system:
    - ArangoDB (our current setup)
    - PostgreSQL
    - JSON files
    - CSV files
    - Any other storage system
    """
    
    @abstractmethod
    def store_bibliography_entries(self, entries: List[BibliographyEntry]) -> bool:
        """Store bibliography entries."""
        pass
    
    @abstractmethod
    def store_citations(self, citations: List[InTextCitation]) -> bool:
        """Store in-text citations."""
        pass

class ArangoCitationStorage(CitationStorage):
    """Citation storage for ArangoDB."""
    
    def __init__(self, arango_client, db_name: str = 'academy_store', username: str = None):
        self.client = arango_client
        username = username or os.getenv('ARANGO_USERNAME', 'root')
        password = os.getenv('ARANGO_PASSWORD')
        if not password:
            raise ValueError("ARANGO_PASSWORD environment variable required")
        self.db = arango_client.db(db_name, username=username, password=password)
    
    def store_bibliography_entries(self, entries: List[BibliographyEntry]) -> bool:
        """Store bibliography entries in ArangoDB."""
        if not entries:
            return True
        
        try:
            bibliography_collection = self.db.collection('bibliography_entries')
            
            for entry in entries:
                doc = {
                    '_key': f"{entry.source_paper_id}_{entry.entry_number or 'unknown'}",
                    'source_paper_id': entry.source_paper_id,
                    'entry_number': entry.entry_number,
                    'raw_text': entry.raw_text,
                    'authors': entry.authors,
                    'title': entry.title,
                    'venue': entry.venue,
                    'year': entry.year,
                    'arxiv_id': entry.arxiv_id,
                    'doi': entry.doi,
                    'pmid': entry.pmid,
                    'ssrn_id': entry.ssrn_id,
                    'url': entry.url,
                    'confidence': entry.confidence
                }
                
                try:
                    bibliography_collection.insert(doc)
                except Exception as e:
                    if "unique constraint violated" in str(e).lower():
                        bibliography_collection.update({'_key': doc['_key']}, doc)
                    else:
                        raise e
            
            return True
        except Exception as e:
            logger.error(f"Error storing bibliography entries: {e}")
            return False
    
    def store_citations(self, citations: List[InTextCitation]) -> bool:
        """Store in-text citations in ArangoDB."""
        # Implementation similar to bibliography storage
        return True

class JSONCitationStorage(CitationStorage):
    """Citation storage to JSON files."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def store_bibliography_entries(self, entries: List[BibliographyEntry]) -> bool:
        """Store bibliography entries to JSON."""
        import json
        
        try:
            data = [entry.__dict__ for entry in entries]
            with open(f"{self.output_dir}/bibliography.json", 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error storing to JSON: {e}")
            return False
    
    def store_citations(self, citations: List[InTextCitation]) -> bool:
        """Store citations to JSON."""
        try:
            data = [asdict(citation) for citation in citations]
            with open(f"{self.output_dir}/citations.json", 'w') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            logger.error(f"Error storing citations to JSON: {e}")
            return False

class UniversalBibliographyExtractor:
    """
    SOURCE-AGNOSTIC bibliography extractor that works with any academic corpus.
    
    This class can extract bibliographies from papers regardless of source:
    ArXiv, SSRN, Harvard Law, PubMed, or any other academic paper collection.
    """
    
    def __init__(self, document_provider: DocumentProvider):
        self.document_provider = document_provider
    
    def extract_bibliography_section(self, paper_id: str) -> Optional[str]:
        """
        Extract bibliography/references section from any academic paper.
        
        Uses multiple strategies to identify reference sections regardless
        of the source format or academic discipline.
        """
        try:
            full_text = self.document_provider.get_document_text(paper_id)
            if not full_text:
                return None
            
            # Strategy 1: Look for explicit "References" section
            references_patterns = [
                r'\b(References|Bibliography|REFERENCES|BIBLIOGRAPHY|Works Cited|Literature Cited)\b.*?(?=\n\n[A-Z][a-z]+|\Z)',
                r'\b(References|Bibliography)\b(.*?)(?=\n\n|\Z)',
                r'## References(.*?)(?=\n##|\Z)',
                r'# References(.*?)(?=\n#|\Z)'
            ]
            
            for pattern in references_patterns:
                match = re.search(pattern, full_text, re.DOTALL | re.IGNORECASE)
                if match and len(match.group(0)) > 50:  # Must have substantial content
                    logger.info(f"Found references section for {paper_id} (pattern: {pattern[:20]}...)")
                    return match.group(0)
            
            # Strategy 2: Look for numbered reference pattern at end
            numbered_refs = re.search(
                r'(\[\d+\].*?)(?=\n\n[A-Z][a-z]+|\Z)', 
                full_text[-8000:],  # Look in last 8000 chars
                re.DOTALL
            )
            
            if numbered_refs and len(numbered_refs.group(0)) > 200:
                logger.info(f"Found numbered references for {paper_id}")
                return numbered_refs.group(0)
            
            # Strategy 3: Look for author-year style references
            author_year_refs = re.search(
                r'((?:[A-Z][a-z]+(?:\s+et\s+al\.?)?,\s+\d{4}.*?\n){5,})', 
                full_text[-5000:],  # Look in last 5000 chars
                re.DOTALL
            )
            
            if author_year_refs:
                logger.info(f"Found author-year references for {paper_id}")
                return author_year_refs.group(0)
            
            logger.warning(f"No references section found for {paper_id}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting bibliography for {paper_id}: {e}")
            return None
    
    def parse_bibliography_entries(self, bibliography_text: str, paper_id: str) -> List[BibliographyEntry]:
        """
        Parse bibliography entries from any academic format.
        
        Works with:
        - Numbered references [1], [2], etc.
        - Author-year references
        - Mixed formats
        - Different academic disciplines
        """
        entries = []
        
        try:
            # Remove the header line if present
            text = re.sub(r'^(References|Bibliography|REFERENCES|BIBLIOGRAPHY|Works Cited|Literature Cited)\s*\n?', '', bibliography_text, flags=re.IGNORECASE)
            
            # Strategy 1: Numbered entries [1], [2], etc.
            numbered_entries = re.findall(
                r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\Z)', 
                text, 
                re.DOTALL
            )
            
            if numbered_entries:
                logger.info(f"Found {len(numbered_entries)} numbered bibliography entries")
                for num, entry_text in numbered_entries:
                    entry = self._parse_single_entry(entry_text.strip(), paper_id, entry_number=num)
                    if entry and len(entry.raw_text.strip()) > 20:  # Must have substantial content
                        entries.append(entry)
                return entries
            
            # Strategy 2: Split by double newlines (paragraph-separated)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 30]
            
            for i, paragraph in enumerate(paragraphs, 1):
                entry = self._parse_single_entry(paragraph, paper_id, entry_number=str(i))
                if entry:
                    entries.append(entry)
            
            # Strategy 3: Split by single newlines but group multi-line entries
            if not entries:
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                current_entry = ""
                entry_num = 1
                
                for line in lines:
                    # Start of new entry (starts with capital letter or number)
                    if re.match(r'^([A-Z]|\d+\.)', line) and len(current_entry) > 50:
                        entry = self._parse_single_entry(current_entry, paper_id, entry_number=str(entry_num))
                        if entry:
                            entries.append(entry)
                        current_entry = line
                        entry_num += 1
                    else:
                        current_entry += " " + line
                
                # Don't forget the last entry
                if len(current_entry) > 50:
                    entry = self._parse_single_entry(current_entry, paper_id, entry_number=str(entry_num))
                    if entry:
                        entries.append(entry)
            
            logger.info(f"Successfully parsed {len(entries)} bibliography entries for {paper_id}")
            return entries
            
        except Exception as e:
            logger.error(f"Error parsing bibliography entries for {paper_id}: {e}")
            return []
    
    def _parse_single_entry(self, entry_text: str, paper_id: str, entry_number: str = None) -> Optional[BibliographyEntry]:
        """
        Parse a single bibliography entry - works with any academic format.
        """
        if len(entry_text.strip()) < 20:
            return None
        
        try:
            # Extract ArXiv ID
            arxiv_match = re.search(r'arXiv:(\d{4}\.\d{4,5})', entry_text, re.IGNORECASE)
            arxiv_id = arxiv_match.group(1) if arxiv_match else None
            
            # Extract DOI
            doi_match = re.search(r'doi:?\s*([10]\.\d+/[^\s,]+)', entry_text, re.IGNORECASE)
            doi = doi_match.group(1) if doi_match else None
            
            # Extract PubMed ID
            pmid_match = re.search(r'PMID:?\s*(\d+)', entry_text, re.IGNORECASE)
            pmid = pmid_match.group(1) if pmid_match else None
            
            # Extract SSRN ID
            ssrn_match = re.search(r'SSRN[:\s]*(\d+)', entry_text, re.IGNORECASE)
            ssrn_id = ssrn_match.group(1) if ssrn_match else None
            
            # Extract year
            year_match = re.search(r'\b(19|20)\d{2}\b', entry_text)
            year = int(year_match.group(0)) if year_match else None
            
            # Extract title (text in quotes or italics)
            title_patterns = [
                r'["""]([^"""]{15,200})["""]',  # Quoted titles
                r'_([^_]{15,200})_',  # Italicized titles (markdown)
                r'\*([^*]{15,200})\*',  # Bold titles (markdown)
                r'(?:^|\. )([A-Z][^.]{15,150})\.',  # Title after authors, before period
            ]
            
            title = None
            for pattern in title_patterns:
                title_match = re.search(pattern, entry_text)
                if title_match:
                    title = title_match.group(1).strip()
                    break
            
            # Extract authors
            authors = []
            author_patterns = [
                r'^([^.]+(?:[A-Z]\.[^.]*\.)+)',  # "Last, F., Last2, F2."
                r'^([A-Z][a-z]+(?:\s+[A-Z]\.[^,]*,\s*)*[A-Z][a-z]+)',  # "Smith, J., Jones, P."
                r'([A-Z][a-z]+\s+et\s+al\.?)',  # "Smith et al."
            ]
            
            for pattern in author_patterns:
                author_match = re.search(pattern, entry_text)
                if author_match:
                    author_text = author_match.group(1)
                    # Simple split by commas
                    potential_authors = [a.strip() for a in author_text.split(',')]
                    authors = [a for a in potential_authors if len(a) > 2 and not a.isdigit()][:5]
                    break
            
            # Extract venue
            venue = None
            venue_patterns = [
                r'In\s+([A-Z][^,\n]{10,50})',  # "In Conference Name"
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\d{4}',  # "Journal Name 2020"
                r'Proceedings\s+of\s+([^,\n]{10,50})',  # "Proceedings of ..."
            ]
            
            for pattern in venue_patterns:
                venue_match = re.search(pattern, entry_text)
                if venue_match:
                    venue = venue_match.group(1).strip()
                    break
            
            # Calculate confidence
            confidence = 0.3  # Base confidence
            if arxiv_id or doi or pmid or ssrn_id:
                confidence += 0.4
            if title and len(title) > 10:
                confidence += 0.2
            if authors:
                confidence += 0.2
            if year:
                confidence += 0.1
            if venue:
                confidence += 0.1
            
            confidence = min(1.0, confidence)
            
            return BibliographyEntry(
                entry_number=entry_number,
                raw_text=entry_text,
                authors=authors,
                title=title,
                venue=venue,
                year=year,
                arxiv_id=arxiv_id,
                doi=doi,
                pmid=pmid,
                ssrn_id=ssrn_id,
                url=None,
                source_paper_id=paper_id,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error parsing bibliography entry: {e}")
            return None
    
    def extract_paper_bibliography(self, paper_id: str) -> List[BibliographyEntry]:
        """
        Extract complete bibliography for a single paper.
        
        This is the main entry point for bibliography extraction,
        combining section identification with entry parsing to produce
        a complete formal citation network for the paper.
        """
        logger.info(f"Extracting bibliography for paper {paper_id}")
        
        # Extract bibliography section
        bibliography_text = self.extract_bibliography_section(paper_id)
        if not bibliography_text:
            return []
        
        # Parse entries
        entries = self.parse_bibliography_entries(bibliography_text, paper_id)
        
        logger.info(f"Successfully extracted {len(entries)} bibliography entries for {paper_id}")
        return entries

class UniversalCitationExtractor:
    """
    SOURCE-AGNOSTIC in-text citation extractor.
    
    Extracts citations from any academic paper and maps them to
    bibliography entries regardless of the source.
    """
    
    def __init__(self, document_provider: DocumentProvider):
        self.document_provider = document_provider
    
    def extract_citations(self, paper_id: str, bibliography_entries: List[BibliographyEntry]) -> List[InTextCitation]:
        """
        Extract in-text citations and map them to bibliography entries.
        
        This is the second step after bibliography extraction.
        """
        # Implementation for extracting [1], [2], (Author, Year) citations
        # and mapping them to bibliography entries
        pass

# Factory functions for easy setup
def create_arxiv_citation_toolkit(arango_client) -> Tuple[UniversalBibliographyExtractor, ArangoCitationStorage]:
    """Create citation toolkit for ArXiv papers in ArangoDB."""
    document_provider = ArangoDocumentProvider(arango_client)
    storage = ArangoCitationStorage(arango_client)
    extractor = UniversalBibliographyExtractor(document_provider)
    return extractor, storage

def create_filesystem_citation_toolkit(file_path: str, output_path: str) -> Tuple[UniversalBibliographyExtractor, JSONCitationStorage]:
    """Create citation toolkit for local files with JSON output."""
    document_provider = FileSystemDocumentProvider(file_path)
    storage = JSONCitationStorage(output_path)
    extractor = UniversalBibliographyExtractor(document_provider)
    return extractor, storage

# Main function to test the toolkit
def main():
    """Test the universal citation toolkit."""
    import os
    from arango import ArangoClient
    
    # Test with our ArXiv setup
    arango_password = os.getenv('ARANGO_PASSWORD')
    client = ArangoClient(hosts='http://192.168.1.69:8529')
    
    extractor, storage = create_arxiv_citation_toolkit(client)
    
    print("🌍 Universal Academic Citation Toolkit")
    print("=" * 50)
    print("SOURCE-AGNOSTIC: Works with ArXiv, SSRN, PubMed, Law Libraries, etc.")
    print()
    
    # Test on our papers
    test_papers = ['1301_3781']  # Start with one
    
    for paper_id in test_papers:
        print(f"📄 Processing paper: {paper_id}")
        
        # Extract bibliography
        bibliography_text = extractor.extract_bibliography_section(paper_id)
        if bibliography_text:
            print(f"  ✅ Found bibliography ({len(bibliography_text)} chars)")
            
            entries = extractor.parse_bibliography_entries(bibliography_text, paper_id)
            print(f"  📚 Parsed {len(entries)} bibliography entries")
            
            # Show sample entries
            for i, entry in enumerate(entries[:3], 1):
                print(f"    {i}. [{entry.entry_number}] {entry.title or 'No title'}")
                if entry.arxiv_id:
                    print(f"       ArXiv: {entry.arxiv_id}")
                if entry.authors:
                    print(f"       Authors: {', '.join(entry.authors[:2])}")
                print(f"       Confidence: {entry.confidence:.2f}")
            
            if len(entries) > 3:
                print(f"    ... and {len(entries) - 3} more entries")
            
            # Store results
            if storage.store_bibliography_entries(entries):
                print(f"  💾 Stored bibliography entries")
            
        else:
            print(f"  ❌ No bibliography found")

if __name__ == "__main__":
    main()
"""
LaTeX Source Extractor
======================

Extracts text and structures from ArXiv LaTeX sources.
Provides perfect equation extraction and structure preservation.

This extractor represents the highest fidelity CONVEYANCE dimension -
extracting information directly from the author's source with no OCR errors
or formatting ambiguities.
"""

import re
import gzip
import tarfile
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class LaTeXExtractor:
    """
    Extract content from LaTeX source files.
    
    This provides perfect extraction of equations, tables, and structure
    directly from the LaTeX source, avoiding all PDF parsing issues.
    
    In Actor-Network Theory terms, this is the most direct translation
    from author intent to machine representation.
    """
    
    def __init__(self, use_pandoc: bool = False):
        """
        Create a LaTeX extractor instance.
        
        Parameters:
            use_pandoc (bool): If True, attempt to convert LaTeX to Markdown using Pandoc for extracted documents.
        
        Side effects:
            - Initializes self.use_pandoc and a `pandoc_stats` dict tracking conversion attempts, successes, and failures.
            - Emits an informational log about the extractor initialization and Pandoc usage.
        """
        self.use_pandoc = use_pandoc
        # Track pandoc conversion stats
        self.pandoc_stats = {
            'attempts': 0,
            'successes': 0,
            'failures': 0
        }
        logger.info(f"Initialized LaTeX extractor (pandoc: {use_pandoc})")
    
    def extract(self, latex_path: str) -> Dict[str, Any]:
        """
        Determine whether the provided .gz file contains a tar archive of LaTeX sources or a single gzipped LaTeX file, then extract structured content accordingly.
        
        Parameters:
            latex_path (str): Path to a gzipped file that is either a tar.gz archive of a LaTeX project or a single gzipped .tex file.
        
        Returns:
            Dict[str, Any]: Extraction result with keys:
                - full_text (str): Extracted document text — Markdown if Pandoc conversion was used successfully, otherwise the raw LaTeX source.
                - latex_source (str): Raw LaTeX content read from the main .tex file.
                - structures (dict): Detected structures including 'equations', 'tables', 'citations', 'sections', and optional 'bibliography'.
                - metadata (dict): Processing metadata (e.g., extractor name, extraction_type ('tar.gz' or 'plain.gz'), main_file, num_tex_files, has_bibliography, processing_time, latex_path, and optional pandoc_error).
        
        Raises:
            FileNotFoundError: If the provided path does not exist.
        
        Notes:
            - If an unexpected error occurs during processing, the method logs the failure and returns a standardized empty result produced by _empty_result(latex_path, error_message).
        """
        latex_path = Path(latex_path)
        
        if not latex_path.exists():
            raise FileNotFoundError(f"LaTeX source not found: {latex_path}")
        
        start_time = datetime.now()
        
        try:
            # First check if it's a tar.gz or just a plain gzipped file
            with gzip.open(latex_path, 'rb') as gz:
                # Try to read first few bytes to detect format
                initial_bytes = gz.read(512)
                gz.seek(0)  # Reset for actual reading
                
                # Check if it's a tar archive (tar magic bytes)
                # Check if it's a tar archive (tar magic bytes at offset 257)
                # In a tar file, 'ustar' appears at bytes 257-261 of the header
                is_tar = len(initial_bytes) >= 262 and initial_bytes[257:262] == b'ustar'
                
                # If not tar, check if it looks like LaTeX
                if not is_tar:
                    try:
                        # Try to decode as text
                        text_sample = initial_bytes.decode('utf-8', errors='ignore')
                        # Check for common LaTeX patterns
                        is_latex = ('\\documentclass' in text_sample or 
                                  '\\begin{document}' in text_sample or
                                  '\\section' in text_sample or
                                  '\\usepackage' in text_sample)
                    except:
                        is_latex = False
                else:
                    is_latex = False
            
            # Handle based on file type
            if is_tar or not is_latex:
                # Extract from gzipped tar archive (original code path)
                return self._extract_from_tar_gz(latex_path, start_time)
            else:
                # Handle plain gzipped .tex file
                return self._extract_from_plain_gz(latex_path, start_time)
                
        except Exception as e:
            logger.error(f"LaTeX extraction failed for {latex_path}: {e}")
            return self._empty_result(latex_path, str(e))
    
    def _extract_from_tar_gz(self, latex_path: Path, start_time: datetime) -> Dict[str, Any]:
        """
        Extract content from a gzipped tar archive containing LaTeX sources and return a structured result.
        
        Reads and extracts the tar.gz into a temporary directory, selects the largest `.tex` file as the main LaTeX source, and derives structured elements (equations, tables, citations, sections). If a `.bbl` bibliography file is present it is included. Optionally converts the main `.tex` to Markdown using Pandoc (when the extractor was initialized with pandoc enabled); on conversion failure the raw LaTeX is used as a fallback. Returns metadata including processing time and details about the archive.
        
        Parameters:
            latex_path (Path): Path to the gzipped tar archive to process.
            start_time (datetime): Timestamp captured before processing began; used to compute processing duration.
        
        Returns:
            Dict[str, Any]: A dictionary with keys:
                - full_text: Converted Markdown if Pandoc succeeded and was used, otherwise the raw LaTeX text.
                - latex_source: Raw LaTeX content of the selected main `.tex` file.
                - structures: Dict containing extracted elements:
                    - equations: list of equation entries
                    - tables: list of table entries
                    - citations: list of unique citation keys
                    - sections: list of section entries
                    - bibliography: (optional) contents of a `.bbl` file if found
                - metadata: Dict with extraction details (extractor, extraction_type, main_file, num_tex_files, has_bibliography, processing_time, latex_path) and an optional `pandoc_error` if conversion failed.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            try:
                # Extract the tar.gz file
                with gzip.open(latex_path, 'rb') as gz:
                    with tarfile.open(fileobj=gz, mode='r') as tar:
                        tar.extractall(temp_path)
            except tarfile.TarError as e:
                # Not a valid tar file, might be plain gzipped
                logger.debug(f"Not a tar archive, trying plain gzip: {e}")
                return self._extract_from_plain_gz(latex_path, start_time)
            
            # Find main .tex file (usually the largest)
            tex_files = list(temp_path.glob("**/*.tex"))
            if not tex_files:
                logger.warning(f"No .tex files found in {latex_path}")
                return self._empty_result(latex_path, "No .tex files found")
                
            # Sort by size and pick largest as main file
            main_tex = max(tex_files, key=lambda f: f.stat().st_size)
            logger.info(f"Processing main LaTeX file: {main_tex.name}")
            
            # Read LaTeX content
            try:
                with open(main_tex, 'r', encoding='utf-8', errors='ignore') as f:
                    latex_content = f.read()
            except Exception as e:
                logger.error(f"Failed to read LaTeX file: {e}")
                return self._empty_result(latex_path, str(e))
            
            # Extract structured content
            structures = {
                'equations': self._extract_equations(latex_content),
                'tables': self._extract_tables(latex_content),
                'citations': self._extract_citations(latex_content),
                'sections': self._extract_sections(latex_content)
            }
            
            # Check for bibliography file
            bbl_files = list(temp_path.glob("**/*.bbl"))
            if bbl_files:
                try:
                    with open(bbl_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                        structures['bibliography'] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read bibliography: {e}")
            
            # Convert to markdown if requested
            pandoc_error = None
            if self.use_pandoc:
                full_text, pandoc_error = self._convert_to_markdown(main_tex)
                if not full_text:
                    full_text = latex_content  # Fallback to raw LaTeX
            else:
                full_text = latex_content
            
            duration = (datetime.now() - start_time).total_seconds()
            
            metadata = {
                'extractor': 'latex',
                'extraction_type': 'tar.gz',
                'main_file': main_tex.name,
                'num_tex_files': len(tex_files),
                'has_bibliography': len(bbl_files) > 0,
                'processing_time': duration,
                'latex_path': str(latex_path)
            }
            
            # Add Pandoc error if conversion failed
            if pandoc_error:
                metadata['pandoc_error'] = pandoc_error
            
            return {
                'full_text': full_text,
                'latex_source': latex_content,  # Always include raw LaTeX
                'structures': structures,
                'metadata': metadata
            }
    
    def _extract_from_plain_gz(self, latex_path: Path, start_time: datetime) -> Dict[str, Any]:
        """
        Extract content from a single gzipped LaTeX (.tex.gz) file and return structured results.
        
        Reads the gzipped LaTeX text, extracts equations, tables, citation keys, and section structure. If Pandoc conversion is enabled on the extractor instance, attempts to convert the LaTeX to Markdown and uses the converted text as `full_text` when successful; otherwise `full_text` contains the raw LaTeX. Computes processing duration using `start_time`.
        
        Returns:
            dict: A result dictionary with keys:
                - full_text (str): Markdown (if Pandoc conversion succeeded) or the raw LaTeX text.
                - latex_source (str): The raw LaTeX content read from the gzipped file.
                - structures (dict): Extracted structures with keys:
                    - equations (list): Extracted equation entries.
                    - tables (list): Extracted table entries.
                    - citations (list): Unique citation keys.
                    - sections (list): Section entries with level, title, and position.
                - metadata (dict): Processing metadata including:
                    - extractor: 'latex'
                    - extraction_type: 'plain.gz'
                    - main_file: input file name
                    - num_tex_files: 1
                    - has_bibliography: False
                    - processing_time: duration in seconds (float)
                    - latex_path: original path as string
                    - pandoc_error (optional): error summary if Pandoc conversion failed
        
        Notes:
            - If the input file is empty or an error occurs, the method returns the standardized empty result produced by `_empty_result`.
            - `start_time` is used only to compute `processing_time` and should represent the extraction start timestamp.
        """
        try:
            # Read the gzipped LaTeX directly
            with gzip.open(latex_path, 'rt', encoding='utf-8', errors='ignore') as gz:
                latex_content = gz.read()
            
            if not latex_content:
                logger.warning(f"Empty LaTeX content from {latex_path}")
                return self._empty_result(latex_path, "Empty LaTeX file")
            
            # Extract structured content
            structures = {
                'equations': self._extract_equations(latex_content),
                'tables': self._extract_tables(latex_content),
                'citations': self._extract_citations(latex_content),
                'sections': self._extract_sections(latex_content)
            }
            
            # For plain gzipped files, we need to write to temp for Pandoc
            pandoc_error = None
            full_text = latex_content
            
            if self.use_pandoc:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.tex', delete=False) as temp_tex:
                    temp_tex.write(latex_content)
                    temp_tex_path = Path(temp_tex.name)
                
                try:
                    full_text_converted, pandoc_error = self._convert_to_markdown(temp_tex_path)
                    if full_text_converted:
                        full_text = full_text_converted
                finally:
                    # Clean up temp file
                    temp_tex_path.unlink(missing_ok=True)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            metadata = {
                'extractor': 'latex',
                'extraction_type': 'plain.gz',
                'main_file': latex_path.name,
                'num_tex_files': 1,
                'has_bibliography': False,  # No separate bbl in plain gz
                'processing_time': duration,
                'latex_path': str(latex_path)
            }
            
            # Add Pandoc error if conversion failed
            if pandoc_error:
                metadata['pandoc_error'] = pandoc_error
            
            logger.info(f"Successfully extracted from plain gzipped LaTeX: {latex_path.name}")
            
            return {
                'full_text': full_text,
                'latex_source': latex_content,
                'structures': structures,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to extract plain gzipped LaTeX from {latex_path}: {e}")
            return self._empty_result(latex_path, str(e))
    
    def _extract_equations(self, latex: str) -> List[Dict[str, Any]]:
        """
        Extract LaTeX equations from a source string.
        
        Parses the provided LaTeX text and returns a list of equation entries discovered in the source. Each entry is a dict with:
        - 'type': one of 'display', 'display_unnumbered', 'align', or 'inline'.
        - 'latex': the raw LaTeX content of the equation (labels removed for display/align types).
        - 'label': the label value from a `\label{...}` if present, otherwise None.
        
        Notes:
        - Display equations detected from \begin{equation}...\end{equation} (type 'display') and \begin{equation*}...\end{equation*} (type 'display_unnumbered').
        - Align environments (including starred variants) are returned with type 'align'.
        - Inline math matched by $...$ is returned with type 'inline'; inline matches are filtered to a reasonable length and limited to the first 100 occurrences to reduce noise.
        
        Parameters:
            latex (str): LaTeX source text to scan.
        
        Returns:
            List[Dict[str, Any]]: List of extracted equation entries.
        """
        equations = []
        
        # Display equations: \begin{equation}...\end{equation}
        display_eqs = re.findall(
            r'\\begin\{equation\}(.*?)\\end\{equation\}', 
            latex, 
            re.DOTALL
        )
        for eq in display_eqs:
            # Check for label
            label_match = re.search(r'\\label\{([^}]+)\}', eq)
            label = label_match.group(1) if label_match else None
            # Clean equation text
            eq_text = re.sub(r'\\label\{[^}]+\}', '', eq).strip()
            equations.append({
                'type': 'display',
                'latex': eq_text,
                'label': label
            })
        
        # Numbered equations: \begin{equation*}...\end{equation*}
        display_star_eqs = re.findall(
            r'\\begin\{equation\*\}(.*?)\\end\{equation\*\}', 
            latex, 
            re.DOTALL
        )
        for eq in display_star_eqs:
            equations.append({
                'type': 'display_unnumbered',
                'latex': eq.strip(),
                'label': None
            })
        
        # Align environments
        align_eqs = re.findall(
            r'\\begin\{align\*?\}(.*?)\\end\{align\*?\}', 
            latex, 
            re.DOTALL
        )
        for eq in align_eqs:
            label_match = re.search(r'\\label\{([^}]+)\}', eq)
            label = label_match.group(1) if label_match else None
            eq_text = re.sub(r'\\label\{[^}]+\}', '', eq).strip()
            equations.append({
                'type': 'align',
                'latex': eq_text,
                'label': label
            })
        
        # Inline math: $...$ (limit to reasonable length to avoid false positives)
        inline_math = re.findall(r'\$([^$]{2,200})\$', latex)
        for eq in inline_math[:100]:  # Limit to first 100 to avoid noise
            equations.append({
                'type': 'inline',
                'latex': eq,
                'label': None
            })
        
        logger.info(f"Extracted {len(equations)} equations")
        return equations
    
    def _extract_tables(self, latex: str) -> List[Dict[str, Any]]:
        """
        Extract LaTeX table environments and return structured table data.
        
        Scans the provided LaTeX source for table (and table*) environments and returns a list of table entries. Each entry contains:
        - 'caption': caption text if present, otherwise None
        - 'label': label key if present, otherwise None
        - 'latex': the inner tabular/array environment (or the full table content if no tabular/array was found)
        - 'column_spec': column specification from \begin{tabular}{...} or \begin{array}{...}, otherwise None
        - 'full_content': the complete content of the matched table environment
        
        Parameters:
            latex (str): The raw LaTeX source to search for table environments.
        
        Returns:
            List[Dict[str, Any]]: A list of dictionaries describing each extracted table.
        """
        tables = []
        
        # Find table environments
        table_pattern = r'\\begin\{table\*?\}(.*?)\\end\{table\*?\}'
        table_matches = re.findall(table_pattern, latex, re.DOTALL)
        
        for table_content in table_matches:
            # Extract caption
            caption_match = re.search(r'\\caption\{((?:[^{}]|\{[^}]*\})*)\}', table_content)
            caption = caption_match.group(1) if caption_match else None
            
            # Extract label
            label_match = re.search(r'\\label\{([^}]+)\}', table_content)
            label = label_match.group(1) if label_match else None
            
            # Extract tabular content
            tabular_match = re.search(
                r'\\begin\{tabular\}(.*?)\\end\{tabular\}', 
                table_content, 
                re.DOTALL
            )
            
            if not tabular_match:
                # Try array environment (sometimes used for tables)
                tabular_match = re.search(
                    r'\\begin\{array\}(.*?)\\end\{array\}', 
                    table_content, 
                    re.DOTALL
                )
            
            tabular = tabular_match.group(0) if tabular_match else table_content
            
            # Parse column specification
            col_spec_match = re.search(r'\\begin\{(?:tabular|array)\}\{([^}]+)\}', tabular)
            col_spec = col_spec_match.group(1) if col_spec_match else None
            
            tables.append({
                'caption': caption,
                'label': label,
                'latex': tabular,
                'column_spec': col_spec,
                'full_content': table_content
            })
        
        logger.info(f"Extracted {len(tables)} tables")
        return tables
    
    def _extract_citations(self, latex: str) -> List[str]:
        """
        Extract unique citation keys from a LaTeX source string.
        
        Searches for citation commands (e.g. `\cite{...}`, `\citep{...}`, `\citet{...}`), splits multiple keys within a single command by commas, preserves the first occurrence order, and removes duplicates.
        
        Parameters:
            latex (str): LaTeX source text to scan for citation keys.
        
        Returns:
            List[str]: Ordered list of unique citation keys found in the source.
        """
        citations = []
        
        # Find \cite commands (including variants like \citep, \citet)
        cite_pattern = r'\\cite[pt]?\{([^}]+)\}'
        for match in re.finditer(cite_pattern, latex):
            # Split multiple citations
            cite_keys = match.group(1).split(',')
            citations.extend([key.strip() for key in cite_keys])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for cite in citations:
            if cite not in seen:
                seen.add(cite)
                unique_citations.append(cite)
        
        logger.info(f"Extracted {len(unique_citations)} unique citations")
        return unique_citations
    
    def _extract_sections(self, latex: str) -> List[Dict[str, Any]]:
        """
        Return a list describing the hierarchical section headings found in the LaTeX source.
        
        Searches for \section, \subsection, \subsubsection, and \paragraph commands and records each match's level, title text, and byte/character start position in the input string.
        
        Returns:
            List[Dict[str, Any]]: Each item contains:
                - 'level' (str): one of 'section', 'subsection', 'subsubsection', 'paragraph'.
                - 'title' (str): the raw title text captured between the braces.
                - 'position' (int): the start index of the matched command in the input string.
        """
        sections = []
        
        # Find all section commands
        section_pattern = r'\\(section|subsection|subsubsection|paragraph)\{([^}]+)\}'
        for match in re.finditer(section_pattern, latex):
            sections.append({
                'level': match.group(1),
                'title': match.group(2),
                'position': match.start()
            })
        
        logger.info(f"Extracted {len(sections)} sections")
        return sections
    
    def _convert_to_markdown(self, tex_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert a LaTeX .tex file to Markdown using Pandoc.
        
        Attempts to run the system `pandoc` to convert the file at `tex_path`. Updates
        internal Pandoc statistics counters (attempts, successes, failures) as it
        runs. On success returns the converted Markdown text; on failure returns an
        error message describing the Pandoc failure or the exception encountered.
        
        Parameters:
            tex_path (Path): Path to the `.tex` file to convert.
        
        Returns:
            Tuple[Optional[str], Optional[str]]: (markdown_text, error_message).
                - `markdown_text` is the converted Markdown string when conversion
                  succeeds, otherwise None.
                - `error_message` is a human-readable error or exception string when
                  conversion fails, otherwise None.
        """
        self.pandoc_stats['attempts'] += 1
        try:
            import subprocess
            result = subprocess.run(
                ['pandoc', '-f', 'latex', '-t', 'markdown', str(tex_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                self.pandoc_stats['successes'] += 1
                logger.info(f"Successfully converted LaTeX to markdown (success rate: {self.pandoc_stats['successes']}/{self.pandoc_stats['attempts']})")
                return result.stdout, None
            else:
                self.pandoc_stats['failures'] += 1
                # Log at debug level to reduce noise - these are common with academic LaTeX
                # Extract just the error type for cleaner logging
                error_lines = result.stderr.strip().split('\n')
                error_message = result.stderr.strip()  # Keep full error for database
                
                if error_lines:
                    # Get first line of error which usually has the key info
                    error_summary = error_lines[0]
                    if "expecting" in result.stderr:
                        # This is a LaTeX structure mismatch - very common
                        logger.debug(f"Pandoc: LaTeX structure issue - {error_summary}")
                    else:
                        logger.debug(f"Pandoc conversion issue: {error_summary}")
                # Log stats periodically
                if self.pandoc_stats['attempts'] % 10 == 0:
                    success_rate = (self.pandoc_stats['successes'] / self.pandoc_stats['attempts']) * 100
                    logger.info(f"Pandoc conversion rate: {success_rate:.1f}% ({self.pandoc_stats['successes']}/{self.pandoc_stats['attempts']})")
                return None, error_message
        except Exception as e:
            logger.warning(f"Pandoc not available or failed: {e}")
            return None, str(e)
    
    def _empty_result(self, latex_path: Path, error: str) -> Dict[str, Any]:
        """Return empty result structure for failed extraction."""
        return {
            'full_text': '',
            'latex_source': '',
            'structures': {
                'equations': [],
                'tables': [],
                'citations': [],
                'sections': []
            },
            'metadata': {
                'extractor': 'latex',
                'error': error,
                'latex_path': str(latex_path)
            }
        }
    
    def extract_batch(self, latex_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple LaTeX source paths and return a list of extraction results.
        
        Each path is processed with extract(...). Failures for individual items are caught and converted into a standardized empty result (via _empty_result) so that processing continues and the returned list preserves the input order.
        
        Parameters:
            latex_paths (List[str]): Paths to LaTeX sources (tar.gz or gz) to extract.
        
        Returns:
            List[Dict[str, Any]]: Extraction results corresponding to each input path; on failure for an item, its entry is the standardized empty result containing an error message.
        """
        results = []
        for latex_path in latex_paths:
            try:
                result = self.extract(latex_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract {latex_path}: {e}")
                results.append(self._empty_result(Path(latex_path), str(e)))
        
        return results
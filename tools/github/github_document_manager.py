#!/usr/bin/env python3
"""
GitHub Document Manager
=======================

Manages GitHub repository documents for processing.
Handles cloning, file extraction, and preparation for the generic processor.
"""

import os
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GitHubRepository:
    """GitHub repository information."""
    owner: str
    name: str
    clone_url: str
    default_branch: str = "main"
    
    @property
    def full_name(self):
        """
        Return the repository's full name in the form "owner/name".
        
        This is derived from the GitHubRepository's `owner` and `name` fields and is suitable for display or constructing GitHub URLs.
        
        Returns:
            str: The repository full name, e.g. "octocat/hello-world".
        """
        return f"{self.owner}/{self.name}"
    
    @property
    def sanitized_name(self):
        """
        Return a filesystem-safe repository identifier composed of the owner and repository name.
        
        This property joins the repository owner and name with an underscore (e.g., `owner_repo`) and is suitable for use as a directory or file-friendly identifier.
        """
        return f"{self.owner}_{self.name}"


class GitHubDocumentManager:
    """
    Manages GitHub repositories for document processing.
    
    Similar to ArXivDocumentManager but for GitHub repositories.
    Clones repos, extracts code files, and prepares them for processing.
    """
    
    def __init__(self, clone_dir: str = "/tmp/github_repos", cleanup: bool = True):
        """
        Initialize the GitHubDocumentManager.
        
        Creates (if needed) and stores the local clone directory, configures whether cloned repositories
        are removed after processing, and initializes the sets of file extensions to include and
        directory names to skip during repository traversal.
        
        Parameters:
            clone_dir (str): Path where repositories will be cloned. The directory will be created if it does not exist.
            cleanup (bool): If True, cloned repository directories are removed after processing.
        """
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup = cleanup
        
        # File extensions to process
        self.code_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
            '.jl', '.m', '.sh', '.yaml', '.yml', '.json', '.md'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', '__pycache__', 'node_modules', '.pytest_cache',
            'venv', 'env', '.env', 'dist', 'build', '.vscode', '.idea'
        }
        
        logger.info(f"Initialized GitHubDocumentManager with clone_dir: {clone_dir}")
    
    def prepare_repository(self, repo_url: str, branch: str = None) -> List:
        """
        Prepare a GitHub repository for downstream document processing and return the resulting DocumentTask objects.
        
        Parses the provided repository URL (supports full GitHub URLs or "owner/repo" shorthand), optionally uses the given branch, clones the repository locally, scans for eligible files to build DocumentTask entries, and (if configured) removes the cloned repository when finished.
        
        Parameters:
            repo_url (str): GitHub repository location (HTTPS/git URL or "owner/repo").
            branch (str | None): Optional branch to use instead of the repository's default.
        
        Returns:
            List: A list of DocumentTask objects representing files prepared for processing. Returns an empty list if cloning fails.
        """
        # Parse repository info from URL
        repo = self._parse_repo_url(repo_url)
        if branch:
            repo.default_branch = branch
        
        logger.info(f"Preparing repository: {repo.full_name}")
        
        # Clone repository
        repo_path = self._clone_repository(repo)
        if not repo_path:
            logger.error(f"Failed to clone {repo.full_name}")
            return []
        
        # Extract code files
        tasks = self._extract_code_files(repo, repo_path)
        
        logger.info(f"Prepared {len(tasks)} files from {repo.full_name}")
        
        # Cleanup if requested
        if self.cleanup and repo_path.exists():
            shutil.rmtree(repo_path)
            logger.debug(f"Cleaned up {repo_path}")
        
        return tasks
    
    def prepare_repositories_from_list(self, repo_urls: List[str]) -> List:
        """
        Prepare multiple GitHub repositories and collect DocumentTask objects.
        
        Given a list of repository URLs (or "owner/repo" strings), calls prepare_repository for each entry, logs failures per repository and continues, and returns a combined list of DocumentTask objects created from successfully processed repositories.
        
        Parameters:
            repo_urls (List[str]): Repository URLs or "owner/repo" identifiers to process.
        
        Returns:
            List: Combined list of DocumentTask objects for all successfully prepared repositories.
        """
        all_tasks = []
        for repo_url in repo_urls:
            try:
                tasks = self.prepare_repository(repo_url)
                all_tasks.extend(tasks)
            except Exception as e:
                logger.error(f"Failed to process {repo_url}: {e}")
        
        return all_tasks
    
    def parse_repo_url(self, url: str) -> GitHubRepository:
        """Parse repository information from URL (public API)."""
        return self._parse_repo_url(url)
    
    def _parse_repo_url(self, url: str) -> GitHubRepository:
        """
        Parse a GitHub repository identifier or URL and return a populated GitHubRepository.
        
        Accepts full GitHub URLs (e.g., "https://github.com/owner/repo", "git@github.com:owner/repo") or the shorthand "owner/repo".
        Trailing ".git" and leading/trailing slashes/colons are removed. Constructs an HTTPS clone URL in the form "https://github.com/{owner}/{name}.git".
        
        Parameters:
            url (str): Repository URL or "owner/repo" identifier.
        
        Returns:
            GitHubRepository: Dataclass with owner, name, clone_url, and default_branch.
        """
        # Handle different URL formats
        url = url.strip()
        
        # Extract owner/repo from various formats
        if "github.com" in url:
            # https://github.com/owner/repo or git@github.com:owner/repo
            parts = url.split("github.com")[-1]
            parts = parts.strip(":/").replace(".git", "")
            owner, name = parts.split("/")
        else:
            # Assume owner/repo format
            owner, name = url.split("/")
        
        # Construct clone URL
        clone_url = f"https://github.com/{owner}/{name}.git"
        
        return GitHubRepository(
            owner=owner,
            name=name,
            clone_url=clone_url
        )
    
    def _clone_repository(self, repo: GitHubRepository) -> Optional[Path]:
        """
        Clone the given GitHubRepository into the manager's clone directory and return the local Path, or None on failure.
        
        The method removes any existing target directory for the repository, attempts a shallow clone (--depth 1) of the repository using repo.default_branch, and falls back to a shallow clone without an explicit branch if the first attempt fails. Cloning is subject to a 5-minute timeout. On any cloning error (including timeout) the function logs the failure and returns None.
        
        Returns:
            Optional[Path]: Path to the cloned repository on success, or None if cloning failed.
        """
        repo_path = self.clone_dir / repo.sanitized_name
        
        # Remove if exists
        if repo_path.exists():
            shutil.rmtree(repo_path)
        
        try:
            # Clone with depth 1 for efficiency
            cmd = [
                "git", "clone",
                "--depth", "1",
                "--branch", repo.default_branch,
                repo.clone_url,
                str(repo_path)
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                # Try without branch specification
                cmd = [
                    "git", "clone",
                    "--depth", "1",
                    repo.clone_url,
                    str(repo_path)
                ]
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode != 0:
                    logger.error(f"Clone failed: {result.stderr}")
                    return None
            
            return repo_path
            
        except subprocess.TimeoutExpired:
            logger.error(f"Clone timeout for {repo.full_name}")
            return None
        except Exception as e:
            logger.error(f"Clone error: {e}")
            return None
    
    def _extract_code_files(self, repo: GitHubRepository, repo_path: Path) -> List:
        """
        Collects eligible code and documentation files from a cloned repository and returns a list of DocumentTask objects ready for downstream processing.
        
        This function recursively walks repo_path, filtering out directories listed in self.skip_dirs, files whose suffix is not in self.code_extensions, and files larger than 10 MB. For each accepted file it constructs a DocumentTask (imported dynamically to avoid circular imports) with:
        - document_id: "{repo.sanitized_name}/{relative_path}"
        - pdf_path: string file path to the file
        - metadata: dict containing repository (owner/name), owner, name, file_path (relative), file_extension, file_size, and source set to "github".
        
        Parameters:
            repo (GitHubRepository): Repository metadata used to build document_id and metadata fields.
            repo_path (Path): Local filesystem path to the root of the cloned repository.
        
        Returns:
            List[DocumentTask]: A list of DocumentTask instances for all files that passed filtering.
        """
        # Import here to avoid circular dependency
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from core.processors.generic_document_processor import DocumentTask
        
        tasks = []
        
        # Walk through repository
        for file_path in repo_path.rglob("*"):
            # Skip directories
            if file_path.is_dir():
                continue
            
            # Skip if in ignored directory
            if any(skip_dir in file_path.parts for skip_dir in self.skip_dirs):
                continue
            
            # Check file extension
            if file_path.suffix.lower() not in self.code_extensions:
                continue
            
            # Skip large files (>10MB)
            if file_path.stat().st_size > 10 * 1024 * 1024:
                logger.warning(f"Skipping large file: {file_path}")
                continue
            
            # Create relative path for document ID
            rel_path = file_path.relative_to(repo_path)
            document_id = f"{repo.sanitized_name}/{rel_path}"
            
            # Create DocumentTask
            task = DocumentTask(
                document_id=document_id,
                pdf_path=str(file_path),  # Using pdf_path for file path
                metadata={
                    'repository': repo.full_name,
                    'owner': repo.owner,
                    'name': repo.name,
                    'file_path': str(rel_path),
                    'file_extension': file_path.suffix,
                    'file_size': file_path.stat().st_size,
                    'source': 'github'
                }
            )
            tasks.append(task)
        
        return tasks
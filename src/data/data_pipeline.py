"""
Data pipeline for loading and preprocessing various file formats.

Supports:
- Text files (.txt, .md)
- Code files (.py, .js, .java, .cpp, etc.)
- PDF files (.pdf)
- JSON files (.json)
- CSV files (.csv)
- Web pages (HTML)
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Iterator, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """A document with text content and metadata."""

    content: str
    source: str
    domain: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def __len__(self) -> int:
        return len(self.content)


class TextLoader:
    """Load plain text files."""

    SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".text"}

    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in TextLoader.SUPPORTED_EXTENSIONS

    @staticmethod
    def load(file_path: Path, encoding: str = "utf-8") -> Document:
        """Load a text file."""
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()

            return Document(
                content=content,
                source=str(file_path),
                domain="text",
                metadata={"encoding": encoding, "size": len(content)},
            )
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise


class CodeLoader:
    """Load code files."""

    SUPPORTED_EXTENSIONS = {
        ".py",
        ".js",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".ts",
        ".jsx",
        ".tsx",
        ".scala",
        ".r",
        ".m",
        ".sh",
    }

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".kt": "kotlin",
    }

    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in CodeLoader.SUPPORTED_EXTENSIONS

    @staticmethod
    def load(file_path: Path, encoding: str = "utf-8") -> Document:
        """Load a code file."""
        try:
            with open(file_path, "r", encoding=encoding, errors="replace") as f:
                content = f.read()

            language = CodeLoader.LANGUAGE_MAP.get(file_path.suffix.lower(), "unknown")

            return Document(
                content=content,
                source=str(file_path),
                domain="code",
                metadata={
                    "language": language,
                    "extension": file_path.suffix,
                    "filename": file_path.name,
                    "size": len(content),
                },
            )
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise


class PDFLoader:
    """Load PDF files."""

    SUPPORTED_EXTENSIONS = {".pdf"}

    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in PDFLoader.SUPPORTED_EXTENSIONS

    @staticmethod
    def load(file_path: Path) -> Document:
        """
        Load a PDF file.

        Note: Requires PyPDF2 or pdfplumber to be installed.
        Falls back to simple extraction if libraries not available.
        """
        try:
            # Try pdfplumber first (better extraction)
            try:
                import pdfplumber

                text_parts = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

                content = "\n\n".join(text_parts)

            except ImportError:
                # Fallback to PyPDF2
                try:
                    from PyPDF2 import PdfReader

                    reader = PdfReader(file_path)
                    text_parts = []

                    for page in reader.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)

                    content = "\n\n".join(text_parts)

                except ImportError:
                    raise ImportError(
                        "PDF support requires 'pdfplumber' or 'PyPDF2'. "
                        "Install with: pip install pdfplumber"
                    )

            return Document(
                content=content,
                source=str(file_path),
                domain="document",
                metadata={"type": "pdf", "filename": file_path.name, "size": len(content)},
            )

        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            raise


class JSONLoader:
    """Load JSON files."""

    SUPPORTED_EXTENSIONS = {".json", ".jsonl"}

    @staticmethod
    def can_load(file_path: Path) -> bool:
        """Check if this loader can handle the file."""
        return file_path.suffix.lower() in JSONLoader.SUPPORTED_EXTENSIONS

    @staticmethod
    def load(file_path: Path, text_field: str = "text") -> List[Document]:
        """
        Load JSON file(s).

        Args:
            file_path: Path to JSON file
            text_field: Field name containing text content

        Returns:
            List of Document objects (one per JSON object)
        """
        try:
            documents = []

            with open(file_path, "r", encoding="utf-8") as f:
                if file_path.suffix == ".jsonl":
                    # JSON Lines format
                    for line_num, line in enumerate(f, 1):
                        if line.strip():
                            data = json.loads(line)
                            content = data.get(text_field, json.dumps(data))

                            documents.append(
                                Document(
                                    content=content,
                                    source=f"{file_path}:line_{line_num}",
                                    domain="json",
                                    metadata=data,
                                )
                            )
                else:
                    # Regular JSON
                    data = json.load(f)

                    if isinstance(data, list):
                        # List of objects
                        for idx, item in enumerate(data):
                            if isinstance(item, dict):
                                content = item.get(text_field, json.dumps(item))
                            else:
                                content = str(item)

                            documents.append(
                                Document(
                                    content=content,
                                    source=f"{file_path}:item_{idx}",
                                    domain="json",
                                    metadata=item if isinstance(item, dict) else {},
                                )
                            )
                    elif isinstance(data, dict):
                        # Single object
                        content = data.get(text_field, json.dumps(data))
                        documents.append(
                            Document(
                                content=content, source=str(file_path), domain="json", metadata=data
                            )
                        )
                    else:
                        # Primitive value
                        documents.append(
                            Document(
                                content=str(data), source=str(file_path), domain="json", metadata={}
                            )
                        )

            return documents

        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {e}")
            raise


class DataPipeline:
    """
    Main data pipeline for loading and preprocessing documents.

    Example:
        >>> pipeline = DataPipeline()
        >>> docs = pipeline.load_directory("data/documents")
        >>> for doc in docs:
        ...     print(f"{doc.source}: {len(doc)} chars")
    """

    def __init__(self, chunk_size: Optional[int] = None, min_length: int = 10):
        """
        Initialize data pipeline.

        Args:
            chunk_size: Optional size to chunk long documents
            min_length: Minimum document length to keep
        """
        self.chunk_size = chunk_size
        self.min_length = min_length

        self.loaders = [TextLoader, CodeLoader, PDFLoader, JSONLoader]

    def load_file(self, file_path: Path) -> List[Document]:
        """
        Load a single file.

        Args:
            file_path: Path to file

        Returns:
            List of Document objects (may be multiple for JSON)
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Find appropriate loader
        for loader in self.loaders:
            if loader.can_load(file_path):
                result = loader.load(file_path)

                # JSONLoader returns list, others return single Document
                if isinstance(result, list):
                    documents = result
                else:
                    documents = [result]

                # Apply preprocessing
                processed = []
                for doc in documents:
                    # Filter by length
                    if len(doc.content) < self.min_length:
                        continue

                    # Chunk if needed
                    if self.chunk_size and len(doc.content) > self.chunk_size:
                        chunks = self._chunk_document(doc)
                        processed.extend(chunks)
                    else:
                        processed.append(doc)

                return processed

        raise ValueError(f"No loader found for file type: {file_path.suffix}")

    def load_directory(
        self, dir_path: Path, recursive: bool = True, pattern: str = "*"
    ) -> Iterator[Document]:
        """
        Load all supported files from a directory.

        Args:
            dir_path: Path to directory
            recursive: Whether to search recursively
            pattern: Glob pattern for files

        Yields:
            Document objects
        """
        dir_path = Path(dir_path)

        if not dir_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")

        # Get all files
        if recursive:
            files = dir_path.rglob(pattern)
        else:
            files = dir_path.glob(pattern)

        # Load each file
        for file_path in files:
            if file_path.is_file():
                try:
                    documents = self.load_file(file_path)
                    for doc in documents:
                        yield doc
                except Exception as e:
                    logger.warning(f"Skipping {file_path}: {e}")
                    continue

    def load_files(self, file_paths: List[Path]) -> Iterator[Document]:
        """
        Load multiple files.

        Args:
            file_paths: List of file paths

        Yields:
            Document objects
        """
        for file_path in file_paths:
            try:
                documents = self.load_file(file_path)
                for doc in documents:
                    yield doc
            except Exception as e:
                logger.warning(f"Skipping {file_path}: {e}")
                continue

    def _chunk_document(self, doc: Document) -> List[Document]:
        """
        Chunk a document into smaller pieces.

        Args:
            doc: Document to chunk

        Returns:
            List of chunked documents
        """
        if not self.chunk_size:
            return [doc]

        content = doc.content
        chunks = []

        # Simple character-based chunking with overlap
        overlap = self.chunk_size // 4  # 25% overlap
        start = 0

        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]

            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings
                last_period = chunk_text.rfind(". ")
                last_newline = chunk_text.rfind("\n")
                break_point = max(last_period, last_newline)

                if break_point > self.chunk_size // 2:  # Only break if not too early
                    chunk_text = chunk_text[: break_point + 1]
                    end = start + break_point + 1

            if len(chunk_text) >= self.min_length:
                chunks.append(
                    Document(
                        content=chunk_text.strip(),
                        source=f"{doc.source}:chunk_{len(chunks)}",
                        domain=doc.domain,
                        metadata={**doc.metadata, "is_chunk": True, "chunk_index": len(chunks)},
                    )
                )

            start = end - overlap

        return chunks if chunks else [doc]


class DataPreprocessor:
    """Preprocess document text for training."""

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean text by removing excessive whitespace and special characters.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)

        # Remove very long runs of special characters
        text = re.sub(r"([^\w\s])\1{5,}", r"\1\1", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata

        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def filter_quality(doc: Document, min_alpha_ratio: float = 0.5) -> bool:
        """
        Filter document by quality heuristics.

        Args:
            doc: Document to check
            min_alpha_ratio: Minimum ratio of alphabetic characters

        Returns:
            True if document passes quality checks
        """
        content = doc.content

        if not content:
            return False

        # Check alpha ratio
        alpha_chars = sum(c.isalpha() for c in content)
        alpha_ratio = alpha_chars / len(content)

        if alpha_ratio < min_alpha_ratio:
            return False

        # Check for excessive repetition
        words = content.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.3:  # More than 70% repeated words
                return False

        return True


# Convenience function
def load_documents(
    paths: List[Path],
    chunk_size: Optional[int] = None,
    clean: bool = True,
    filter_quality: bool = True,
) -> List[Document]:
    """
    Convenience function to load documents from multiple paths.

    Args:
        paths: List of file or directory paths
        chunk_size: Optional size to chunk long documents
        clean: Whether to clean text
        filter_quality: Whether to filter by quality

    Returns:
        List of Document objects
    """
    pipeline = DataPipeline(chunk_size=chunk_size)
    preprocessor = DataPreprocessor()

    documents = []

    for path in paths:
        path = Path(path)

        if path.is_file():
            docs = pipeline.load_file(path)
        elif path.is_dir():
            docs = list(pipeline.load_directory(path))
        else:
            logger.warning(f"Skipping invalid path: {path}")
            continue

        for doc in docs:
            # Clean if requested
            if clean:
                doc.content = preprocessor.clean_text(doc.content)
                doc.content = preprocessor.normalize_unicode(doc.content)

            # Filter if requested
            if filter_quality and not preprocessor.filter_quality(doc):
                continue

            documents.append(doc)

    return documents


if __name__ == "__main__":
    # Example usage
    print("✓ Data pipeline module ready!")
    print("  - TextLoader for .txt, .md files")
    print("  - CodeLoader for .py, .js, etc.")
    print("  - PDFLoader for .pdf files (requires pdfplumber)")
    print("  - JSONLoader for .json, .jsonl files")
    print("  - DataPipeline for orchestration")
    print("  - DataPreprocessor for cleaning")

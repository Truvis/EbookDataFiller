#!/usr/bin/env python3
"""
Ebook Metadata Pipeline
=======================
Comprehensive metadata extraction, enrichment, writing, and renaming for ebook collections.

Pipeline Order:
  1. Gutenberg RDF Catalog (fastest, most reliable for PG books)
  2. Embedded metadata extraction from file
  3. Public APIs (Open Library, Google Books)
  4. AI text analysis via Claude API (last resort)

Supported Formats: .epub, .pdf, .mobi, .azw, .azw3, .fb2, .txt, .html, .htm, .djvu, .cbz, .cbr, .lit

Requirements:
  pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4 --break-system-packages
  Also: Calibre CLI tools (ebook-meta) for metadata writing

Usage:
  python ebook_metadata_pipeline.py /path/to/ebooks [--rdf-catalog /path/to/cache/rdf-files] [options]

  Run with --help for full options.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from xml.etree import ElementTree as ET

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

SUPPORTED_EXTENSIONS = {
    '.epub', '.pdf', '.mobi', '.azw', '.azw3', '.fb2',
    '.txt', '.htm', '.html', '.djvu', '.cbz', '.cbr', '.lit',
    '.doc', '.docx', '.rtf', '.odt',
}

# Gutenberg RDF namespaces
RDF_NS = {
    'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
    'dcterms': 'http://purl.org/dc/terms/',
    'pgterms': 'http://www.gutenberg.org/2009/pgterms/',
    'dcam': 'http://purl.org/dc/dcam/',
    'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
    'cc': 'http://web.resource.org/cc/',
    'marcrel': 'http://id.loc.gov/vocabulary/relators/',
}

# Open Library API
OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPENLIBRARY_BOOK_URL = "https://openlibrary.org/api/books"

# Google Books API
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

# Max pages/chars to extract for AI analysis
AI_EXTRACT_MAX_CHARS = 15000

# Rate limiting
API_DELAY_SECONDS = 1.0

# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class BookMetadata:
    """Comprehensive book metadata container."""
    # Core identifiers
    title: Optional[str] = None
    subtitle: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    editors: List[str] = field(default_factory=list)
    translators: List[str] = field(default_factory=list)
    illustrators: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)

    # Publication info
    publisher: Optional[str] = None
    publication_date: Optional[str] = None  # ISO date or year
    original_publication_date: Optional[str] = None
    edition: Optional[str] = None
    revision: Optional[str] = None

    # Identifiers
    isbn_10: Optional[str] = None
    isbn_13: Optional[str] = None
    gutenberg_id: Optional[str] = None
    oclc: Optional[str] = None
    lccn: Optional[str] = None
    openlibrary_id: Optional[str] = None
    google_books_id: Optional[str] = None
    asin: Optional[str] = None
    doi: Optional[str] = None

    # Classification
    subjects: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    lcc: Optional[str] = None  # Library of Congress Classification
    ddc: Optional[str] = None  # Dewey Decimal Classification
    bisac: List[str] = field(default_factory=list)

    # Series info
    series: Optional[str] = None
    series_index: Optional[float] = None

    # Content info
    language: Optional[str] = None
    description: Optional[str] = None
    long_description: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None

    # Rights
    rights: Optional[str] = None
    copyright: Optional[str] = None
    license_url: Optional[str] = None

    # Cover
    cover_url: Optional[str] = None

    # Source tracking
    source_file: Optional[str] = None
    source_format: Optional[str] = None
    file_hash: Optional[str] = None
    sources_used: List[str] = field(default_factory=list)  # Track where data came from

    # Processing info
    confidence_score: float = 0.0
    processing_notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def merge(self, other: 'BookMetadata', source_name: str = "unknown"):
        """Merge another metadata object, filling in gaps without overwriting."""
        merged_fields = []
        for fld in self.__dataclass_fields__:
            if fld in ('sources_used', 'processing_notes', 'errors', 'confidence_score',
                       'source_file', 'source_format', 'file_hash'):
                continue

            current = getattr(self, fld)
            incoming = getattr(other, fld)

            if incoming is None or incoming == [] or incoming == 0:
                continue

            if current is None or current == [] or current == 0:
                setattr(self, fld, incoming)
                merged_fields.append(fld)
            elif isinstance(current, list) and isinstance(incoming, list):
                # Merge lists, dedup
                combined = list(current)
                for item in incoming:
                    if item not in combined:
                        combined.append(item)
                if len(combined) > len(current):
                    setattr(self, fld, combined)
                    merged_fields.append(fld)

        if merged_fields:
            self.sources_used.append(source_name)
            self.processing_notes.append(
                f"Merged from {source_name}: {', '.join(merged_fields)}"
            )

    def completeness_score(self) -> float:
        """Calculate how complete the metadata is (0.0 - 1.0)."""
        weights = {
            'title': 20, 'authors': 20, 'language': 5, 'publication_date': 10,
            'description': 10, 'subjects': 8, 'isbn_13': 5, 'isbn_10': 3,
            'publisher': 5, 'genres': 5, 'series': 3, 'page_count': 3,
            'cover_url': 3,
        }
        total_weight = sum(weights.values())
        score = 0
        for fld, weight in weights.items():
            val = getattr(self, fld, None)
            if val and val != [] and val != 0:
                score += weight
        return round(score / total_weight, 3)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if v is not None and v != [] and v != 0 and v != 0.0}


# =============================================================================
# LOGGING SETUP
# =============================================================================

class ColorFormatter(logging.Formatter):
    """Colored log output for terminal — no level prefix, just colors."""
    COLORS = {
        'DEBUG': '\033[90m',      # Gray (dim)
        'INFO': '\033[0m',        # Default (white)
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m',   # Red bg
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{record.getMessage()}{self.RESET}"


# Custom TRACE level (below DEBUG) — only goes to file log, never console
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class BufferedLogHandler(logging.Handler):
    """Thread-safe handler that captures log messages for a specific thread."""

    def __init__(self, min_level=logging.INFO, thread_id=None):
        super().__init__(min_level)
        self.buffer = []
        self.thread_id = thread_id or threading.get_ident()
        self.setFormatter(ColorFormatter())

    def emit(self, record):
        # Only capture messages from our thread
        if threading.get_ident() != self.thread_id:
            return
        try:
            self.buffer.append(self.format(record))
        except Exception:
            self.handleError(record)

    def get_output(self) -> str:
        return '\n'.join(self.buffer)

    def clear(self):
        self.buffer.clear()


def setup_logging(log_dir: str, verbose: bool = False) -> logging.Logger:
    logger = logging.getLogger('ebook_pipeline')
    logger.setLevel(TRACE)  # Capture everything

    # File handler - everything including TRACE
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fh = logging.FileHandler(os.path.join(log_dir, f'pipeline_{timestamp}.log'))
    fh.setLevel(TRACE)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    logger.addHandler(fh)

    # Error-only log
    eh = logging.FileHandler(os.path.join(log_dir, f'errors_{timestamp}.log'))
    eh.setLevel(logging.ERROR)
    eh.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    ))
    logger.addHandler(eh)

    # Console handler — colors only, no prefixes
    # verbose = DEBUG, normal = INFO (TRACE never shows on console)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    # Stats log (JSON-lines for easy parsing)
    sh = logging.FileHandler(os.path.join(log_dir, f'stats_{timestamp}.jsonl'))
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(message)s'))
    stats_logger = logging.getLogger('ebook_pipeline.stats')
    stats_logger.addHandler(sh)
    stats_logger.propagate = False  # Don't show JSON on console

    return logger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Create a safe filename from a string."""
    if not name:
        return "Unknown"
    # Normalize unicode
    name = unicodedata.normalize('NFKD', name)
    # Remove/replace problematic characters
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'[\s]+', ' ', name).strip()
    name = name.strip('. ')
    if len(name) > max_length:
        name = name[:max_length].rsplit(' ', 1)[0].strip()
    return name or "Unknown"


def clean_title(title: str) -> str:
    """Clean artifacts from a book title string."""
    if not title:
        return title
    # Replace underscore-as-colon (calibre artifact): "Title_ Subtitle" → "Title: Subtitle"
    title = re.sub(r'_\s+', ': ', title)
    title = re.sub(r'\s+_', ': ', title)
    # Remove release group tags: [Team-IRA], [WWRG], etc.
    title = re.sub(r'\s*\[(?:Team[- ]\w+|WWRG|scan|OCR|ebook|eBook|converted)\]\s*', '', title, flags=re.IGNORECASE)
    # Remove quality markers: (True PDF), (True EPUB), (retail), (scan)
    title = re.sub(r'\s*\((?:True\s+)?(?:PDF|EPUB|AZW3?|MOBI|retail|scan|OCR)\)\s*', '', title, flags=re.IGNORECASE)
    # Remove trailing WW (World Wide web marker)
    title = re.sub(r'\s+WW\s*$', '', title)
    # Remove "HQ" quality marker at end
    title = re.sub(r'\s+HQ\s*$', '', title)
    # Clean up whitespace
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def file_sha256(filepath: str) -> str:
    """Calculate SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_year(date_str: Optional[str]) -> Optional[str]:
    """Extract a 4-digit year from various date formats. Rejects invalid years."""
    if not date_str:
        return None
    # Only match years 1400-2030
    match = re.search(r'(1[4-9]\d{2}|20[0-3]\d)', str(date_str))
    return match.group(1) if match else None


def ordinal(n) -> str:
    """Return ordinal string for a number: 1→'1st', 2→'2nd', 3→'3rd', etc."""
    n = int(n)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][n % 10] if n % 10 < 4 else 'th'}"


# =============================================================================
# GENRE / SUBJECT INFERENCE (no AI required)
# =============================================================================

# Dewey Decimal Classification → Genre mapping (top 2 levels)
DDC_TO_GENRE = {
    '0': 'Computers & General Works', '00': 'Computers & General Works',
    '001': 'Knowledge & Systems', '004': 'Computers', '005': 'Computers',
    '006': 'Computers',
    '1': 'Philosophy', '10': 'Philosophy', '13': 'Psychology',
    '14': 'Philosophy', '15': 'Psychology', '16': 'Philosophy',
    '17': 'Philosophy', '18': 'Philosophy', '19': 'Philosophy',
    '2': 'Religion', '20': 'Religion', '22': 'Religion',
    '23': 'Religion', '24': 'Religion', '25': 'Religion',
    '28': 'Religion', '29': 'Religion',
    '3': 'Social Sciences', '30': 'Social Sciences', '31': 'Social Sciences',
    '32': 'Political Science', '33': 'Business & Economics',
    '34': 'Law', '35': 'Public Administration', '36': 'Social Services',
    '37': 'Education', '38': 'Business & Economics', '39': 'Social Sciences',
    '4': 'Language', '40': 'Language', '41': 'Language',
    '5': 'Science', '50': 'Science', '51': 'Mathematics',
    '52': 'Science', '53': 'Science', '54': 'Science', '55': 'Science',
    '56': 'Science', '57': 'Science', '58': 'Science', '59': 'Science',
    '6': 'Technology & Engineering', '60': 'Technology & Engineering',
    '61': 'Medical', '62': 'Technology & Engineering',
    '63': 'Technology & Engineering', '64': 'Home & Garden',
    '65': 'Business & Economics', '66': 'Technology & Engineering',
    '67': 'Technology & Engineering', '68': 'Technology & Engineering',
    '69': 'Architecture',
    '7': 'Arts & Recreation', '70': 'Arts', '71': 'Architecture',
    '72': 'Architecture', '73': 'Arts', '74': 'Arts',
    '75': 'Arts', '76': 'Arts', '77': 'Photography',
    '78': 'Music', '79': 'Sports & Recreation',
    '8': 'Literature', '80': 'Literature', '81': 'Literature',
    '82': 'Literature', '83': 'Literature', '84': 'Literature',
    '85': 'Literature', '86': 'Literature', '87': 'Literature',
    '89': 'Literature',
    '9': 'History & Geography', '90': 'History', '91': 'Geography',
    '92': 'Biography & Autobiography', '93': 'History', '94': 'History',
    '95': 'History', '96': 'History', '97': 'History', '98': 'History',
    '99': 'History',
}

# Library of Congress Classification → Genre mapping
LCC_TO_GENRE = {
    'A': 'General Works', 'B': 'Philosophy & Religion',
    'BF': 'Psychology', 'BL': 'Religion', 'BM': 'Religion',
    'BP': 'Religion', 'BR': 'Religion', 'BS': 'Religion',
    'BT': 'Religion', 'BV': 'Religion', 'BX': 'Religion',
    'C': 'History', 'D': 'History', 'E': 'History', 'F': 'History',
    'G': 'Geography', 'H': 'Social Sciences',
    'HA': 'Social Sciences', 'HB': 'Business & Economics',
    'HC': 'Business & Economics', 'HD': 'Business & Economics',
    'HE': 'Business & Economics', 'HF': 'Business & Economics',
    'HG': 'Business & Economics', 'HJ': 'Business & Economics',
    'HM': 'Social Sciences', 'HN': 'Social Sciences',
    'HQ': 'Social Sciences', 'HV': 'Social Services',
    'J': 'Political Science', 'K': 'Law',
    'L': 'Education', 'M': 'Music',
    'N': 'Arts', 'P': 'Literature & Language',
    'PA': 'Literature', 'PB': 'Language', 'PC': 'Language',
    'PD': 'Language', 'PE': 'Language', 'PF': 'Language',
    'PG': 'Literature', 'PH': 'Literature', 'PJ': 'Literature',
    'PK': 'Literature', 'PL': 'Literature', 'PM': 'Language',
    'PN': 'Literature', 'PQ': 'Literature', 'PR': 'Literature',
    'PS': 'Literature', 'PT': 'Literature', 'PZ': 'Literature',
    'Q': 'Science', 'QA': 'Mathematics', 'QB': 'Science',
    'QC': 'Science', 'QD': 'Science', 'QE': 'Science',
    'QH': 'Science', 'QK': 'Science', 'QL': 'Science',
    'QM': 'Medical', 'QP': 'Medical', 'QR': 'Medical',
    'R': 'Medical', 'S': 'Agriculture',
    'T': 'Technology & Engineering', 'TA': 'Technology & Engineering',
    'TC': 'Technology & Engineering', 'TD': 'Technology & Engineering',
    'TE': 'Technology & Engineering', 'TF': 'Technology & Engineering',
    'TG': 'Technology & Engineering', 'TH': 'Architecture',
    'TJ': 'Technology & Engineering', 'TK': 'Technology & Engineering',
    'TL': 'Technology & Engineering', 'TN': 'Technology & Engineering',
    'TP': 'Technology & Engineering', 'TR': 'Photography',
    'TS': 'Technology & Engineering', 'TT': 'Crafts & Hobbies',
    'TX': 'Cooking', 'U': 'Military Science', 'V': 'Naval Science',
    'Z': 'Library Science',
}

# Title keyword → genre/subject inference
TITLE_KEYWORD_GENRES = {
    # Technology
    'programming': 'Computers', 'software': 'Computers', 'algorithm': 'Computers',
    'python': 'Computers', 'javascript': 'Computers', 'java': 'Computers',
    'docker': 'Computers', 'linux': 'Computers', 'kubernetes': 'Computers',
    'database': 'Computers', 'machine learning': 'Computers',
    'artificial intelligence': 'Computers', 'cybersecurity': 'Computers',
    'malware': 'Computers', 'hacking': 'Computers', 'forensics': 'Computers',
    'reverse engineering': 'Computers', 'networking': 'Computers',
    'cloud computing': 'Computers', 'devops': 'Computers',
    'web development': 'Computers', 'data science': 'Computers',
    'deep learning': 'Computers', 'neural network': 'Computers',
    'compiler': 'Computers', 'operating system': 'Computers',
    'cryptography': 'Computers', 'blockchain': 'Computers',
    # Engineering
    'circuit': 'Technology & Engineering', 'electronics': 'Technology & Engineering',
    'transistor': 'Technology & Engineering', 'microprocessor': 'Technology & Engineering',
    'embedded system': 'Technology & Engineering', 'robotics': 'Technology & Engineering',
    'arduino': 'Technology & Engineering', 'raspberry pi': 'Technology & Engineering',
    'signal processing': 'Technology & Engineering', 'FPGA': 'Technology & Engineering',
    'radar': 'Technology & Engineering', 'antenna': 'Technology & Engineering',
    'semiconductor': 'Technology & Engineering', 'VLSI': 'Technology & Engineering',
    'power supply': 'Technology & Engineering', 'amplifier': 'Technology & Engineering',
    # Self-help / Psychology / Personal Growth
    'happiness': 'Self-Help', 'mindfulness': 'Self-Help', 'meditation': 'Self-Help',
    'emotional intelligence': 'Self-Help', 'self-help': 'Self-Help',
    'affirmation': 'Self-Help', 'gratitude': 'Self-Help', 'resilience': 'Self-Help',
    'self-love': 'Self-Help', 'confidence': 'Self-Help', 'motivation': 'Self-Help',
    'habit': 'Self-Help', 'productivity': 'Self-Help', 'wellness': 'Self-Help',
    'personal growth': 'Self-Help', 'self-improvement': 'Self-Help',
    'change your life': 'Self-Help', 'change your thoughts': 'Self-Help',
    'mental toughness': 'Self-Help', 'inner peace': 'Self-Help',
    'self-discipline': 'Self-Help', 'self-esteem': 'Self-Help',
    'overthinking': 'Self-Help', 'procrastination': 'Self-Help',
    'stoicism': 'Self-Help', 'emotional agility': 'Self-Help',
    'get unstuck': 'Self-Help', 'thrive': 'Self-Help',
    'empowering': 'Self-Help', 'inner strength': 'Self-Help',
    'anxiety': 'Psychology', 'therapy': 'Psychology', 'psychology': 'Psychology',
    'cognitive': 'Psychology', 'behavioral': 'Psychology',
    'emotional rescue': 'Psychology', 'emotions': 'Psychology',
    # Business
    'business': 'Business & Economics', 'management': 'Business & Economics',
    'leadership': 'Business & Economics', 'entrepreneur': 'Business & Economics',
    'marketing': 'Business & Economics', 'finance': 'Business & Economics',
    'investing': 'Business & Economics', 'economics': 'Business & Economics',
    'negotiation': 'Business & Economics', 'startup': 'Business & Economics',
    # Health
    'nutrition': 'Health & Fitness', 'fitness': 'Health & Fitness',
    'diet': 'Health & Fitness', 'exercise': 'Health & Fitness',
    'yoga': 'Health & Fitness', 'medical': 'Medical',
    # Science
    'physics': 'Science', 'chemistry': 'Science', 'biology': 'Science',
    'mathematics': 'Mathematics', 'calculus': 'Mathematics',
    'algebra': 'Mathematics', 'statistics': 'Mathematics',
    # History & Biography
    'history': 'History', 'biography': 'Biography & Autobiography',
    'memoir': 'Biography & Autobiography', 'autobiography': 'Biography & Autobiography',
    # Fiction
    'novel': 'Fiction', 'mystery': 'Fiction', 'thriller': 'Fiction',
    'romance': 'Fiction', 'fantasy': 'Fiction', 'science fiction': 'Fiction',
    # Education
    'textbook': 'Education', 'curriculum': 'Education', 'teaching': 'Education',
}

# Publisher → likely genre
PUBLISHER_GENRES = {
    'packt': 'Computers', "o'reilly": 'Computers', 'apress': 'Computers',
    'no starch': 'Computers', 'manning': 'Computers', 'pragmatic': 'Computers',
    'wiley': None,  # too broad
    'springer': 'Science', 'elsevier': None,  # too broad
    'zondervan': 'Religion', 'baker': 'Religion', 'bethany': 'Religion',
    'hay house': 'Self-Help', 'sounds true': 'Self-Help',
    'penguin random': None, 'harpercollins': None, 'simon & schuster': None,
    'mcgraw': None, 'oxford university': None, 'cambridge university': None,
    'harvard business': 'Business & Economics', 'mit press': 'Science',
    'artech': 'Technology & Engineering', 'newnes': 'Technology & Engineering',
}


def infer_genres_subjects(meta) -> tuple:
    """Infer genres and subjects from DDC, LCC, title, tags, publisher.
    Returns (inferred_genres: list, inferred_subjects: list)."""
    inferred_genres = set()
    inferred_subjects = set()

    # 1. DDC code → genre
    if meta.ddc:
        ddc_num = re.sub(r'[^\d.]', '', str(meta.ddc))
        # Try progressively shorter prefixes: 005.1 → 005 → 00 → 0
        for length in [3, 2, 1]:
            prefix = ddc_num[:length]
            if prefix in DDC_TO_GENRE:
                inferred_genres.add(DDC_TO_GENRE[prefix])
                break

    # 2. LCC code → genre
    if meta.lcc:
        lcc_str = str(meta.lcc).strip()
        # Try 2-char prefix first (e.g., "QA"), then 1-char
        prefix2 = re.match(r'^([A-Z]{1,2})', lcc_str)
        if prefix2:
            p2 = prefix2.group(1)
            if p2 in LCC_TO_GENRE:
                inferred_genres.add(LCC_TO_GENRE[p2])
            elif p2[0] in LCC_TO_GENRE:
                inferred_genres.add(LCC_TO_GENRE[p2[0]])

    # 3. Title keyword → genre/subject
    if meta.title:
        title_lower = meta.title.lower()
        # Also check subtitle
        full_text = title_lower
        if meta.subtitle:
            full_text += ' ' + meta.subtitle.lower()
        if meta.description:
            full_text += ' ' + meta.description.lower()[:200]

        for keyword, genre in TITLE_KEYWORD_GENRES.items():
            if keyword in full_text:
                inferred_genres.add(genre)
                # Also add the keyword as a subject if it's a real topic noun/concept
                # Skip detection phrases like "change your life", "get unstuck"
                skip_words = {'your', 'get', 'change', 'inner', 'self-'}
                if (len(keyword) > 5
                        and keyword not in ('business', 'history', 'medical', 'thrive',
                                            'empowering', 'novel')
                        and not any(sw in keyword for sw in skip_words)
                        and len(keyword.split()) <= 2):
                    inferred_subjects.add(keyword.title())

    # 4. Publisher → genre hint
    if meta.publisher:
        pub_lower = meta.publisher.lower()
        for pub_key, genre in PUBLISHER_GENRES.items():
            if pub_key in pub_lower and genre:
                inferred_genres.add(genre)
                break

    # 5. Normalize existing tags into subjects
    if meta.tags:
        for tag in meta.tags:
            tag_clean = tag.strip()
            if tag_clean and len(tag_clean) > 2:
                # Skip very generic tags
                if tag_clean.lower() not in {'book', 'ebook', 'pdf', 'epub', 'general',
                                             'referex', 'unknown', 'none', 'other'}:
                    inferred_subjects.add(tag_clean)

    return list(inferred_genres), list(inferred_subjects)


# =============================================================================
# PROCESSING CACHE (SQLite)
# =============================================================================

class ProcessingCache:
    """SQLite-backed cache to track processed files, avoiding redundant API/AI calls."""

    SCHEMA_VERSION = 1

    def __init__(self, cache_path: str, logger: logging.Logger):
        self.cache_path = cache_path
        self.log = logger
        self._lock = threading.Lock()
        self._conn = None
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with robust settings."""
        # Fix: Add timeout for locking to prevent "database is locked" errors
        self._conn = sqlite3.connect(
            self.cache_path,
            check_same_thread=False,
            timeout=30.0
        )

        # Fix: Try WAL mode but fallback gracefully if file locking fails (common on network shares)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            self.log.warning("Could not set WAL mode (likely network share/locking issue). Using default journal.")
            self._conn.execute("PRAGMA journal_mode=DELETE")

        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_books (
                filepath TEXT PRIMARY KEY,
                file_size INTEGER NOT NULL,
                file_mtime REAL,
                completeness REAL,
                status TEXT NOT NULL DEFAULT 'success',
                sources_used TEXT,
                processed_at TEXT NOT NULL,
                title TEXT,
                authors TEXT,
                isbn TEXT,
                metadata_json TEXT
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        # Store schema version
        self._conn.execute(
            "INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?)",
            (str(self.SCHEMA_VERSION),)
        )
        self._conn.commit()

    def is_processed(self, filepath: str) -> Optional[dict]:
        """Check if a file has been processed and hasn't changed.
        Returns cached record dict if valid, None if needs (re)processing."""
        abspath = os.path.abspath(filepath)
        with self._lock:
            row = self._conn.execute(
                "SELECT file_size, file_mtime, completeness, status, processed_at "
                "FROM processed_books WHERE filepath = ?",
                (abspath,)
            ).fetchone()

        if not row:
            return None

        cached_size, cached_mtime, completeness, status, processed_at = row

        # Check if file has changed (size + mtime)
        try:
            stat = os.stat(filepath)
            if stat.st_size != cached_size or abs(stat.st_mtime - cached_mtime) > 1.0:
                self.log.log(TRACE, f"Cache stale (file changed): {filepath}")
                return None
        except OSError:
            return None

        return {
            'completeness': completeness,
            'status': status,
            'processed_at': processed_at,
        }

    def mark_processed(self, filepath: str, meta, status: str = 'success'):
        """Record that a file has been processed."""
        abspath = os.path.abspath(filepath)
        try:
            stat = os.stat(filepath)
            file_size = stat.st_size
            file_mtime = stat.st_mtime
        except OSError:
            file_size = 0
            file_mtime = 0

        # Serialize minimal metadata for cache
        metadata_json = json.dumps({
            'title': meta.title,
            'authors': meta.authors,
            'isbn_13': meta.isbn_13,
            'isbn_10': meta.isbn_10,
            'publisher': meta.publisher,
            'genres': meta.genres,
            'subjects': meta.subjects[:10],
            'language': meta.language,
            'completeness': meta.completeness_score(),
            'sources_used': meta.sources_used,
        }, ensure_ascii=False)

        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO processed_books
                (filepath, file_size, file_mtime, completeness, status,
                 sources_used, processed_at, title, authors, isbn, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                abspath, file_size, file_mtime,
                meta.completeness_score(), status,
                ','.join(meta.sources_used),
                datetime.now().isoformat(),
                meta.title,
                ' | '.join(meta.authors) if meta.authors else None,
                meta.isbn_13 or meta.isbn_10,
                metadata_json,
            ))
            self._conn.commit()

    def get_stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM processed_books").fetchone()[0]
            success = self._conn.execute(
                "SELECT COUNT(*) FROM processed_books WHERE status = 'success'"
            ).fetchone()[0]
            avg_comp = self._conn.execute(
                "SELECT AVG(completeness) FROM processed_books WHERE status = 'success'"
            ).fetchone()[0]
        return {
            'total': total,
            'success': success,
            'errors': total - success,
            'avg_completeness': round(avg_comp or 0, 2),
        }

    def remove_entry(self, filepath: str):
        """Remove a cache entry (for re-processing)."""
        abspath = os.path.abspath(filepath)
        with self._lock:
            self._conn.execute("DELETE FROM processed_books WHERE filepath = ?", (abspath,))
            self._conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()


def detect_gutenberg_id(filepath: str) -> Optional[str]:
    """Try to detect Gutenberg ID from filepath."""
    # Pattern: pg12345 in filename
    basename = os.path.basename(filepath)
    match = re.search(r'pg(\d+)', basename)
    if match:
        return match.group(1)
    # Pattern: /12345/ in path
    match = re.search(r'/(\d{1,6})/', filepath)
    if match:
        return match.group(1)
    return None


def clean_author_string(author: str) -> str:
    """Clean up an author name: strip publisher/year artifacts, brackets, etc."""
    # Remove (Publisher, Year) patterns
    author = re.sub(r'\s*\([^)]*\d{4}[^)]*\)\s*', ' ', author).strip()
    # Remove trailing "WW" or other common filename artifacts
    author = re.sub(r'\s+WW\s*$', '', author).strip()
    # Remove ALL bracketed content
    author = re.sub(r'\s*\[.*?\]\s*', ' ', author).strip()
    # Remove stray brackets
    author = re.sub(r'[\[\]]', '', author).strip()
    # Clean whitespace
    author = re.sub(r'\s+', ' ', author).strip()
    return author


def deduplicate_authors(authors: List[str]) -> List[str]:
    """Remove duplicate authors in different formats.
    "McMahon, David." + "David McMahon" → keeps first one seen.
    """
    if not authors or len(authors) <= 1:
        return authors

    def normalize(name: str) -> set:
        """Get significant name parts for comparison."""
        n = name.strip().rstrip('.')
        # Flip "Last, First" → "First Last"
        if ',' in n and n.count(',') == 1:
            parts = n.split(',', 1)
            n = f"{parts[1].strip()} {parts[0].strip()}"
        # Split on spaces AND periods (handles "E.AYERS" → "E" + "AYERS")
        tokens = re.split(r'[\s.]+', n)
        # Return set of lowercase tokens > 1 char (skip initials)
        return {t.lower() for t in tokens if len(t) > 1}

    result = []
    seen_parts = []
    for author in authors:
        parts = normalize(author)
        is_dup = False
        for existing in seen_parts:
            if parts and existing:
                overlap = parts & existing
                if len(overlap) >= min(len(parts), len(existing)):
                    is_dup = True
                    break
        if not is_dup:
            seen_parts.append(parts)
            result.append(author)
    return result


def parse_filename_metadata(filepath: str) -> BookMetadata:
    """Extract metadata heuristically from filename patterns.

    Handles: Author - Title (Pub, Year), Title - Subtitle - Author (Pub, Year),
    multi-dash filenames, WW suffixes, edition markers.
    """
    meta = BookMetadata()
    basename = Path(filepath).stem

    # Remove common suffixes: WW, images, etc.
    basename = re.sub(r'[-_\s]*\bWW\b\s*$', '', basename).strip()
    basename = re.sub(r'[-_\s]*(images|images-\d+)\s*$', '', basename, flags=re.IGNORECASE).strip()

    # Extract (Publisher, Year) from end
    publisher = None
    year = None
    pub_match = re.search(r'\(([^,)]+),\s*(\d{4})\)\s*$', basename)
    if pub_match:
        publisher = pub_match.group(1).strip()
        year = pub_match.group(2)
        basename = basename[:pub_match.start()].strip()
    if not year:
        year_match = re.search(r'\((\d{4})\)\s*$', basename)
        if year_match:
            year = year_match.group(1)
            basename = basename[:year_match.start()].strip()

    # Split on " - " / " – " / " — " separators (require spaces around dash)
    parts = re.split('\\s+[-\u2013\u2014]\\s+', basename)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        meta.processing_notes.append("filename-parsed")
        return meta

    if len(parts) == 1:
        meta.title = parts[0]
        if publisher: meta.publisher = publisher
        if year: meta.publication_date = year
        meta.processing_notes.append("filename-parsed")
        return meta

    def author_score(s):
        """Score how likely a string is an author name."""
        score = 0
        words = s.split()
        wc = len(words)
        # 2-3 words is ideal for a name ("Bob Katz", "J. Sanchez")
        if wc == 2 or wc == 3: score += 4
        # 1 word can be a surname but also a product name — slight bonus only
        elif wc == 1: score += 1
        # 4+ words is unlikely for just an author name
        if re.search(r'\b[A-Z]\.\s', s): score += 3       # Initials: "J." "M."
        if 'et al' in s.lower(): score += 4
        if re.search(r'\w+\s*[&,]\s*[A-Z]', s) and wc <= 6: score += 2
        if all(w[0].isupper() for w in words if w and w[0].isalpha()) and wc <= 4: score += 1
        # Penalty for title-like words
        title_words = {'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'with',
                       'how', 'why', 'what', 'guide', 'introduction', 'handbook',
                       'fundamentals', 'principles', 'analysis', 'programming',
                       'mastering', 'learning', 'practical', 'complete', 'essential',
                       'step-by-step', 'comprehensive', 'beginner', 'concepts'}
        score -= len(set(s.lower().split()) & title_words) * 2
        return score

    if len(parts) == 2:
        p0_words = len(parts[0].split())
        p1_words = len(parts[1].split())

        # Strong heuristic: single capitalized word vs 3+ words → single word is surname
        # This is the dominant "Surname - Book Title" pattern
        if p0_words == 1 and p1_words >= 3 and parts[0][0].isupper():
            meta.authors = [parts[0]]
            meta.title = parts[1]
        elif p1_words == 1 and p0_words >= 3 and parts[1][0].isupper():
            meta.title = parts[0]
            meta.authors = [parts[1]]
        else:
            # Fall through to scoring for ambiguous cases
            s_a = author_score(parts[0])
            s_b = author_score(parts[1])
            if s_b > s_a:
                meta.title = parts[0]
                meta.authors = [parts[1]]
            elif s_a > s_b:
                meta.authors = [parts[0]]
                meta.title = parts[1]
            elif len(parts[0]) <= len(parts[1]):
                meta.authors = [parts[0]]
                meta.title = parts[1]
            else:
                meta.title = parts[0]
                meta.authors = [parts[1]]
    else:
        # 3+ parts: find the author (usually first or last, shortest, most name-like)
        first_score = author_score(parts[0])
        last_score = author_score(parts[-1])

        if last_score >= 3 and last_score > first_score:
            meta.authors = [parts[-1]]
            meta.title = ' - '.join(parts[:-1])
        elif first_score >= 3 and first_score >= last_score:
            meta.authors = [parts[0]]
            meta.title = ' - '.join(parts[1:])
        else:
            meta.authors = [parts[0]]
            meta.title = ' - '.join(parts[1:])

    # Split multiple authors on & , ;
    if meta.authors:
        raw = meta.authors
        meta.authors = []
        for a in raw:
            a = a.strip()
            if not a: continue
            for sa in re.split(r'\s*[;&]\s*', a):
                sa = sa.strip()
                if not sa: continue
                if ', ' in sa:
                    comma_parts = [p.strip() for p in sa.split(',')]
                    if all(len(p.split()) >= 2 for p in comma_parts):
                        meta.authors.extend(comma_parts)
                        continue
                meta.authors.append(sa)

    if publisher: meta.publisher = publisher
    if year: meta.publication_date = year

    # Extract edition from title
    if meta.title:
        ed = re.search(r'(\d+)(?:st|nd|rd|th)\s*(?:ed|edition)', meta.title, re.IGNORECASE)
        if ed:
            meta.edition = f"{ordinal(ed.group(1))} Edition"
        else:
            ed2 = re.search(r'\b(\d+)e\b', meta.title)
            if ed2:
                meta.edition = f"{ordinal(ed2.group(1))} Edition"

    meta.processing_notes.append("filename-parsed")
    # Clean title artifacts
    if meta.title:
        meta.title = clean_title(meta.title)
    return meta
def run_command(cmd: list, timeout: int = 60) -> Tuple[int, str, str]:
    """Run a subprocess command with timeout."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


# =============================================================================
# TEXT EXTRACTORS
# =============================================================================

class TextExtractor:
    """Extract text from various ebook formats."""

    def __init__(self, logger: logging.Logger):
        self.log = logger

    def extract(self, filepath: str, max_chars: int = AI_EXTRACT_MAX_CHARS) -> Optional[str]:
        """Extract text from a file, dispatching by format."""
        ext = Path(filepath).suffix.lower()
        self.log.log(TRACE, f"Extracting text from {filepath} (format: {ext})")

        extractors = {
            '.epub': self._extract_epub,
            '.pdf': self._extract_pdf,
            '.mobi': self._extract_mobi,
            '.azw': self._extract_mobi,
            '.azw3': self._extract_mobi,
            '.fb2': self._extract_fb2,
            '.txt': self._extract_text,
            '.html': self._extract_html,
            '.htm': self._extract_html,
        }

        extractor = extractors.get(ext)
        if not extractor:
            self.log.warning(f"No text extractor for {ext}, trying calibre ebook-convert")
            return self._extract_via_calibre(filepath, max_chars)

        try:
            text = extractor(filepath, max_chars)
            if text:
                self.log.log(TRACE, f"Extracted {len(text)} chars from {filepath}")
            else:
                self.log.warning(f"No text extracted from {filepath}")
            return text
        except Exception as e:
            self.log.error(f"Text extraction failed for {filepath}: {e}")
            self.log.debug(traceback.format_exc())
            return None

    def _extract_epub(self, filepath: str, max_chars: int) -> Optional[str]:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        book = epub.read_epub(filepath, options={'ignore_ncx': True})
        text_parts = []
        total = 0

        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            if total >= max_chars:
                break
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            text_parts.append(text)
            total += len(text)

        return '\n\n'.join(text_parts)[:max_chars] if text_parts else None

    def _extract_pdf(self, filepath: str, max_chars: int) -> Optional[str]:
        import fitz  # PyMuPDF

        doc = fitz.open(filepath)
        text_parts = []
        total = 0
        max_pages = min(20, len(doc))  # First 20 pages

        for page_num in range(max_pages):
            if total >= max_chars:
                break
            page = doc[page_num]
            text = page.get_text()
            text_parts.append(text)
            total += len(text)

        doc.close()

        full_text = '\n\n'.join(text_parts)[:max_chars]

        # If very little text extracted, might be scanned - try OCR hint
        if len(full_text.strip()) < 100:
            self.log.warning(
                f"Very little text in PDF {filepath} - may be scanned/image-based. "
                f"Consider OCR with `ocrmypdf` first."
            )

        return full_text if full_text.strip() else None

    def _extract_mobi(self, filepath: str, max_chars: int) -> Optional[str]:
        # Try mobi library first
        try:
            import mobi
            tempdir, extracted = mobi.extract(filepath)
            # Find HTML file in extracted content
            for root, dirs, files in os.walk(tempdir):
                for f in files:
                    if f.endswith(('.html', '.htm')):
                        return self._extract_html(os.path.join(root, f), max_chars)
            # Try text files
            for root, dirs, files in os.walk(tempdir):
                for f in files:
                    if f.endswith('.txt'):
                        return self._extract_text(os.path.join(root, f), max_chars)
        except Exception as e:
            self.log.log(TRACE, f"mobi library failed: {e}, trying calibre")

        return self._extract_via_calibre(filepath, max_chars)

    def _extract_fb2(self, filepath: str, max_chars: int) -> Optional[str]:
        tree = ET.parse(filepath)
        root = tree.getroot()
        # FB2 namespace
        ns = re.match(r'\{.*\}', root.tag)
        ns = ns.group(0) if ns else ''

        text_parts = []
        for body in root.iter(f'{ns}body'):
            for p in body.iter(f'{ns}p'):
                text = ''.join(p.itertext()).strip()
                if text:
                    text_parts.append(text)
                if sum(len(t) for t in text_parts) >= max_chars:
                    break

        return '\n'.join(text_parts)[:max_chars] if text_parts else None

    def _extract_text(self, filepath: str, max_chars: int) -> Optional[str]:
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    return f.read(max_chars)
            except (UnicodeDecodeError, UnicodeError):
                continue
        return None

    def _extract_html(self, filepath: str, max_chars: int) -> Optional[str]:
        from bs4 import BeautifulSoup
        raw = self._extract_text(filepath, max_chars * 2)
        if not raw:
            return None
        soup = BeautifulSoup(raw, 'html.parser')
        return soup.get_text(separator='\n', strip=True)[:max_chars]

    def _extract_via_calibre(self, filepath: str, max_chars: int) -> Optional[str]:
        """Use calibre's ebook-convert to convert to txt for extraction."""
        tmp_txt = filepath + '.tmp_extract.txt'
        try:
            rc, out, err = run_command(
                ['ebook-convert', filepath, tmp_txt, '--txt-output-formatting=plain'],
                timeout=120
            )
            if rc == 0 and os.path.exists(tmp_txt):
                return self._extract_text(tmp_txt, max_chars)
            else:
                self.log.warning(f"Calibre conversion failed: {err}")
                return None
        finally:
            if os.path.exists(tmp_txt):
                os.remove(tmp_txt)


# =============================================================================
# EMBEDDED METADATA EXTRACTOR
# =============================================================================

class EmbeddedMetadataExtractor:
    """Extract metadata already embedded in ebook files."""

    def __init__(self, logger: logging.Logger):
        self.log = logger

    def extract(self, filepath: str) -> BookMetadata:
        """Extract embedded metadata from file."""
        ext = Path(filepath).suffix.lower()
        meta = BookMetadata()

        # Try calibre's ebook-meta first (works for most formats)
        calibre_meta = self._extract_calibre(filepath)
        if calibre_meta:
            meta.merge(calibre_meta, "calibre-embedded")

        # Format-specific extraction for additional fields
        extractors = {
            '.epub': self._extract_epub_meta,
            '.pdf': self._extract_pdf_meta,
            '.fb2': self._extract_fb2_meta,
        }

        specific = extractors.get(ext)
        if specific:
            try:
                specific_meta = specific(filepath)
                if specific_meta:
                    meta.merge(specific_meta, f"{ext}-embedded")
            except Exception as e:
                self.log.log(TRACE, f"Format-specific extraction failed for {filepath}: {e}")

        return meta

    def _extract_calibre(self, filepath: str) -> Optional[BookMetadata]:
        """Use calibre's ebook-meta CLI to read metadata."""
        rc, out, err = run_command(['ebook-meta', filepath])
        if rc != 0:
            self.log.log(TRACE, f"ebook-meta failed for {filepath}: {err}")
            return None

        meta = BookMetadata()
        field_map = {
            'Title': 'title',
            'Author(s)': '_authors',
            'Publisher': 'publisher',
            'Language': 'language',
            'Published': 'publication_date',
            'Identifiers': '_identifiers',
            'Tags': '_tags',
            'Series': '_series',
            'Comments': 'description',
            'Rights': 'rights',
        }

        for line in out.strip().split('\n'):
            if ':' not in line:
                continue
            key, _, value = line.partition(':')
            key = key.strip()
            value = value.strip()

            if not value or value.lower() == 'unknown':
                continue

            mapped = field_map.get(key)
            if not mapped:
                continue

            if mapped == '_authors':
                # "Author1 & Author2" or "Author1; Author2"
                # Also handle calibre bracket format: "Last First [First Last]"
                # First strip all bracketed content
                raw = re.sub(r'\s*\[.*?\]\s*', ' ', value)
                # Also catch unmatched/stray brackets
                raw = re.sub(r'[\[\]]', '', raw)
                raw = re.sub(r'\s+', ' ', raw).strip()
                authors = re.split(r'\s*[;&]\s*', raw)
                meta.authors = [a.strip() for a in authors if a.strip()]
            elif mapped == '_identifiers':
                # "isbn:123, google:abc, amazon:xyz"
                for ident in value.split(','):
                    ident = ident.strip()
                    if ':' in ident:
                        id_type, _, id_val = ident.partition(':')
                        id_type = id_type.strip().lower()
                        id_val = id_val.strip()
                        if id_type == 'isbn' and len(id_val) == 13:
                            meta.isbn_13 = id_val
                        elif id_type == 'isbn' and len(id_val) == 10:
                            meta.isbn_10 = id_val
                        elif id_type == 'google':
                            meta.google_books_id = id_val
                        elif id_type == 'amazon':
                            meta.asin = id_val
                        elif id_type == 'doi':
                            meta.doi = id_val
            elif mapped == '_tags':
                meta.tags = [t.strip() for t in value.split(',') if t.strip()]
            elif mapped == '_series':
                # "Series Name #3"
                series_match = re.match(r'(.+?)(?:\s*#\s*(\d+(?:\.\d+)?))?$', value)
                if series_match:
                    meta.series = series_match.group(1).strip()
                    if series_match.group(2):
                        meta.series_index = float(series_match.group(2))
            else:
                setattr(meta, mapped, value)

        return meta

    def _extract_epub_meta(self, filepath: str) -> Optional[BookMetadata]:
        """Deep EPUB OPF metadata extraction."""
        try:
            import ebooklib
            from ebooklib import epub
        except ImportError:
            return None

        meta = BookMetadata()
        book = epub.read_epub(filepath, options={'ignore_ncx': True})

        # Title
        titles = book.get_metadata('DC', 'title')
        if titles:
            meta.title = titles[0][0]

        # Authors/creators
        creators = book.get_metadata('DC', 'creator')
        for creator in creators:
            name = creator[0]
            role = creator[1].get('{http://www.idpf.org/2007/opf}role', 'aut')
            if role == 'aut':
                meta.authors.append(name)
            elif role == 'edt':
                meta.editors.append(name)
            elif role == 'trl':
                meta.translators.append(name)
            elif role == 'ill':
                meta.illustrators.append(name)
            else:
                meta.contributors.append(name)

        # Language
        langs = book.get_metadata('DC', 'language')
        if langs:
            meta.language = langs[0][0]

        # Date
        dates = book.get_metadata('DC', 'date')
        if dates:
            meta.publication_date = dates[0][0]

        # Publisher
        pubs = book.get_metadata('DC', 'publisher')
        if pubs:
            meta.publisher = pubs[0][0]

        # Description
        descs = book.get_metadata('DC', 'description')
        if descs:
            meta.description = descs[0][0]

        # Subjects
        subjects = book.get_metadata('DC', 'subject')
        for subj in subjects:
            meta.subjects.append(subj[0])

        # Rights
        rights = book.get_metadata('DC', 'rights')
        if rights:
            meta.rights = rights[0][0]

        # Identifiers
        identifiers = book.get_metadata('DC', 'identifier')
        for ident in identifiers:
            val = ident[0]
            scheme = ident[1].get('{http://www.idpf.org/2007/opf}scheme', '').lower()
            if scheme == 'isbn' or re.match(r'^97[89]\d{10}$', val):
                if len(val) == 13:
                    meta.isbn_13 = val
                elif len(val) == 10:
                    meta.isbn_10 = val
            elif 'gutenberg' in val.lower() or 'gutenberg' in scheme:
                gid = re.search(r'(\d+)', val)
                if gid:
                    meta.gutenberg_id = gid.group(1)

        return meta

    def _extract_pdf_meta(self, filepath: str) -> Optional[BookMetadata]:
        """Extract PDF metadata."""
        try:
            import fitz
        except ImportError:
            return None

        meta = BookMetadata()
        doc = fitz.open(filepath)
        pdf_meta = doc.metadata

        if pdf_meta:
            meta.title = pdf_meta.get('title') or None
            author = pdf_meta.get('author')
            if author:
                meta.authors = [a.strip() for a in re.split(r'[,;&]', author) if a.strip()]
            meta.description = pdf_meta.get('subject') or None
            if pdf_meta.get('keywords'):
                meta.tags = [k.strip() for k in pdf_meta['keywords'].split(',') if k.strip()]
            meta.publisher = pdf_meta.get('producer') or pdf_meta.get('creator') or None
            if pdf_meta.get('creationDate'):
                # PDF dates: D:YYYYMMDDHHmmSS
                date_match = re.search(r'(\d{4})', pdf_meta['creationDate'])
                if date_match:
                    meta.publication_date = date_match.group(1)

        meta.page_count = len(doc)
        doc.close()

        return meta

    def _extract_fb2_meta(self, filepath: str) -> Optional[BookMetadata]:
        """Extract FB2 metadata."""
        meta = BookMetadata()
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns_match = re.match(r'\{(.*)\}', root.tag)
        ns = ns_match.group(1) if ns_match else ''
        nsmap = {'fb': ns} if ns else {}

        def find(parent, tag):
            if ns:
                return parent.find(f'fb:{tag}', nsmap)
            return parent.find(tag)

        def findall(parent, tag):
            if ns:
                return parent.findall(f'fb:{tag}', nsmap)
            return parent.findall(tag)

        def findtext(parent, tag):
            el = find(parent, tag)
            return el.text.strip() if el is not None and el.text else None

        desc = find(root, 'description')
        if desc is None:
            return meta

        title_info = find(desc, 'title-info')
        if title_info is not None:
            bt = find(title_info, 'book-title')
            if bt is not None and bt.text:
                meta.title = bt.text.strip()

            for author_el in findall(title_info, 'author'):
                parts = []
                for tag in ['first-name', 'middle-name', 'last-name']:
                    el = find(author_el, tag)
                    if el is not None and el.text:
                        parts.append(el.text.strip())
                if parts:
                    meta.authors.append(' '.join(parts))

            lang = findtext(title_info, 'lang')
            if lang:
                meta.language = lang

            for genre_el in findall(title_info, 'genre'):
                if genre_el.text:
                    meta.genres.append(genre_el.text.strip())

            ann = find(title_info, 'annotation')
            if ann is not None:
                meta.description = ''.join(ann.itertext()).strip()

            date_el = find(title_info, 'date')
            if date_el is not None:
                meta.publication_date = date_el.get('value') or (date_el.text or '').strip()

            seq = find(title_info, 'sequence')
            if seq is not None:
                meta.series = seq.get('name')
                num = seq.get('number')
                if num:
                    try:
                        meta.series_index = float(num)
                    except ValueError:
                        pass

        publish_info = find(desc, 'publish-info')
        if publish_info is not None:
            meta.publisher = findtext(publish_info, 'publisher')
            meta.isbn_13 = findtext(publish_info, 'isbn')
            year = findtext(publish_info, 'year')
            if year and not meta.publication_date:
                meta.publication_date = year

        return meta


# =============================================================================
# GUTENBERG RDF CATALOG
# =============================================================================

class GutenbergRDFCatalog:
    """Parse and query Gutenberg's RDF/XML catalog."""

    def __init__(self, rdf_dir: str, logger: logging.Logger):
        self.rdf_dir = rdf_dir
        self.log = logger
        self._cache: Dict[str, BookMetadata] = {}

    def lookup(self, gutenberg_id: str) -> Optional[BookMetadata]:
        """Look up a book by Gutenberg ID in the RDF catalog."""
        if gutenberg_id in self._cache:
            self.log.log(TRACE, f"RDF cache hit for PG#{gutenberg_id}")
            return self._cache[gutenberg_id]

        rdf_path = os.path.join(self.rdf_dir, 'cache', 'epub', gutenberg_id, f'pg{gutenberg_id}.rdf')
        if not os.path.exists(rdf_path):
            # Try alternate paths
            alt_paths = [
                os.path.join(self.rdf_dir, gutenberg_id, f'pg{gutenberg_id}.rdf'),
                os.path.join(self.rdf_dir, f'pg{gutenberg_id}.rdf'),
                os.path.join(self.rdf_dir, 'epub', gutenberg_id, f'pg{gutenberg_id}.rdf'),
            ]
            rdf_path = None
            for alt in alt_paths:
                if os.path.exists(alt):
                    rdf_path = alt
                    break

            if not rdf_path:
                self.log.log(TRACE, f"No RDF file found for PG#{gutenberg_id}")
                return None

        self.log.log(TRACE, f"Parsing RDF for PG#{gutenberg_id}: {rdf_path}")
        try:
            meta = self._parse_rdf(rdf_path, gutenberg_id)
            self._cache[gutenberg_id] = meta
            return meta
        except Exception as e:
            self.log.error(f"Failed to parse RDF for PG#{gutenberg_id}: {e}")
            self.log.debug(traceback.format_exc())
            return None

    def _parse_rdf(self, rdf_path: str, gutenberg_id: str) -> BookMetadata:
        """Parse a single RDF file into BookMetadata."""
        tree = ET.parse(rdf_path)
        root = tree.getroot()
        meta = BookMetadata()
        meta.gutenberg_id = gutenberg_id

        # Find the ebook element
        ebook = root.find('.//pgterms:ebook', RDF_NS)
        if ebook is None:
            # Try without namespace
            for child in root:
                if 'ebook' in child.tag.lower():
                    ebook = child
                    break

        if ebook is None:
            self.log.warning(f"No ebook element found in RDF for PG#{gutenberg_id}")
            return meta

        # Title
        title_el = ebook.find('dcterms:title', RDF_NS)
        if title_el is not None and title_el.text:
            full_title = title_el.text.strip()
            # Split "Title\nSubtitle" or "Title: Subtitle"
            if '\n' in full_title:
                parts = full_title.split('\n', 1)
                meta.title = parts[0].strip()
                meta.subtitle = parts[1].strip()
            elif ': ' in full_title and len(full_title.split(': ', 1)[0]) > 3:
                parts = full_title.split(': ', 1)
                meta.title = parts[0].strip()
                meta.subtitle = parts[1].strip()
            else:
                meta.title = full_title

        # Creators (authors, editors, etc.)
        for creator in ebook.findall('dcterms:creator', RDF_NS):
            agent = creator.find('pgterms:agent', RDF_NS)
            if agent is not None:
                name_el = agent.find('pgterms:name', RDF_NS)
                if name_el is not None and name_el.text:
                    meta.authors.append(name_el.text.strip())

        # Additional roles via marcrel
        role_map = {
            'marcrel:edt': 'editors',
            'marcrel:trl': 'translators',
            'marcrel:ill': 'illustrators',
            'marcrel:ctb': 'contributors',
        }
        for role_ns, field_name in role_map.items():
            prefix, tag = role_ns.split(':')
            for el in ebook.findall(f'{prefix}:{tag}', RDF_NS):
                agent = el.find('pgterms:agent', RDF_NS)
                if agent is not None:
                    name_el = agent.find('pgterms:name', RDF_NS)
                    if name_el is not None and name_el.text:
                        getattr(meta, field_name).append(name_el.text.strip())

        # Language
        lang_el = ebook.find('.//dcterms:language//rdf:value', RDF_NS)
        if lang_el is not None and lang_el.text:
            meta.language = lang_el.text.strip()

        # Subjects (LCSH)
        for subject in ebook.findall('dcterms:subject', RDF_NS):
            desc = subject.find('rdf:Description', RDF_NS)
            if desc is not None:
                member_of = desc.find('dcam:memberOf', RDF_NS)
                value = desc.find('rdf:value', RDF_NS)
                if value is not None and value.text:
                    val = value.text.strip()
                    if member_of is not None:
                        resource = member_of.get(f'{{{RDF_NS["rdf"]}}}resource', '')
                        if 'LCSH' in resource:
                            meta.subjects.append(val)
                        elif 'LCC' in resource:
                            meta.lcc = val
                    else:
                        meta.subjects.append(val)

        # Rights
        rights_el = ebook.find('dcterms:rights', RDF_NS)
        if rights_el is not None and rights_el.text:
            meta.rights = rights_el.text.strip()

        # License
        license_el = ebook.find('cc:license', RDF_NS)
        if license_el is not None:
            meta.license_url = license_el.get(f'{{{RDF_NS["rdf"]}}}resource', '')

        # Issued date
        issued = ebook.find('dcterms:issued', RDF_NS)
        if issued is not None and issued.text:
            meta.publication_date = issued.text.strip()

        # Publisher
        publisher_el = ebook.find('dcterms:publisher', RDF_NS)
        if publisher_el is not None and publisher_el.text:
            meta.publisher = publisher_el.text.strip()

        # Description
        desc_el = ebook.find('dcterms:description', RDF_NS)
        if desc_el is not None and desc_el.text:
            meta.description = desc_el.text.strip()

        return meta


# =============================================================================
# PUBLIC API LOOKUPS
# =============================================================================

class PublicAPILookup:
    """Search Open Library and Google Books for metadata."""

    # Google Books free tier: ~100 requests/100 seconds without API key
    # Open Library: ~100 requests/5 minutes
    GOOGLE_RATE_LIMIT = 1.5   # seconds between Google requests
    OL_RATE_LIMIT = 0.5       # seconds between Open Library requests
    MAX_RETRIES = 3
    BACKOFF_BASE = 2.0        # exponential backoff base (2s, 4s, 8s)
    CIRCUIT_BREAK_THRESHOLD = 5  # disable API after this many consecutive 429s

    def __init__(self, logger: logging.Logger, google_api_keys: Optional[List[str]] = None):
        self.log = logger
        # API key rotation: list of keys with round-robin index and per-key disable tracking
        self.google_api_keys = google_api_keys or []
        self._google_key_index = 0
        self._google_key_lock = threading.Lock()
        self._disabled_google_keys: set = set()  # keys that got 403/quota-exhausted
        self._session = None
        self._session_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_google_call = 0.0
        self._last_ol_call = 0.0
        self._google_backoff = 0  # current backoff level
        self._ol_backoff = 0
        self._google_consecutive_429s = 0
        self._ol_consecutive_429s = 0
        self._google_disabled = False
        self._ol_disabled = False

    @property
    def session(self):
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    import requests
                    self._session = requests.Session()
                    self._session.headers.update({
                        'User-Agent': 'EbookMetadataPipeline/1.0 (contact: github.com)'
                    })
        return self._session

    def _next_google_key(self) -> Optional[str]:
        """Get the next available Google API key (round-robin, skipping disabled keys)."""
        if not self.google_api_keys:
            return None
        with self._google_key_lock:
            total = len(self.google_api_keys)
            # Try each key once looking for a non-disabled one
            for _ in range(total):
                key = self.google_api_keys[self._google_key_index % total]
                self._google_key_index += 1
                if key not in self._disabled_google_keys:
                    return key
            # All keys disabled
            return None

    def _disable_google_key(self, key: str):
        """Disable a specific Google API key (403/quota exhausted)."""
        with self._google_key_lock:
            self._disabled_google_keys.add(key)
            remaining = len(self.google_api_keys) - len(self._disabled_google_keys)
            self.log.warning(
                f"⚠ Disabled Google API key ...{key[-8:]} — "
                f"{remaining}/{len(self.google_api_keys)} keys remaining"
            )
            if remaining == 0:
                self._google_disabled = True
                self.log.warning(
                    f"⚠ All {len(self.google_api_keys)} Google API key(s) exhausted! "
                    f"Disabling Google Books for this run."
                )

    def _clean_author(self, author: str) -> str:
        """Clean author name from calibre format like 'Last First [First Last]'."""
        if not author:
            return author
        # Remove ALL bracketed alternate formats (handles double brackets too)
        # "Alwill Leyba Cara [Cara, Alwill Leyba] [Cara, Alwill Leyba]"
        author = re.sub(r'\s*\[.*?\]\s*', ' ', author).strip()
        # Clean up any leftover whitespace
        author = re.sub(r'\s+', ' ', author).strip()
        # If in "Last, First" format, flip it (only simple single-comma case)
        if ',' in author and author.count(',') == 1:
            parts = author.split(',', 1)
            author = f"{parts[1].strip()} {parts[0].strip()}"
        return author

    def _rate_limit(self, api: str):
        """Enforce rate limiting between API calls (thread-safe)."""
        with self._rate_lock:
            now = time.time()
            if api == 'google':
                delay = self.GOOGLE_RATE_LIMIT + (self.BACKOFF_BASE ** self._google_backoff - 1 if self._google_backoff else 0)
                elapsed = now - self._last_google_call
                if elapsed < delay:
                    wait = delay - elapsed
                    self.log.log(TRACE, f"Rate limiting Google API: waiting {wait:.1f}s")
                    time.sleep(wait)
                self._last_google_call = time.time()
            elif api == 'openlibrary':
                delay = self.OL_RATE_LIMIT + (self.BACKOFF_BASE ** self._ol_backoff - 1 if self._ol_backoff else 0)
                elapsed = now - self._last_ol_call
                if elapsed < delay:
                    wait = delay - elapsed
                    self.log.log(TRACE, f"Rate limiting Open Library API: waiting {wait:.1f}s")
                    time.sleep(wait)
                self._last_ol_call = time.time()

    def _request_with_retry(self, api: str, method: str, url: str,
                            **kwargs) -> Optional['requests.Response']:
        """Make an HTTP request with retry and exponential backoff on 429."""
        import requests

        # Circuit breaker check
        if api == 'google' and self._google_disabled:
            self.log.log(TRACE, f"Google API circuit breaker OPEN — skipping request")
            return None
        if api == 'openlibrary' and self._ol_disabled:
            self.log.log(TRACE, f"Open Library circuit breaker OPEN — skipping request")
            return None

        for attempt in range(self.MAX_RETRIES + 1):
            self._rate_limit(api)
            try:
                kwargs.setdefault('timeout', 15)
                resp = self.session.request(method, url, **kwargs)

                if resp.status_code == 429:
                    # Track consecutive 429s for circuit breaker
                    if api == 'google':
                        self._google_consecutive_429s += 1
                        if self._google_consecutive_429s >= self.CIRCUIT_BREAK_THRESHOLD:
                            self._google_disabled = True
                            self.log.warning(
                                f"⚠ Google API circuit breaker OPEN after "
                                f"{self._google_consecutive_429s} consecutive 429s. "
                                f"Skipping Google for remaining files. "
                                f"Consider using --google-api-key for higher limits."
                            )
                            return None
                    else:
                        self._ol_consecutive_429s += 1
                        if self._ol_consecutive_429s >= self.CIRCUIT_BREAK_THRESHOLD:
                            self._ol_disabled = True
                            self.log.warning(
                                f"⚠ Open Library circuit breaker OPEN after "
                                f"{self._ol_consecutive_429s} consecutive 429s."
                            )
                            return None

                    # Parse Retry-After header if present
                    retry_after = resp.headers.get('Retry-After')
                    if retry_after:
                        try:
                            wait = int(retry_after)
                        except ValueError:
                            wait = self.BACKOFF_BASE ** (attempt + 1)
                    else:
                        wait = self.BACKOFF_BASE ** (attempt + 1)

                    # Increase persistent backoff for this API
                    if api == 'google':
                        self._google_backoff = min(self._google_backoff + 1, 4)
                    else:
                        self._ol_backoff = min(self._ol_backoff + 1, 4)

                    if attempt < self.MAX_RETRIES:
                        self.log.warning(
                            f"429 rate limited by {api} (attempt {attempt+1}/{self.MAX_RETRIES+1}), "
                            f"waiting {wait:.0f}s..."
                        )
                        time.sleep(wait)
                        continue
                    else:
                        self.log.warning(f"429 rate limited by {api}, max retries exhausted")
                        return None

                resp.raise_for_status()

                # Successful — reset consecutive 429 counter and reduce backoff
                if api == 'google':
                    self._google_consecutive_429s = 0
                    self._google_backoff = max(0, self._google_backoff - 1)
                else:
                    self._ol_consecutive_429s = 0
                    self._ol_backoff = max(0, self._ol_backoff - 1)

                return resp

            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response is not None else 0
                if status_code == 403 and api == 'google':
                    # Bad API key or quota exceeded — disable THIS key, try others
                    current_key = kwargs.get('params', {}).get('key')
                    if current_key:
                        self._disable_google_key(current_key)
                    else:
                        self._google_disabled = True
                        self.log.warning(
                            f"⚠ Google API returned 403 Forbidden — disabling for this run."
                        )
                    return None
                if '429' not in str(e):
                    self.log.warning(f"{api} HTTP error: {e}")
                    return None
            except requests.exceptions.Timeout:
                self.log.warning(f"{api} request timed out (attempt {attempt+1})")
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.BACKOFF_BASE ** attempt)
                    continue
                return None
            except requests.exceptions.ConnectionError as e:
                self.log.warning(f"{api} connection error: {e}")
                return None

        return None

    def _simplify_title(self, title: str) -> str:
        """Strip subtitle, edition markers, and parentheticals for better search."""
        # Remove everything after colon (subtitle)
        simplified = re.split(r'\s*[:]\s*', title)[0]
        # Remove edition markers: "- Second Edition", "4th ed", etc.
        simplified = re.sub(r'\s*[-–—]\s*((\d+\w*|first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\s+edition)\b.*', '', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\s*\d+(?:st|nd|rd|th)\s*(?:ed|edition)\b', '', simplified, flags=re.IGNORECASE)
        simplified = re.sub(r'\s*[-–—]\s*\w+\s+Edition\b', '', simplified, flags=re.IGNORECASE)
        # Remove parenthetical content
        simplified = re.sub(r'\s*\([^)]*\)\s*', ' ', simplified)
        return simplified.strip()

    def _title_matches(self, result_title: str, search_title: str, author: Optional[str] = None) -> bool:
        """Check if an API result title is a reasonable match for our search."""
        if not result_title or not search_title:
            return False
        # Normalize both
        rt = re.sub(r'[^\w\s]', '', result_title.lower()).split()
        st = re.sub(r'[^\w\s]', '', search_title.lower()).split()
        if not rt or not st:
            return False
        stop_words = {'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'with', 'on', 'by', 'is', 'at'}
        rt_words = {w for w in rt if w not in stop_words and len(w) > 2}
        st_words = {w for w in st if w not in stop_words and len(w) > 2}
        if not rt_words or not st_words:
            return True  # Nothing to compare
        overlap = rt_words & st_words
        # Require overlap from BOTH sides: prevents "Docker" matching any Docker book
        # At least 40% of search title words must appear in result
        search_ratio = len(overlap) / len(st_words) if st_words else 0
        # At least 30% of result title words must appear in search
        result_ratio = len(overlap) / len(rt_words) if rt_words else 0
        return search_ratio >= 0.4 and result_ratio >= 0.3

    def search(self, title: Optional[str] = None, author: Optional[str] = None,
               isbn: Optional[str] = None, current_completeness: float = 0.0) -> BookMetadata:
        """Search public APIs: Open Library first (free), Google Books only if needed.

        Google fallback logic:
          - If OL returned nothing → try Google
          - If OL succeeded but completeness still < 70% → try Google
          - If OL got us to >= 70% → skip Google

        Returns:
          BookMetadata with api_diagnostics list attached for caller logging.
        """
        meta = BookMetadata()

        # Diagnostics list: [(api_name, status, detail_str), ...]
        # status: 'success', 'no_match', 'error', 'skipped'
        meta._api_diagnostics = []

        # Clean author name
        if author:
            author = self._clean_author(author)
            self.log.log(TRACE, f"Cleaned author: '{author}'")

        # Build search title variants once
        search_titles = []
        if title:
            clean_t = re.split(r'\s*:\s*', title)[0].strip()
            simplified = self._simplify_title(title)
            for t in [title, clean_t, simplified]:
                if (t and len(t) >= 4 and t not in search_titles
                        and len(t.split()) >= 2):
                    search_titles.append(t)
            if not search_titles and title and len(title) >= 4:
                search_titles = [title]

        # ── Phase 1: Open Library (free, no key needed) ──────────
        ol_got_results = False
        ol_fields = []

        def _track_ol_fields(ol_meta):
            """Track which fields OL actually provided."""
            found = []
            if ol_meta.isbn_13 or ol_meta.isbn_10: found.append('ISBN')
            if ol_meta.language: found.append('language')
            if ol_meta.description: found.append('description')
            if ol_meta.page_count: found.append('pages')
            if ol_meta.cover_url: found.append('cover')
            if ol_meta.subjects: found.append('subjects')
            if ol_meta.genres: found.append('genres')
            if ol_meta.publisher: found.append('publisher')
            if ol_meta.lcc: found.append('LCC')
            if ol_meta.ddc: found.append('DDC')
            if ol_meta.original_publication_date: found.append('orig_date')
            if ol_meta.authors: found.append('authors')
            return found

        # ISBN lookup
        if isbn:
            ol_meta = self._openlibrary_isbn(isbn)
            if ol_meta:
                ol_fields = _track_ol_fields(ol_meta)
                meta.merge(ol_meta, "openlibrary-isbn")
                ol_got_results = True

        # Title/author search (fill remaining gaps)
        if search_titles and not (meta.subjects and meta.description and meta.page_count):
            for search_t in search_titles:
                ol_meta = self._openlibrary_search(search_t, author)
                if ol_meta and ol_meta.title:
                    if self._title_matches(ol_meta.title, title, author):
                        new_fields = _track_ol_fields(ol_meta)
                        # Add fields not already tracked
                        for f in new_fields:
                            if f not in ol_fields:
                                ol_fields.append(f)
                        meta.merge(ol_meta, "openlibrary-search")
                        ol_got_results = True
                        break
                    else:
                        self.log.log(TRACE, f"OL result '{ol_meta.title}' doesn't match '{title}' — skipping")

        # Record OL diagnostic
        if ol_got_results:
            meta._api_diagnostics.append(('Open Library', 'success', ', '.join(ol_fields) if ol_fields else 'matched'))
        else:
            meta._api_diagnostics.append(('Open Library', 'no_match', 'no results'))

        # ── Phase 2: Google Books fallback decision ──────────────
        estimated_completeness = current_completeness
        if ol_got_results:
            estimated_completeness = max(current_completeness, meta.completeness_score())

        need_google = False
        google_reason = ""
        if not ol_got_results:
            need_google = True
            google_reason = "OL had no match"
        elif estimated_completeness < 0.7:
            need_google = True
            google_reason = f"OL only reached {estimated_completeness:.0%}"
        else:
            meta._api_diagnostics.append(('Google Books', 'skipped', f'OL sufficient at {estimated_completeness:.0%}'))

        if need_google:
            google_got_results = False
            google_fields = []

            def _track_google_fields(gb_meta):
                found = []
                if gb_meta.isbn_13 or gb_meta.isbn_10: found.append('ISBN')
                if gb_meta.language: found.append('language')
                if gb_meta.description: found.append('description')
                if gb_meta.page_count: found.append('pages')
                if gb_meta.cover_url: found.append('cover')
                if gb_meta.subjects: found.append('subjects')
                if gb_meta.genres: found.append('genres')
                if gb_meta.publisher: found.append('publisher')
                if gb_meta.authors: found.append('authors')
                return found

            # ISBN lookup on Google
            if isbn:
                gb_meta = self._google_books_isbn(isbn)
                if gb_meta:
                    google_fields = _track_google_fields(gb_meta)
                    meta.merge(gb_meta, "google-books-isbn")
                    google_got_results = True

            # Title search on Google
            if search_titles and not (meta.description and meta.page_count):
                for search_t in search_titles:
                    gb_meta = self._google_books_search(search_t, author)
                    if gb_meta and gb_meta.title:
                        if self._title_matches(gb_meta.title, title, author):
                            new_fields = _track_google_fields(gb_meta)
                            for f in new_fields:
                                if f not in google_fields:
                                    google_fields.append(f)
                            meta.merge(gb_meta, "google-books-search")
                            google_got_results = True
                            break
                        else:
                            self.log.log(TRACE,
                                f"Google result '{gb_meta.title}' doesn't match '{title}' — skipping")

            # Record Google diagnostic
            if google_got_results:
                meta._api_diagnostics.append(('Google Books', 'success', ', '.join(google_fields) if google_fields else 'matched'))
            elif self._google_disabled:
                meta._api_diagnostics.append(('Google Books', 'error', 'disabled (keys exhausted)'))
            else:
                meta._api_diagnostics.append(('Google Books', 'no_match', f'no results ({google_reason})'))

        return meta

    def _openlibrary_isbn(self, isbn: str) -> Optional[BookMetadata]:
        """Look up by ISBN on Open Library."""
        self.log.log(TRACE, f"Open Library ISBN lookup: {isbn}")
        resp = self._request_with_retry(
            'openlibrary', 'GET', OPENLIBRARY_SEARCH_URL,
            params={'isbn': isbn, 'fields': '*', 'limit': 1}
        )
        if resp:
            data = resp.json()
            if data.get('docs'):
                return self._parse_openlibrary(data['docs'][0])
        return None

    def _openlibrary_search(self, title: str, author: Optional[str] = None) -> Optional[BookMetadata]:
        """Search Open Library by title/author."""
        self.log.log(TRACE, f"Open Library search: title='{title}', author='{author}'")
        params = {'title': title, 'fields': '*', 'limit': 3}
        if author:
            params['author'] = author
        resp = self._request_with_retry(
            'openlibrary', 'GET', OPENLIBRARY_SEARCH_URL, params=params
        )
        if resp:
            data = resp.json()
            if data.get('docs'):
                best = self._best_match(data['docs'], title, author)
                if best:
                    return self._parse_openlibrary(best)
        return None

    def _parse_openlibrary(self, doc: dict) -> BookMetadata:
        """Parse Open Library search result into BookMetadata."""
        meta = BookMetadata()
        meta.title = doc.get('title')
        meta.subtitle = doc.get('subtitle')
        meta.authors = doc.get('author_name', [])
        meta.publisher = (doc.get('publisher') or [None])[0]
        meta.language = (doc.get('language') or [None])[0]

        # Dates
        first_pub = doc.get('first_publish_year')
        if first_pub:
            meta.original_publication_date = str(first_pub)
            if not meta.publication_date:
                meta.publication_date = str(first_pub)

        # ISBNs
        isbns = doc.get('isbn', [])
        for isbn in isbns:
            if len(isbn) == 13 and not meta.isbn_13:
                meta.isbn_13 = isbn
            elif len(isbn) == 10 and not meta.isbn_10:
                meta.isbn_10 = isbn

        # Identifiers
        meta.oclc = (doc.get('oclc') or [None])[0]
        meta.lccn = (doc.get('lccn') or [None])[0]
        ol_key = doc.get('key')
        if ol_key:
            meta.openlibrary_id = ol_key.replace('/works/', '')

        # Subjects
        meta.subjects = doc.get('subject', [])[:20]  # Cap at 20
        meta.lcc = (doc.get('lcc') or [None])[0]
        meta.ddc = (doc.get('ddc') or [None])[0]

        # Pages
        pages = doc.get('number_of_pages_median')
        if pages:
            meta.page_count = int(pages)

        # Cover
        cover_id = doc.get('cover_i')
        if cover_id:
            meta.cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"

        # Description + more subjects from Works API
        if ol_key:
            try:
                resp = self._request_with_retry(
                    'openlibrary', 'GET', f"https://openlibrary.org{ol_key}.json"
                )
                if resp:
                    work = resp.json()
                    desc = work.get('description')
                    if isinstance(desc, dict):
                        meta.description = desc.get('value', '')
                    elif isinstance(desc, str):
                        meta.description = desc
                    # Works API has richer subject data
                    if not meta.subjects:
                        meta.subjects = [s for s in work.get('subjects', [])[:20]]
                    # Also extract subject_places, subject_people, subject_times
                    for subj in work.get('subject_places', [])[:5]:
                        if subj not in meta.subjects:
                            meta.subjects.append(subj)
            except Exception:
                pass

        return meta

    def _google_books_isbn(self, isbn: str) -> Optional[BookMetadata]:
        """Look up by ISBN on Google Books."""
        self.log.log(TRACE, f"Google Books ISBN lookup: {isbn}")
        params = {'q': f'isbn:{isbn}', 'maxResults': 1}
        google_key = self._next_google_key()
        if google_key:
            params['key'] = google_key
        resp = self._request_with_retry(
            'google', 'GET', GOOGLE_BOOKS_URL, params=params
        )
        if resp:
            data = resp.json()
            if data.get('items'):
                return self._parse_google_books(data['items'][0])
        return None

    def _google_books_search(self, title: str, author: Optional[str] = None) -> Optional[BookMetadata]:
        """Search Google Books by title/author."""
        search_title = title[:100] if len(title) > 100 else title
        self.log.log(TRACE, f"Google Books search: title='{search_title}', author='{author}'")

        # For short titles, use strict intitle:"..."; for long ones, use loose matching
        if len(search_title.split()) <= 6:
            query = f'intitle:"{search_title}"'
        else:
            # Use first ~5 significant words for intitle, rest as general terms
            words = search_title.split()
            query = f'intitle:"{" ".join(words[:5])}" {" ".join(words[5:])}'

        if author:
            query += f' inauthor:"{author}"'
        params = {'q': query, 'maxResults': 3}
        google_key = self._next_google_key()
        if google_key:
            params['key'] = google_key
        resp = self._request_with_retry(
            'google', 'GET', GOOGLE_BOOKS_URL, params=params
        )
        if resp:
            data = resp.json()
            if data.get('items'):
                return self._parse_google_books(data['items'][0])

        # If author-specific search failed, try without author
        if author:
            self.log.log(TRACE, f"Google Books retry without author")
            query_no_author = f'intitle:"{search_title[:60]}"'
            params = {'q': query_no_author, 'maxResults': 3}
            google_key = self._next_google_key()
            if google_key:
                params['key'] = google_key
            resp = self._request_with_retry(
                'google', 'GET', GOOGLE_BOOKS_URL, params=params
            )
            if resp:
                data = resp.json()
                if data.get('items'):
                    return self._parse_google_books(data['items'][0])

        return None

    def _parse_google_books(self, item: dict) -> BookMetadata:
        """Parse Google Books result into BookMetadata."""
        meta = BookMetadata()
        vol = item.get('volumeInfo', {})

        meta.title = vol.get('title')
        meta.subtitle = vol.get('subtitle')
        meta.authors = vol.get('authors', [])
        meta.publisher = vol.get('publisher')
        meta.publication_date = vol.get('publishedDate')
        meta.description = vol.get('description')
        meta.page_count = vol.get('pageCount')
        meta.language = vol.get('language')
        meta.google_books_id = item.get('id')

        # Categories → genres/subjects
        categories = vol.get('categories', [])
        meta.genres = categories[:10]

        # ISBNs
        for ident in vol.get('industryIdentifiers', []):
            if ident['type'] == 'ISBN_13':
                meta.isbn_13 = ident['identifier']
            elif ident['type'] == 'ISBN_10':
                meta.isbn_10 = ident['identifier']

        # Cover
        images = vol.get('imageLinks', {})
        meta.cover_url = images.get('thumbnail') or images.get('smallThumbnail')

        return meta

    def _best_match(self, docs: list, title: str, author: Optional[str]) -> Optional[dict]:
        """Pick the best matching result from a list."""
        if not docs:
            return None

        title_lower = title.lower().strip()

        for doc in docs:
            doc_title = (doc.get('title') or '').lower().strip()
            if doc_title == title_lower:
                if author:
                    doc_authors = [a.lower() for a in doc.get('author_name', [])]
                    if any(author.lower() in a for a in doc_authors):
                        return doc
                else:
                    return doc

        # Fallback to first result
        return docs[0]


# =============================================================================
# AI TEXT ANALYSIS (Claude API)
# =============================================================================

class AIMetadataExtractor:
    """Use Claude API to analyze extracted text and determine metadata."""

    def __init__(self, logger: logging.Logger, api_key: Optional[str] = None):
        self.log = logger
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')

    def analyze(self, text: str, filename: str, existing_meta: Optional[BookMetadata] = None) -> Optional[BookMetadata]:
        """Send text to Claude for metadata extraction."""
        if not self.api_key:
            self.log.warning("No Anthropic API key available, skipping AI analysis")
            return None

        self.log.info(f"Running AI metadata extraction for {filename}")

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            existing_info = ""
            if existing_meta and existing_meta.title:
                existing_info = f"\nPartially known metadata: title='{existing_meta.title}', authors={existing_meta.authors}"

            prompt = f"""Analyze this text extracted from an ebook file named "{filename}" and extract as much metadata as possible.{existing_info}

Return a JSON object with these fields (use null for unknown):
{{
  "title": "exact book title",
  "subtitle": "subtitle if any",
  "authors": ["list of authors"],
  "editors": ["list of editors"],
  "translators": ["list of translators"],
  "illustrators": ["list of illustrators"],
  "publisher": "publisher name",
  "publication_date": "date or year",
  "original_publication_date": "original date if this is a reprint/translation",
  "edition": "edition info",
  "language": "ISO 639-1 language code",
  "description": "brief description of the book (2-3 sentences)",
  "subjects": ["subject headings"],
  "genres": ["genre classifications"],
  "series": "series name if part of a series",
  "series_index": null,
  "isbn_10": null,
  "isbn_13": null
}}

ONLY return valid JSON, no other text. Be precise with the title - use the exact title as printed.

TEXT (first ~{len(text)} characters):
{text}"""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )

            response_text = response.content[0].text.strip()
            # Clean up potential markdown code fences
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)

            data = json.loads(response_text)
            meta = BookMetadata()

            # Map JSON to BookMetadata
            simple_fields = [
                'title', 'subtitle', 'publisher', 'publication_date',
                'original_publication_date', 'edition', 'language',
                'description', 'series', 'isbn_10', 'isbn_13',
            ]
            for fld in simple_fields:
                val = data.get(fld)
                if val:
                    setattr(meta, fld, val)

            list_fields = [
                'authors', 'editors', 'translators', 'illustrators',
                'subjects', 'genres',
            ]
            for fld in list_fields:
                val = data.get(fld)
                if val and isinstance(val, list):
                    setattr(meta, fld, [v for v in val if v])

            si = data.get('series_index')
            if si is not None:
                try:
                    meta.series_index = float(si)
                except (ValueError, TypeError):
                    pass

            self.log.info(f"AI extraction successful: title='{meta.title}', authors={meta.authors}")
            return meta

        except json.JSONDecodeError as e:
            self.log.error(f"AI returned invalid JSON: {e}")
            return None
        except Exception as e:
            self.log.error(f"AI analysis failed: {e}")
            self.log.debug(traceback.format_exc())
            return None


# =============================================================================
# METADATA WRITER
# =============================================================================

class MetadataWriter:
    """Write metadata back into ebook files using calibre's ebook-meta."""

    def __init__(self, logger: logging.Logger):
        self.log = logger
        self._calibre_available = None

    def calibre_available(self) -> bool:
        if self._calibre_available is None:
            rc, _, _ = run_command(['ebook-meta', '--version'])
            self._calibre_available = (rc == 0)
            if not self._calibre_available:
                self.log.error(
                    "Calibre CLI not found! Install with: sudo apt install calibre\n"
                    "Metadata writing will be limited."
                )
        return self._calibre_available

    def write(self, filepath: str, meta: BookMetadata) -> bool:
        """Write metadata to an ebook file."""
        ext = Path(filepath).suffix.lower()

        success = False

        # Use calibre for most formats
        if self.calibre_available():
            success = self._write_calibre(filepath, meta)

        # Format-specific enrichment
        if ext == '.epub':
            epub_ok = self._write_epub_extra(filepath, meta)
            success = success or epub_ok
        elif ext == '.pdf':
            pdf_ok = self._write_pdf_meta(filepath, meta)
            success = success or pdf_ok

        return success

    def _write_calibre(self, filepath: str, meta: BookMetadata) -> bool:
        """Write metadata using calibre's ebook-meta CLI."""
        cmd = ['ebook-meta', filepath]

        if meta.title:
            cmd.extend(['--title', meta.title])
        if meta.authors:
            cmd.extend(['--authors', ' & '.join(meta.authors)])
        if meta.publisher:
            cmd.extend(['--publisher', meta.publisher])
        if meta.publication_date:
            date = meta.publication_date
            # Calibre wants ISO date
            if re.match(r'^\d{4}$', date):
                date = f'{date}-01-01'
            cmd.extend(['--date', date])
        if meta.language:
            cmd.extend(['--language', meta.language])
        if meta.description:
            cmd.extend(['--comments', meta.description])
        if meta.subjects or meta.genres or meta.tags:
            all_tags = list(set(meta.subjects + meta.genres + meta.tags))
            cmd.extend(['--tags', ', '.join(all_tags[:50])])  # Cap tags
        if meta.series:
            cmd.extend(['--series', meta.series])
        if meta.series_index is not None:
            cmd.extend(['--index', str(meta.series_index)])
        if meta.isbn_13:
            cmd.extend(['--isbn', meta.isbn_13])
        elif meta.isbn_10:
            cmd.extend(['--isbn', meta.isbn_10])

        # Build identifier string
        identifiers = []
        if meta.isbn_13:
            identifiers.append(f'isbn:{meta.isbn_13}')
        if meta.isbn_10:
            identifiers.append(f'isbn10:{meta.isbn_10}')
        if meta.gutenberg_id:
            identifiers.append(f'gutenberg:{meta.gutenberg_id}')
        if meta.google_books_id:
            identifiers.append(f'google:{meta.google_books_id}')
        if meta.openlibrary_id:
            identifiers.append(f'openlibrary:{meta.openlibrary_id}')
        if meta.asin:
            identifiers.append(f'amazon:{meta.asin}')
        if meta.doi:
            identifiers.append(f'doi:{meta.doi}')
        if meta.lccn:
            identifiers.append(f'lccn:{meta.lccn}')
        if meta.oclc:
            identifiers.append(f'oclc:{meta.oclc}')

        # ebook-meta doesn't have a direct --identifiers flag for all,
        # but we can use --identifier for calibre >= 5.x
        for ident in identifiers:
            cmd.extend(['--identifier', ident])

        # Download and embed cover image
        cover_path = None
        if meta.cover_url:
            cover_path = self._download_cover(meta.cover_url, filepath)
            if cover_path:
                cmd.extend(['--cover', cover_path])

        self.log.log(TRACE, f"Running: {' '.join(cmd)}")
        rc, out, err = run_command(cmd, timeout=60)

        # Clean up temp cover file
        if cover_path and os.path.exists(cover_path):
            try:
                os.remove(cover_path)
            except OSError:
                pass

        if rc == 0:
            self.log.log(TRACE, f"Calibre metadata written to {os.path.basename(filepath)}")
            return True
        else:
            # Some flags might not be supported in older calibre
            self.log.warning(f"Calibre ebook-meta returned {rc}: {err}")
            # Retry with minimal flags
            if '--identifier' in cmd:
                self.log.debug("Retrying without --identifier flags")
                cmd_minimal = [c for i, c in enumerate(cmd)
                               if not (cmd[max(0,i-1):i+1] == ['--identifier'] or
                                       (i > 0 and cmd[i-1] == '--identifier'))]
                # Simpler approach: rebuild without identifiers
                return self._write_calibre_minimal(filepath, meta)
            return False

    def _download_cover(self, url: str, filepath: str) -> Optional[str]:
        """Download cover image to a temp file. Returns path or None."""
        import tempfile
        try:
            import requests
            resp = requests.get(url, timeout=15, stream=True)
            resp.raise_for_status()

            # Determine extension from content type
            ct = resp.headers.get('content-type', '')
            if 'png' in ct:
                ext = '.png'
            elif 'gif' in ct:
                ext = '.gif'
            else:
                ext = '.jpg'

            # Write to temp file next to the ebook
            tmp_dir = os.path.dirname(filepath)
            fd, tmp_path = tempfile.mkstemp(suffix=ext, prefix='.cover_', dir=tmp_dir)
            with os.fdopen(fd, 'wb') as f:
                for chunk in resp.iter_content(8192):
                    f.write(chunk)

            # Verify it's a real image (at least a few KB)
            if os.path.getsize(tmp_path) < 500:
                os.remove(tmp_path)
                return None

            self.log.log(TRACE, f"Cover downloaded: {os.path.getsize(tmp_path)} bytes")
            return tmp_path
        except Exception as e:
            self.log.log(TRACE, f"Cover download failed: {e}")
            return None

    def _write_calibre_minimal(self, filepath: str, meta: BookMetadata) -> bool:
        """Minimal calibre metadata write without advanced flags."""
        cmd = ['ebook-meta', filepath]
        if meta.title:
            cmd.extend(['--title', meta.title])
        if meta.authors:
            cmd.extend(['--authors', ' & '.join(meta.authors)])
        if meta.publisher:
            cmd.extend(['--publisher', meta.publisher])
        if meta.language:
            cmd.extend(['--language', meta.language])
        if meta.description:
            cmd.extend(['--comments', meta.description])
        if meta.isbn_13:
            cmd.extend(['--isbn', meta.isbn_13])

        rc, out, err = run_command(cmd, timeout=60)
        if rc == 0:
            self.log.info(f"✓ Minimal metadata written to {os.path.basename(filepath)}")
            return True
        self.log.error(f"Even minimal calibre write failed: {err}")
        return False

    def _write_epub_extra(self, filepath: str, meta: BookMetadata) -> bool:
        """Write additional EPUB metadata directly into OPF."""
        try:
            import ebooklib
            from ebooklib import epub

            book = epub.read_epub(filepath, options={'ignore_ncx': True})

            # Add subjects
            for subject in meta.subjects:
                book.add_metadata('DC', 'subject', subject)

            # Add contributors with roles
            for editor in meta.editors:
                book.add_metadata('DC', 'contributor', editor,
                                  {'{http://www.idpf.org/2007/opf}role': 'edt'})
            for translator in meta.translators:
                book.add_metadata('DC', 'contributor', translator,
                                  {'{http://www.idpf.org/2007/opf}role': 'trl'})
            for illustrator in meta.illustrators:
                book.add_metadata('DC', 'contributor', illustrator,
                                  {'{http://www.idpf.org/2007/opf}role': 'ill'})

            # Rights
            if meta.rights:
                book.add_metadata('DC', 'rights', meta.rights)

            # Source/identifiers
            if meta.gutenberg_id:
                book.add_metadata('DC', 'source',
                                  f'https://www.gutenberg.org/ebooks/{meta.gutenberg_id}')

            epub.write_epub(filepath, book)
            self.log.log(TRACE, f"EPUB extra metadata written to {os.path.basename(filepath)}")
            return True

        except Exception as e:
            self.log.warning(f"EPUB extra metadata write failed: {e}")
            return False

    def _write_pdf_meta(self, filepath: str, meta: BookMetadata) -> bool:
        """Write PDF metadata using PyMuPDF."""
        try:
            import fitz
            doc = fitz.open(filepath)

            pdf_meta = {
                'title': meta.title or '',
                'author': ', '.join(meta.authors) if meta.authors else '',
                'subject': meta.description or '',
                'keywords': ', '.join(
                    meta.subjects[:10] + meta.genres[:5] + meta.tags[:5]
                ),
                'creator': 'EbookMetadataPipeline',
                'producer': meta.publisher or '',
            }

            doc.set_metadata(pdf_meta)
            doc.saveIncr()
            doc.close()
            self.log.log(TRACE, f"PDF metadata written to {os.path.basename(filepath)}")
            return True

        except Exception as e:
            self.log.warning(f"PDF metadata write failed: {e}")
            return False


# =============================================================================
# FILE RENAMER
# =============================================================================

class FileRenamer:
    """Rename ebook files based on metadata."""

    def __init__(self, logger: logging.Logger, dry_run: bool = False):
        self.log = logger
        self.dry_run = dry_run

    def _display_author(self, author: str) -> str:
        """Flip 'Last, First' to 'First Last' for filenames."""
        # Only flip if it's "Word, Word" (exactly one comma, and each part is 1-3 words)
        if ',' in author and author.count(',') == 1:
            parts = author.split(',', 1)
            last = parts[0].strip()
            first = parts[1].strip().rstrip('.')
            # Only flip if both parts look like name components (not "Concepts, Choices")
            if (first and last and
                len(last.split()) <= 2 and len(first.split()) <= 3 and
                first[0].isupper() and last[0].isupper()):
                return f"{first} {last}"
        return author

    def compute_new_path(self, filepath: str, meta: BookMetadata) -> Optional[str]:
        """Compute what the new filename would be without renaming."""
        if not meta.title:
            return None

        ext = Path(filepath).suffix
        directory = os.path.dirname(filepath)

        title = sanitize_filename(meta.title, max_length=120)
        year = extract_year(meta.publication_date) or extract_year(meta.original_publication_date)

        edition_part = ""
        if meta.edition:
            edition_part = f" [{sanitize_filename(meta.edition, 30)}]"
        elif meta.revision:
            edition_part = f" [rev {sanitize_filename(meta.revision, 20)}]"

        if meta.authors:
            display_authors = [self._display_author(a) for a in meta.authors]
            if len(display_authors) == 1:
                author_str = sanitize_filename(display_authors[0], 60)
            elif len(display_authors) == 2:
                author_str = sanitize_filename(
                    f"{display_authors[0]} & {display_authors[1]}", 80
                )
            else:
                author_str = sanitize_filename(
                    f"{display_authors[0]} et al.", 60
                )
        else:
            author_str = "Unknown Author"

        series_part = ""
        if meta.series:
            series_str = sanitize_filename(meta.series, 50)
            if meta.series_index is not None:
                idx = int(meta.series_index) if meta.series_index == int(meta.series_index) else meta.series_index
                series_part = f" ({series_str} #{idx})"
            else:
                series_part = f" ({series_str})"

        year_part = f" ({year})" if year else ""
        new_name = f"{title}{year_part}{edition_part}{series_part} - {author_str}{ext}"

        new_path = os.path.join(directory, new_name)
        if os.path.abspath(new_path) == os.path.abspath(filepath):
            return filepath
        return new_path

    def rename(self, filepath: str, meta: BookMetadata) -> Optional[str]:
        """Rename file based on metadata. Returns new filepath or None."""
        if not meta.title:
            self.log.warning(f"Cannot rename {filepath}: no title")
            return None

        ext = Path(filepath).suffix
        directory = os.path.dirname(filepath)

        # Build filename: Title (Year) - Author.ext
        title = sanitize_filename(meta.title, max_length=120)

        year = extract_year(meta.publication_date) or extract_year(meta.original_publication_date)

        # Edition/revision info
        edition_part = ""
        if meta.edition:
            edition_part = f" [{sanitize_filename(meta.edition, 30)}]"
        elif meta.revision:
            edition_part = f" [rev {sanitize_filename(meta.revision, 20)}]"

        # Author(s) — flip "Last, First" to "First Last" for filename
        if meta.authors:
            display_authors = [self._display_author(a) for a in meta.authors]
            if len(display_authors) == 1:
                author_str = sanitize_filename(display_authors[0], 60)
            elif len(display_authors) == 2:
                author_str = sanitize_filename(
                    f"{display_authors[0]} & {display_authors[1]}", 80
                )
            else:
                author_str = sanitize_filename(
                    f"{display_authors[0]} et al.", 60
                )
        else:
            author_str = "Unknown Author"

        # Series info
        series_part = ""
        if meta.series:
            series_str = sanitize_filename(meta.series, 50)
            if meta.series_index is not None:
                idx = int(meta.series_index) if meta.series_index == int(meta.series_index) else meta.series_index
                series_part = f" ({series_str} #{idx})"
            else:
                series_part = f" ({series_str})"

        # Assemble: Title (Year) [Edition] (Series #N) - Author.ext
        year_part = f" ({year})" if year else ""
        new_name = f"{title}{year_part}{edition_part}{series_part} - {author_str}{ext}"

        # Ensure unique
        new_path = os.path.join(directory, new_name)
        if os.path.abspath(new_path) == os.path.abspath(filepath):
            self.log.debug(f"File already correctly named: {new_name}")
            return filepath

        # Handle collision
        counter = 1
        base_new_path = new_path
        while os.path.exists(new_path):
            stem = Path(base_new_path).stem
            new_path = os.path.join(directory, f"{stem} ({counter}){ext}")
            counter += 1

        if self.dry_run:
            return new_path

        try:
            os.rename(filepath, new_path)
            self.log.log(TRACE, f"Renamed: {os.path.basename(filepath)} → {os.path.basename(new_path)}")
            return new_path
        except OSError as e:
            self.log.error(f"Rename failed: {e}")
            return None


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    """Generate processing reports."""

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.results: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def add_result(self, filepath: str, meta: BookMetadata,
                   new_path: Optional[str], elapsed: float, status: str):
        with self._lock:
            self.results.append({
                'original_path': filepath,
                'new_path': new_path,
                'title': meta.title,
                'authors': meta.authors,
                'completeness': meta.completeness_score(),
                'sources_used': meta.sources_used,
                'errors': meta.errors,
                'elapsed_seconds': round(elapsed, 2),
                'status': status,
            })

    def save(self):
        """Save reports to disk."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # JSON report
        json_path = os.path.join(self.log_dir, f'report_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'total_processed': len(self.results),
                'successful': sum(1 for r in self.results if r['status'] == 'success'),
                'failed': sum(1 for r in self.results if r['status'] == 'error'),
                'skipped': sum(1 for r in self.results if r['status'] == 'skipped'),
                'avg_completeness': (
                    sum(r['completeness'] for r in self.results) / len(self.results)
                    if self.results else 0
                ),
                'results': self.results,
            }, f, indent=2)

        # Summary stats
        print(f"\n{'='*60}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*60}")
        print(f"Total files:      {len(self.results)}")
        print(f"Successful:       {sum(1 for r in self.results if r['status'] == 'success')}")
        print(f"Failed:           {sum(1 for r in self.results if r['status'] == 'error')}")
        print(f"Skipped:          {sum(1 for r in self.results if r['status'] == 'skipped')}")
        if self.results:
            avg_comp = sum(r['completeness'] for r in self.results) / len(self.results)
            print(f"Avg completeness: {avg_comp:.1%}")
        print(f"Report saved:     {json_path}")
        print(f"{'='*60}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class EbookMetadataPipeline:
    """Main orchestrator for the ebook metadata pipeline."""

    RDF_DOWNLOAD_URL = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
    DEFAULT_RDF_DIR = os.path.expanduser("~/gutenberg-rdf")

    def __init__(self, args):
        self.args = args
        self.log_dir = args.log_dir or os.path.join(args.ebook_dir, '.metadata_logs')
        self.log = setup_logging(self.log_dir, args.verbose)
        self.stats = logging.getLogger('ebook_pipeline.stats')

        # Initialize components
        self.text_extractor = TextExtractor(self.log)
        self.embedded_extractor = EmbeddedMetadataExtractor(self.log)

        # Auto-setup RDF catalog
        rdf_path = args.rdf_catalog or self.DEFAULT_RDF_DIR
        self.rdf_catalog = None
        if not args.skip_rdf:
            self.rdf_catalog = self._setup_rdf_catalog(rdf_path)

        self.api_lookup = PublicAPILookup(self.log, args.google_api_key)  # now a list
        self.ai_extractor = AIMetadataExtractor(self.log, args.anthropic_api_key)
        self.writer = MetadataWriter(self.log)
        self.renamer = FileRenamer(self.log, args.dry_run)
        self.report = ReportGenerator(self.log_dir)

        # Processing cache
        # Fix: Save cache to /tmp if using network shares to avoid locking issues
        if os.path.exists("/tmp"):
            path_hash = hashlib.md5(args.ebook_dir.encode()).hexdigest()
            cache_path = os.path.join("/tmp", f"ebook_metadata_{path_hash}.db")
            self.log.info(f"Using cache at: {cache_path}")
        else:
            cache_path = os.path.join(args.ebook_dir, '.metadata_cache.db')

        self.cache = ProcessingCache(cache_path, self.log)
        self.force_reprocess = getattr(args, 'force', False)

    def _setup_rdf_catalog(self, rdf_dir: str) -> Optional[GutenbergRDFCatalog]:
        """Check for RDF catalog and auto-download if missing."""
        # Check common locations for extracted RDF files
        possible_epub_dirs = [
            os.path.join(rdf_dir, 'cache', 'epub'),
            os.path.join(rdf_dir, 'epub'),
            rdf_dir,
        ]

        for d in possible_epub_dirs:
            if os.path.isdir(d):
                # Verify it actually has RDF files
                sample_dirs = [x for x in os.listdir(d) if x.isdigit()][:1]
                if sample_dirs:
                    self.log.info(f"✓ RDF catalog found at {d}")
                    return GutenbergRDFCatalog(rdf_dir, self.log)

        # Not found — offer to download
        tarball = os.path.join(rdf_dir, 'rdf-files.tar.bz2')

        if os.path.exists(tarball):
            self.log.info(f"RDF tarball found at {tarball}, extracting...")
            return self._extract_rdf(rdf_dir, tarball)

        self.log.warning(f"Gutenberg RDF catalog not found at {rdf_dir}")

        if self.args.auto_download_rdf or self._confirm_download():
            return self._download_and_extract_rdf(rdf_dir)
        else:
            self.log.info("Skipping RDF catalog — Gutenberg books will use other sources")
            return None

    def _confirm_download(self) -> bool:
        """Ask user if they want to download the RDF catalog."""
        if not sys.stdin.isatty():
            self.log.info("Non-interactive mode, use --auto-download-rdf to enable auto-download")
            return False
        try:
            resp = input(
                "\nGutenberg RDF catalog not found. Download it now? (~300MB download, ~1.5GB extracted)\n"
                f"Location: {self.DEFAULT_RDF_DIR}\n"
                "[Y/n]: "
            ).strip().lower()
            return resp in ('', 'y', 'yes')
        except (EOFError, KeyboardInterrupt):
            return False

    def _download_and_extract_rdf(self, rdf_dir: str) -> Optional[GutenbergRDFCatalog]:
        """Download and extract the RDF catalog."""
        import urllib.request

        os.makedirs(rdf_dir, exist_ok=True)
        tarball = os.path.join(rdf_dir, 'rdf-files.tar.bz2')

        self.log.info(f"Downloading Gutenberg RDF catalog (~300MB)...")
        self.log.info(f"  URL: {self.RDF_DOWNLOAD_URL}")
        self.log.info(f"  Destination: {tarball}")

        try:
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                if total_size > 0:
                    pct = min(100, downloaded * 100 // total_size)
                    mb = downloaded / (1024 * 1024)
                    total_mb = total_size / (1024 * 1024)
                    print(f"\r  Downloading: {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end='', flush=True)

            urllib.request.urlretrieve(self.RDF_DOWNLOAD_URL, tarball, progress_hook)
            print()  # newline after progress
            self.log.info("✓ Download complete")

            return self._extract_rdf(rdf_dir, tarball)

        except Exception as e:
            self.log.error(f"RDF download failed: {e}")
            self.log.info("You can manually download from:")
            self.log.info(f"  wget {self.RDF_DOWNLOAD_URL} -O {tarball}")
            self.log.info(f"  cd {rdf_dir} && tar xjf rdf-files.tar.bz2")
            return None

    def _extract_rdf(self, rdf_dir: str, tarball: str) -> Optional[GutenbergRDFCatalog]:
        """Extract the RDF tarball."""
        self.log.info(f"Extracting RDF catalog (this takes a few minutes)...")
        try:
            rc, out, err = run_command(
                ['tar', 'xjf', tarball, '-C', rdf_dir],
                timeout=600  # 10 min timeout for large extraction
            )
            if rc == 0:
                self.log.info("✓ RDF catalog extracted successfully")
                return GutenbergRDFCatalog(rdf_dir, self.log)
            else:
                self.log.error(f"Extraction failed: {err}")
                return None
        except Exception as e:
            self.log.error(f"Extraction failed: {e}")
            return None

    def discover_files(self) -> List[str]:
        """Find all ebook files in the target directory."""
        self.log.info(f"Scanning {self.args.ebook_dir} for ebooks...")
        files = []
        for root, dirs, filenames in os.walk(self.args.ebook_dir):
            # Skip hidden dirs and log dirs
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in filenames:
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, fname))

        self.log.info(f"Found {len(files)} ebook files")
        return sorted(files)

    def process_file(self, filepath: str) -> Tuple[BookMetadata, Optional[str], str]:
        """Process a single ebook file through the full pipeline."""
        start_time = time.time()
        basename = os.path.basename(filepath)
        ext = Path(filepath).suffix.lower()

        self.log.info(f"")
        self.log.info(f"\033[36m┌───\033[0m \033[1m📖 {basename}\033[0m")
        self.log.info(f"\033[36m│\033[0m  \033[90m{ext}  •  {os.path.getsize(filepath) / 1024:.0f} KB\033[0m")

        meta = BookMetadata()
        meta.source_file = filepath
        meta.source_format = ext
        meta.file_hash = file_sha256(filepath)

        # Track what was originally in the file
        original_meta = BookMetadata()

        # ─── Stage 1: Gutenberg RDF ─────────────────────────
        pg_id = detect_gutenberg_id(filepath)
        if pg_id and self.rdf_catalog:
            rdf_meta = self.rdf_catalog.lookup(pg_id)
            if rdf_meta:
                meta.merge(rdf_meta, "gutenberg-rdf")
                self.log.debug(f"\033[36m│\033[0m  \033[36m✓ RDF catalog match for PG#{pg_id}\033[0m")

        # ─── Stage 2: Embedded metadata ─────────────────────
        try:
            embedded = self.embedded_extractor.extract(filepath)
            # Copy embedded for FOUND section
            for fld in embedded.__dataclass_fields__:
                val = getattr(embedded, fld)
                if val and val != [] and val != 0 and val != 0.0:
                    try:
                        if isinstance(val, list):
                            setattr(original_meta, fld, list(val))
                        else:
                            setattr(original_meta, fld, val)
                    except Exception:
                        pass
            meta.merge(embedded, "embedded")
        except Exception as e:
            self.log.warning(f"\033[36m│\033[0m  \033[33m✗ Embedded extraction failed: {e}\033[0m")
            meta.errors.append(f"embedded_extraction: {e}")

        # ─── Stage 2.5: Filename parsing (sanity check / fallback) ────
        fn_meta = parse_filename_metadata(filepath)

        # Normalize whitespace-only metadata fields (common in PDFs)
        # e.g. title=" " or author=" " should be treated as empty
        if meta.title and not meta.title.strip():
            meta.title = None
        if meta.authors:
            meta.authors = [a for a in meta.authors if a and a.strip()]

        # Clean embedded author strings (split semicolons, remove artifacts)
        if meta.authors:
            expanded = []
            for a in meta.authors:
                if ';' in a:
                    expanded.extend([s.strip() for s in a.split(';') if s.strip()])
                else:
                    expanded.append(a)
            meta.authors = [clean_author_string(a) for a in expanded]
            meta.authors = [a for a in meta.authors if a]  # Remove empties

        if fn_meta.title:
            # Check if embedded metadata is clearly garbage
            if meta.title and meta.authors:
                t = meta.title.lower().strip()
                a = ' '.join(meta.authors).lower() if meta.authors else ''
                embedded_looks_garbage = (
                    # Title looks like a filename (e.g. "out.jpg", "Katz2.indd", "doc.pdf")
                    bool(re.search(r'\.\w{2,4}$', meta.title)) or
                    # Title is a generic section/page name from PDF structure
                    t in ('frontmatter', 'prelims', 'preface', 'contents',
                          'table of contents', 'copyright', 'copyright page',
                          'title page', 'cover', 'half title', 'halftitle') or
                    bool(re.match(r'^module\s', t)) or  # "module tem-1" etc.
                    bool(re.match(r'^chapter\s*\d', t)) or  # "Chapter 1" as title
                    # Title contains app artifacts
                    'microsoft word' in t or 'untitled' in t or '.indd' in t or
                    'adobe' in t or 'acrobat' in t or
                    # Title is a single word matching an author surname from filename
                    (len(meta.title.split()) == 1 and fn_meta.authors and
                     meta.title.lower() in [a_fn.split()[-1].lower() for a_fn in fn_meta.authors if a_fn]) or
                    # Title starts with "Surname - ..." matching filename author
                    (' - ' in meta.title and fn_meta.authors and
                     meta.title.split(' - ')[0].strip().lower() in
                     [a_fn.split()[-1].lower() for a_fn in fn_meta.authors if a_fn]) or
                    # Date is clearly invalid (year 0101, 0001, etc.)
                    (meta.publication_date and re.match(r'^0\d{3}', str(meta.publication_date))) or
                    # Author contains publisher/year pattern like "(CRC, 2007)"
                    bool(re.search(r'\(\w+,\s*\d{4}\)', a)) or
                    # Title contains semicolons with "Last, First" name patterns
                    (';' in meta.title and
                     len(re.findall(r'\w+,\s*\w+', meta.title.split(' - ')[0] if ' - ' in meta.title else meta.title)) >= 2) or
                    # Author is a single "word" that looks like a book title (3+ words with title keywords)
                    (meta.authors and len(meta.authors) == 1 and
                     len(meta.authors[0].split()) >= 3 and
                     len(set(meta.authors[0].lower().split()) &
                         {'concepts', 'choices', 'introduction', 'fundamentals',
                          'programming', 'analysis', 'guide', 'principles',
                          'handbook', 'mastering', 'learning', 'engineering',
                          'edition', 'comprehensive', 'practical', 'deploy',
                          'create', 'manage', 'flexible', 'technology',
                          'design', 'systems', 'digital', 'complete',
                          'electronics', 'circuits', 'communications'}) >= 1)
                )
                if embedded_looks_garbage and fn_meta.title:
                    self.log.info(f"\033[36m│\033[0m  \033[33m⚠ Embedded metadata looks suspect — using filename\033[0m")

                    # Check if embedded author looks like a real person name
                    # (has initials, multiple words, periods — don't replace with just a surname)
                    embedded_author_looks_real = False
                    if meta.authors and len(meta.authors) >= 1:
                        a0 = meta.authors[0]
                        title_indicator_words = {
                            'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for',
                            'guide', 'introduction', 'fundamentals', 'analysis',
                            'programming', 'complete', 'design', 'circuits',
                            'technology', 'communications', 'electronics',
                            'systems', 'digital', 'engineering', 'comprehensive',
                            'practical', 'principles', 'handbook', 'secrets',
                        }
                        embedded_author_looks_real = (
                            # Has initials like "J." or "Stuart R."
                            bool(re.search(r'\b[A-Z]\.\s', a0)) or
                            # Has 2+ words that look like a name (no title words)
                            (len(a0.split()) >= 2 and
                             len(set(a0.lower().split()) & title_indicator_words) == 0)
                        )

                    # Always replace title if it's clearly garbage
                    title_is_garbage = (
                        bool(re.search(r'\.\w{2,4}$', meta.title)) or
                        t in ('frontmatter', 'prelims', 'preface', 'contents',
                              'table of contents', 'copyright', 'copyright page',
                              'title page', 'cover', 'half title', 'halftitle') or
                        bool(re.match(r'^module\s', t)) or
                        bool(re.match(r'^chapter\s*\d', t)) or
                        'microsoft word' in t or 'untitled' in t or '.indd' in t or
                        'adobe' in t or 'acrobat' in t or
                        (len(meta.title.split()) == 1 and fn_meta.authors and
                         meta.title.lower() in [x.split()[-1].lower() for x in fn_meta.authors if x]) or
                        (';' in meta.title) or
                        # Title starts with "Surname - ..." → use just the title part after dash
                        (' - ' in meta.title and fn_meta.authors and
                         meta.title.split(' - ')[0].strip().lower() in
                         [x.split()[-1].lower() for x in fn_meta.authors if x])
                    )
                    if title_is_garbage:
                        # Special case: "Author1;Author2;Author3 - Real Title"
                        # Extract the real title AND the authors from the embedded title
                        if ';' in meta.title and ' - ' in meta.title:
                            # Split on last " - " to separate authors from title
                            parts = meta.title.rsplit(' - ', 1)
                            semicolon_part = parts[0].strip()
                            real_title = parts[1].strip() if len(parts) > 1 else None
                            # If the semicolon part has name patterns, use it as authors
                            if (real_title and len(real_title) >= 3
                                    and len(re.findall(r'\w+', semicolon_part.split(';')[0])) <= 6):
                                meta.title = real_title
                                # Extract authors from semicolon-delimited part
                                extracted_authors = [s.strip() for s in semicolon_part.split(';') if s.strip()]
                                if extracted_authors:
                                    meta.authors = extracted_authors
                                    self.log.log(TRACE, f"Extracted authors from embedded title: {meta.authors}")
                            else:
                                meta.title = fn_meta.title
                        # "Surname - Real Title" pattern
                        elif ' - ' in meta.title and fn_meta.authors:
                            parts = meta.title.split(' - ', 1)
                            if parts[0].strip().lower() in [x.split()[-1].lower() for x in fn_meta.authors if x]:
                                meta.title = parts[1].strip()
                            else:
                                meta.title = fn_meta.title
                        else:
                            meta.title = fn_meta.title

                    # Replace authors only if they also look bad, or filename has better ones
                    if not embedded_author_looks_real and fn_meta.authors:
                        meta.authors = fn_meta.authors

                    if fn_meta.publisher:
                        meta.publisher = fn_meta.publisher

                    # Fix or fill date
                    if fn_meta.publication_date:
                        meta.publication_date = fn_meta.publication_date
                    elif meta.publication_date and re.match(r'^0\d{3}', str(meta.publication_date)):
                        meta.publication_date = None

                    meta.processing_notes.append("corrected-from-filename")

            # Fill gaps from filename (never override existing good data)
            if not meta.title and fn_meta.title:
                meta.title = fn_meta.title
                meta.processing_notes.append("title-from-filename")
            if not meta.authors and fn_meta.authors:
                meta.authors = fn_meta.authors
                meta.processing_notes.append("authors-from-filename")
            if not meta.publisher and fn_meta.publisher:
                meta.publisher = fn_meta.publisher
            if not meta.publication_date and fn_meta.publication_date:
                meta.publication_date = fn_meta.publication_date
            if not meta.edition and fn_meta.edition:
                meta.edition = fn_meta.edition

        # Always clean and deduplicate authors
        if meta.authors:
            meta.authors = [clean_author_string(a) for a in meta.authors]
            meta.authors = [a for a in meta.authors if a]
            meta.authors = deduplicate_authors(meta.authors)

        # ─── Stage 3: Public APIs ───────────────────────────
        # Always try APIs if important fields are missing, not just below a threshold
        needs_api = False
        if not self.args.skip_api:
            missing_important = []
            if not meta.subjects and not meta.genres:
                missing_important.append('subjects')
            if not meta.language:
                missing_important.append('language')
            if not meta.description:
                missing_important.append('description')
            if not meta.page_count:
                missing_important.append('pages')
            if not meta.cover_url:
                missing_important.append('cover')
            if not meta.isbn_13 and not meta.isbn_10:
                missing_important.append('ISBN')
            if not meta.publisher:
                missing_important.append('publisher')
            if not meta.lcc and not meta.ddc:
                missing_important.append('classification')
            needs_api = len(missing_important) >= 2  # At least 2 important fields missing

        if needs_api:
            self.log.info(f"\033[36m│\033[0m  \033[36m↻ Searching APIs...\033[0m")
            try:
                search_author = meta.authors[0] if meta.authors else None
                api_meta = self.api_lookup.search(
                    title=meta.title,
                    author=search_author,
                    isbn=meta.isbn_13 or meta.isbn_10,
                    current_completeness=meta.completeness_score(),
                )
                meta.merge(api_meta, "public-api")
                # Propagate individual API source names
                for src in api_meta.sources_used:
                    if src not in meta.sources_used:
                        meta.sources_used.append(src)

                # Display per-API diagnostics
                for api_name, status, detail in getattr(api_meta, '_api_diagnostics', []):
                    if status == 'success':
                        self.log.info(f"\033[36m│\033[0m    \033[32m✓ {api_name}: {detail}\033[0m")
                    elif status == 'skipped':
                        self.log.info(f"\033[36m│\033[0m    \033[90m⊘ {api_name}: skipped ({detail})\033[0m")
                    elif status == 'no_match':
                        self.log.info(f"\033[36m│\033[0m    \033[33m✗ {api_name}: {detail}\033[0m")
                    elif status == 'error':
                        self.log.info(f"\033[36m│\033[0m    \033[31m✗ {api_name}: {detail}\033[0m")
            except Exception as e:
                self.log.warning(f"\033[36m│\033[0m  \033[33m✗ API lookup failed: {e}\033[0m")
                meta.errors.append(f"api_lookup: {e}")

        # ─── Stage 4: AI Analysis (last resort) ─────────────
        if meta.completeness_score() < self.args.ai_threshold and not self.args.skip_ai:
            self.log.info(f"\033[36m│\033[0m  \033[35m↻ AI analysis...\033[0m")
            text = self.text_extractor.extract(filepath)
            if text and len(text.strip()) > 100:
                try:
                    ai_meta = self.ai_extractor.analyze(text, basename, meta)
                    if ai_meta:
                        meta.merge(ai_meta, "ai-claude")
                except Exception as e:
                    self.log.warning(f"\033[36m│\033[0m  \033[33m✗ AI analysis failed: {e}\033[0m")
                    meta.errors.append(f"ai_analysis: {e}")

        # ─── Stage 4.5: Genre / Subject Inference ────────────
        if not meta.genres or not meta.subjects:
            inferred_genres, inferred_subjects = infer_genres_subjects(meta)
            if inferred_genres and not meta.genres:
                meta.genres = inferred_genres
                if "inferred" not in meta.sources_used:
                    meta.sources_used.append("inferred")
            if inferred_subjects and not meta.subjects:
                meta.subjects = inferred_subjects
                if "inferred" not in meta.sources_used:
                    meta.sources_used.append("inferred")

        # ═══════════════════════════════════════════════════════
        # Final cleanup: split compound authors, clean, and dedup
        # ═══════════════════════════════════════════════════════
        if meta.authors:
            # First: split any semicolon-delimited author strings
            expanded = []
            for a in meta.authors:
                if ';' in a:
                    expanded.extend([s.strip() for s in a.split(';') if s.strip()])
                else:
                    expanded.append(a)
            meta.authors = expanded

            meta.authors = [clean_author_string(a) for a in meta.authors]
            meta.authors = [a for a in meta.authors if a]
            meta.authors = deduplicate_authors(meta.authors)

            # Remove author entries that look like book titles (not person names)
            # This catches cases where swapped PDF metadata leaves the title in the author list
            if meta.title and len(meta.authors) > 0:
                title_words = set(re.sub(r'[^\w\s]', '', meta.title.lower()).split())
                # Words that indicate a title/subject, not a person name
                title_indicator_words = {
                    'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'with',
                    'guide', 'introduction', 'fundamentals', 'analysis', 'handbook',
                    'programming', 'complete', 'design', 'circuits', 'systems',
                    'digital', 'engineering', 'comprehensive', 'practical',
                    'principles', 'electronics', 'technology', 'communications',
                    'edition', 'second', 'third', 'fourth', 'fifth',
                    'controllers', 'techniques', 'devices', 'applications',
                    'concepts', 'methods', 'advanced', 'modern', 'embedded',
                    'construction', 'equipment', 'projects', 'outlines',
                }
                cleaned_authors = []
                for author in meta.authors:
                    author_lower = re.sub(r'[^\w\s]', '', author.lower())
                    author_words = set(author_lower.split())
                    # Check: does this "author" share most words with the title?
                    overlap = author_words & title_words
                    is_title_text = (
                        len(author_words) >= 3 and
                        len(overlap) >= len(author_words) * 0.6
                    )
                    # Check: does this "author" have lots of title-indicator words?
                    has_title_words = (
                        len(author_words) >= 3 and
                        len(author_words & title_indicator_words) >= 2
                    )
                    if is_title_text or has_title_words:
                        self.log.log(TRACE, f"Removed title-like author entry: '{author}'")
                    else:
                        cleaned_authors.append(author)
                # Only apply if we still have at least one author left
                if cleaned_authors:
                    meta.authors = cleaned_authors

        # ═══════════════════════════════════════════════════════
        # Final safety net: if we STILL have no title, force filename parse
        # ═══════════════════════════════════════════════════════
        if not meta.title or not meta.title.strip():
            fn_fallback = parse_filename_metadata(filepath)
            if fn_fallback.title:
                meta.title = fn_fallback.title
                meta.processing_notes.append("title-from-filename-fallback")
                self.log.info(f"\033[36m│\033[0m  \033[33m⚠ No title after all stages — using filename\033[0m")
            if not meta.authors and fn_fallback.authors:
                meta.authors = fn_fallback.authors
                meta.processing_notes.append("authors-from-filename-fallback")
            if not meta.publisher and fn_fallback.publisher:
                meta.publisher = fn_fallback.publisher
            if not meta.publication_date and fn_fallback.publication_date:
                meta.publication_date = fn_fallback.publication_date

        # ═══════════════════════════════════════════════════════
        # Final title cleanup (remove artifacts)
        # ═══════════════════════════════════════════════════════
        if meta.title:
            meta.title = clean_title(meta.title)

        # ═══════════════════════════════════════════════════════
        # Compute rename early so we can show it at the top
        # ═══════════════════════════════════════════════════════
        new_path = filepath
        new_basename = None
        if not self.args.skip_rename:
            renamed = self.renamer.compute_new_path(filepath, meta)
            if renamed and renamed != filepath:
                new_basename = os.path.basename(renamed)

        # ═══════════════════════════════════════════════════════
        # Display: rename at top, then FOUND, then CHANGES
        # ═══════════════════════════════════════════════════════
        if new_basename:
            self.log.info(f"\033[36m│\033[0m  \033[36m{'→' if not self.args.dry_run else '⟶'} {new_basename}\033[0m")

        # FOUND IN FILE
        # ═══════════════════════════════════════════════════════
        self.log.info(f"\033[36m│\033[0m")
        self.log.info(f"\033[36m│\033[0m  \033[1;90m─── FOUND IN FILE ───\033[0m")
        found_fields = []
        if original_meta.title:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Title:       {original_meta.title}\033[0m")
        if original_meta.authors:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Authors:     {', '.join(original_meta.authors)}\033[0m")
        if original_meta.publisher:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Publisher:   {original_meta.publisher}\033[0m")
        if original_meta.publication_date:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Date:        {original_meta.publication_date}\033[0m")
        if original_meta.language:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Language:    {original_meta.language}\033[0m")
        if original_meta.isbn_13 or original_meta.isbn_10:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  ISBN:        {original_meta.isbn_13 or original_meta.isbn_10}\033[0m")
        if original_meta.subjects:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Subjects:    {', '.join(original_meta.subjects[:5])}\033[0m")
        if original_meta.genres:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Genres:      {', '.join(original_meta.genres[:5])}\033[0m")
        if original_meta.tags:
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Tags:        {', '.join(original_meta.tags[:5])}\033[0m")
        if original_meta.description:
            desc = re.sub(r'<[^>]+>', '', original_meta.description)[:80].strip()
            found_fields.append(f"\033[36m│\033[0m  \033[90m  Description: {desc}...\033[0m")

        if found_fields:
            for line in found_fields:
                self.log.info(line)
        else:
            self.log.info(f"\033[36m│\033[0m  \033[90m  (no embedded metadata)\033[0m")

        # ═══════════════════════════════════════════════════════
        # CHANGES (only show new or modified fields)
        # ═══════════════════════════════════════════════════════
        changes_lines = []

        def check_change(label: str, old_val, new_val, is_list=False):
            """Only collect fields that are new or modified."""
            if new_val is None or new_val == [] or new_val == 0:
                return
            if is_list and isinstance(new_val, list):
                display = ', '.join(str(x) for x in new_val[:8])
                if len(new_val) > 8:
                    display += f" (+{len(new_val) - 8} more)"
            else:
                display = str(new_val)
                if len(display) > 120:
                    display = re.sub(r'<[^>]+>', '', display)[:120] + "..."

            old_empty = (old_val is None or old_val == [] or old_val == 0)
            same = (old_val == new_val) or (is_list and not old_empty and set(old_val or []) == set(new_val or []))

            if same:
                return  # Skip unchanged
            elif old_empty:
                changes_lines.append(f"\033[36m│\033[0m  \033[32m+ {label:<14}{display}\033[0m")
            else:
                changes_lines.append(f"\033[36m│\033[0m  \033[33m✎ {label:<14}{display}\033[0m")
                changes_lines.append(f"\033[36m│\033[0m  \033[90m  {'was:':<14}{old_val if not is_list else ', '.join(str(x) for x in (old_val or [])[:5])}\033[0m")

        check_change("Title", original_meta.title, meta.title)
        check_change("Subtitle", original_meta.subtitle, meta.subtitle)
        check_change("Authors", original_meta.authors, meta.authors, True)
        check_change("Editors", original_meta.editors, meta.editors, True)
        check_change("Translators", original_meta.translators, meta.translators, True)
        check_change("Illustrators", original_meta.illustrators, meta.illustrators, True)
        check_change("Publisher", original_meta.publisher, meta.publisher)
        check_change("Date", original_meta.publication_date, meta.publication_date)
        check_change("Orig. Date", original_meta.original_publication_date, meta.original_publication_date)
        check_change("Edition", original_meta.edition, meta.edition)
        check_change("Language", original_meta.language, meta.language)
        check_change("ISBN-13", original_meta.isbn_13, meta.isbn_13)
        check_change("ISBN-10", original_meta.isbn_10, meta.isbn_10)
        if meta.gutenberg_id:
            check_change("Gutenberg", None, f"PG#{meta.gutenberg_id}")
        check_change("ASIN", original_meta.asin, meta.asin)
        if meta.series:
            s = meta.series
            if meta.series_index is not None:
                idx = int(meta.series_index) if meta.series_index == int(meta.series_index) else meta.series_index
                s += f" #{idx}"
            check_change("Series", original_meta.series, s)
        check_change("Subjects", original_meta.subjects, meta.subjects, True)
        check_change("Genres", original_meta.genres, meta.genres, True)
        check_change("Tags", original_meta.tags, meta.tags, True)
        check_change("LCC", original_meta.lcc, meta.lcc)
        check_change("DDC", original_meta.ddc, meta.ddc)
        check_change("Pages", original_meta.page_count, meta.page_count)
        check_change("Rights", original_meta.rights, meta.rights)
        check_change("Description", original_meta.description, meta.description)
        # Show cover source nicely (domain only, not full URL)
        if meta.cover_url and not original_meta.cover_url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(meta.cover_url).netloc or meta.cover_url[:40]
            except Exception:
                domain = meta.cover_url[:40]
            check_change("Cover", None, f"✓ {domain}")

        if changes_lines:
            self.log.info(f"\033[36m│\033[0m")
            dry_label = " (DRY RUN)" if self.args.dry_run else ""
            self.log.info(f"\033[36m│\033[0m  \033[1;32m─── CHANGES{dry_label} ───\033[0m")
            for line in changes_lines:
                self.log.info(line)
        else:
            self.log.info(f"\033[36m│\033[0m")
            self.log.info(f"\033[36m│\033[0m  \033[32m✓ No changes needed\033[0m")

        # ─── Stage 5: Write metadata ────────────────────────
        if self.args.dry_run:
            pass
        elif not self.args.skip_write:
            try:
                self.writer.write(filepath, meta)
            except Exception as e:
                self.log.error(f"\033[36m│\033[0m  \033[31m✗ Write failed: {e}\033[0m")
                meta.errors.append(f"write: {e}")

        # ─── Stage 6: Execute rename ────────────────────────
        if not self.args.skip_rename and new_basename:
            renamed = self.renamer.rename(filepath, meta)
            if renamed and renamed != filepath:
                new_path = renamed

        # ─── Footer ─────────────────────────────────────────
        elapsed = time.time() - start_time
        meta.confidence_score = meta.completeness_score()
        status = 'success' if meta.title else 'error'

        score = meta.completeness_score()
        score_color = '\033[32m' if score >= 0.7 else ('\033[33m' if score >= 0.5 else '\033[31m')
        icon = '✅' if status == 'success' else '❌'

        # Build source checkmarks (only show what was actually used)
        sources = meta.sources_used
        source_parts = []
        if any('gutenberg' in s for s in sources):
            source_parts.append('\033[32m☑\033[90m Catalog')
        if 'embedded' in sources:
            source_parts.append('\033[32m☑\033[90m Embedded')
        if any('openlibrary' in s for s in sources):
            source_parts.append('\033[32m☑\033[90m OpenLibrary')
        if any('google' in s for s in sources):
            source_parts.append('\033[32m☑\033[90m Google')
        if any('ai' in s for s in sources):
            source_parts.append('\033[35m☑\033[90m AI')
        sources_display = '  '.join(source_parts) if source_parts else '\033[90mnone'

        # Show what's missing (only important fields)
        missing = []
        if not meta.title: missing.append('title')
        if not meta.authors: missing.append('authors')
        if not meta.publisher: missing.append('publisher')
        if not meta.publication_date and not meta.original_publication_date: missing.append('date')
        if not meta.isbn_13 and not meta.isbn_10: missing.append('ISBN')
        if not meta.language: missing.append('language')
        if not meta.subjects and not meta.genres: missing.append('subjects/genres')
        if not meta.description: missing.append('description')
        if not meta.page_count: missing.append('pages')
        if not meta.lcc and not meta.ddc: missing.append('classification')
        if not meta.cover_url: missing.append('cover')

        self.log.info(f"\033[36m│\033[0m")
        self.log.info(f"\033[36m│\033[0m  {icon} {score_color}{score:.0%} complete\033[0m  \033[90m•  {elapsed:.1f}s\033[0m  •  {sources_display}\033[0m")
        if missing:
            self.log.info(f"\033[36m│\033[0m  \033[90mMissing: {', '.join(missing)}\033[0m")
        if meta.errors:
            self.log.info(f"\033[36m│\033[0m  \033[31mErrors: {'; '.join(meta.errors)}\033[0m")
        self.log.info(f"\033[36m└{'─' * 60}\033[0m")

        # Stats JSON line (file log only)
        self.stats.info(json.dumps({
            'file': filepath, 'status': status, 'title': meta.title,
            'completeness': score, 'sources': meta.sources_used,
            'elapsed': round(elapsed, 2),
        }))

        return meta, new_path, status


    def run(self):
        """Run the full pipeline."""
        self.log.info("")
        self.log.info("╔════════════════════════════════════════════════════════════╗")
        self.log.info("║                       EBOOK METADATA PIPELINE              ║")
        self.log.info("╚════════════════════════════════════════════════════════════╝")
        self.log.info(f"  Target:       {self.args.ebook_dir}")
        self.log.info(f"  RDF Catalog:  {'✓ Loaded' if self.rdf_catalog else '✗ Not available'}")
        self.log.info(f"  Dry run:      {self.args.dry_run}")
        self.log.info(f"  Skip API:     {self.args.skip_api}")
        key_count = len(self.args.google_api_key) if self.args.google_api_key else 0
        google_status = f"✓ {key_count} key(s)" if key_count > 0 else "✗ None (rate limits apply)"
        self.log.info(f"  Google API:   {google_status}")
        self.log.info(f"  Skip AI:      {self.args.skip_ai}")
        self.log.info(f"  Skip write:   {self.args.skip_write}")
        self.log.info(f"  Skip rename:  {self.args.skip_rename}")
        threads = getattr(self.args, 'threads', 1) or 1
        if threads > 1:
            self.log.info(f"  Threads:      {threads}")
        self.log.info(f"  Log dir:      {self.log_dir}")
        self.log.info("")

        files = self.discover_files()
        if not files:
            self.log.warning("No ebook files found!")
            return

        # Deduplicate files by absolute path AND by (filename + size) to catch
        # identical files in different directories or via symlinks
        seen_paths = set()
        seen_file_keys = set()  # (basename, size) pairs
        unique_files = []
        dup_count = 0
        for f in files:
            abspath = os.path.abspath(f)
            # Resolve symlinks for path dedup
            realpath = os.path.realpath(f)
            if abspath in seen_paths or realpath in seen_paths:
                self.log.debug(f"Skipping duplicate path: {f}")
                dup_count += 1
                continue
            # Also dedup by (filename, size) — catches copies in different dirs
            try:
                fkey = (os.path.basename(f), os.path.getsize(f))
                if fkey in seen_file_keys:
                    self.log.debug(f"Skipping duplicate file (same name+size): {f}")
                    dup_count += 1
                    continue
                seen_file_keys.add(fkey)
            except OSError:
                pass
            seen_paths.add(abspath)
            seen_paths.add(realpath)
            unique_files.append(f)
        if dup_count > 0:
            self.log.info(f"Skipped {dup_count} duplicate file(s)")
        files = unique_files

        # Process with optional limit
        limit = self.args.limit or len(files)
        files = files[:limit]

        # Cache check: skip already-processed files
        if not self.force_reprocess:
            to_process = []
            cached_count = 0
            for f in files:
                cached = self.cache.is_processed(f)
                if cached:
                    cached_count += 1
                    self.log.debug(f"Cache hit ({cached['completeness']*100:.0f}%): {os.path.basename(f)}")
                else:
                    to_process.append(f)
            if cached_count > 0:
                self.log.info(f"  ⚡ Cache: {cached_count} already processed, {len(to_process)} remaining")
            files = to_process
        else:
            self.log.info(f"  ⚡ Cache: --force enabled, reprocessing all files")

        if not files:
            self.log.info("All files already processed! Use --force to reprocess.")
            cache_stats = self.cache.get_stats()
            self.log.info(f"  Cache: {cache_stats['total']} books, "
                          f"avg {cache_stats['avg_completeness']*100:.0f}% complete")
            return

        self.log.info(f"Processing {len(files)} files...")

        if threads <= 1:
            self._run_sequential(files)
        else:
            self._run_threaded(files, threads)

        self.report.save()

        # Show cache stats
        cache_stats = self.cache.get_stats()
        self.log.info(f"\n  📊 Cache: {cache_stats['total']} books tracked, "
                      f"avg {cache_stats['avg_completeness']*100:.0f}% complete")

    def _run_sequential(self, files: list):
        """Process files one at a time (original behavior)."""
        for i, filepath in enumerate(files, 1):
            try:
                self.log.info(f"\n\033[36m{'━' * 62}\033[0m")
                self.log.info(f"  [{i}/{len(files)}] ({i/len(files)*100:.0f}%)")
                meta, new_path, status = self.process_file(filepath)
                self.report.add_result(filepath, meta, new_path, 0, status)
                self.cache.mark_processed(filepath, meta, status)
            except KeyboardInterrupt:
                self.log.warning("\nInterrupted by user!")
                break
            except Exception as e:
                self.log.error(f"FATAL error processing {filepath}: {e}")
                self.log.debug(traceback.format_exc())
                empty_meta = BookMetadata(source_file=filepath)
                empty_meta.errors.append(str(e))
                self.report.add_result(filepath, empty_meta, None, 0, 'error')
                self.cache.mark_processed(filepath, empty_meta, 'error')

    def _run_threaded(self, files: list, max_workers: int):
        """Process files in parallel using thread pool."""
        print_lock = threading.Lock()
        completed = [0]
        total = len(files)

        # Find and temporarily suppress the console handler
        console_handler = None
        for h in self.log.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                console_handler = h
                break

        def process_one(index: int, filepath: str):
            """Process a single file with buffered output."""
            buf = BufferedLogHandler(thread_id=threading.get_ident())
            self.log.addHandler(buf)

            try:
                self.log.info(f"\n\033[36m{'━' * 62}\033[0m")
                self.log.info(f"  [{index}/{total}] ({index/total*100:.0f}%)")
                meta, new_path, status = self.process_file(filepath)
                return (filepath, meta, new_path, status, buf.get_output(), None)
            except Exception as e:
                self.log.error(f"FATAL error processing {filepath}: {e}")
                empty_meta = BookMetadata(source_file=filepath)
                empty_meta.errors.append(str(e))
                return (filepath, empty_meta, None, 'error', buf.get_output(), e)
            finally:
                self.log.removeHandler(buf)

        # Temporarily remove console handler to avoid double output
        if console_handler:
            self.log.removeHandler(console_handler)

        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i, filepath in enumerate(files, 1):
                    future = executor.submit(process_one, i, filepath)
                    futures[future] = (i, filepath)

                for future in as_completed(futures):
                    try:
                        filepath, meta, new_path, status, output, error = future.result()
                        with print_lock:
                            completed[0] += 1
                            # Print buffered output
                            print(output)
                            sys.stdout.flush()
                        self.report.add_result(filepath, meta, new_path, 0, status)
                        self.cache.mark_processed(filepath, meta, status)
                    except KeyboardInterrupt:
                        self.log.warning("\nInterrupted! Cancelling remaining tasks...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as e:
                        idx, fp = futures[future]
                        with print_lock:
                            print(f"\033[31mFATAL error on {fp}: {e}\033[0m")
                        empty_meta = BookMetadata(source_file=fp)
                        empty_meta.errors.append(str(e))
                        self.report.add_result(fp, empty_meta, None, 0, 'error')
                        self.cache.mark_processed(fp, empty_meta, 'error')
        finally:
            # Restore console handler
            if console_handler:
                self.log.addHandler(console_handler)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive ebook metadata extraction, enrichment, and renaming pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with RDF catalog
  python ebook_metadata_pipeline.py /data/ebooks --rdf-catalog /data/gutenberg-rdf

  # Dry run (no changes, just preview)
  python ebook_metadata_pipeline.py /data/ebooks --dry-run --verbose

  # Skip AI and API lookups (RDF + embedded only)
  python ebook_metadata_pipeline.py /data/ebooks --rdf-catalog /data/rdf --skip-api --skip-ai

  # Process only first 10 files for testing
  python ebook_metadata_pipeline.py /data/ebooks --limit 10 --verbose

  # Full pipeline with all sources
  python ebook_metadata_pipeline.py /data/ebooks \\
      --rdf-catalog /data/gutenberg-rdf \\
      --anthropic-api-key sk-ant-... \\
      --verbose
        """,
    )

    parser.add_argument('ebook_dir', help='Directory containing ebook files')
    parser.add_argument('--rdf-catalog', help='Path to Gutenberg RDF catalog directory')
    parser.add_argument('--anthropic-api-key', help='Anthropic API key (or set ANTHROPIC_API_KEY env)')
    parser.add_argument('--google-api-key', action='append',
                        default=None,
                        help='Google Books API key (repeat flag or comma-separate for multiple keys)')
    parser.add_argument('--log-dir', help='Directory for logs (default: <ebook_dir>/.metadata_logs)')
    parser.add_argument('--limit', type=int, help='Process only first N files')
    parser.add_argument('--dry-run', action='store_true', help='Preview changes without modifying files')
    parser.add_argument('--skip-api', action='store_true', help='Skip public API lookups')
    parser.add_argument('--skip-ai', action='store_true', help='Skip AI text analysis')
    parser.add_argument('--ai-threshold', type=float, default=0.4,
                        help='Completeness threshold below which AI kicks in (0.0-1.0, default: 0.4)')
    parser.add_argument('--api-threshold', type=float, default=0.7,
                        help='Completeness threshold below which API lookups kick in (0.0-1.0, default: 0.7)')
    parser.add_argument('--skip-write', action='store_true', help='Skip metadata writing to files')
    parser.add_argument('--skip-rename', action='store_true', help='Skip file renaming')
    parser.add_argument('--skip-rdf', action='store_true', help='Skip Gutenberg RDF catalog entirely')
    parser.add_argument('--auto-download-rdf', action='store_true',
                        help='Auto-download RDF catalog without prompting if missing')
    parser.add_argument('--threads', '-t', type=int, default=1,
                        help='Number of parallel processing threads (default: 1)')
    parser.add_argument('--force', '-f', action='store_true',
                        help='Force reprocessing, ignore cache')
    parser.add_argument('--cache-stats', action='store_true',
                        help='Show cache statistics and exit')
    parser.add_argument('--cache-clear', action='store_true',
                        help='Clear the processing cache and exit')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose debug output')

    args = parser.parse_args()

    # Normalize Google API keys: flatten comma-separated, apply default
    if args.google_api_key:
        flat_keys = []
        for k in args.google_api_key:
            flat_keys.extend([x.strip() for x in k.split(',') if x.strip()])
        args.google_api_key = flat_keys
    else:
        args.google_api_key = ['AIzaSyCAR_zu_QkIR8P2amDHvHFvos2T7eg0u7c']

    if not os.path.isdir(args.ebook_dir):
        print(f"Error: {args.ebook_dir} is not a directory")
        sys.exit(1)

    # Handle cache commands before full pipeline init

    # Fix: Use temp location for cache path logic to handle network shares correctly
    if os.path.exists("/tmp"):
        path_hash = hashlib.md5(args.ebook_dir.encode()).hexdigest()
        cache_path = os.path.join("/tmp", f"ebook_metadata_{path_hash}.db")
    else:
        cache_path = os.path.join(args.ebook_dir, '.metadata_cache.db')

    if args.cache_stats:
        if os.path.exists(cache_path):
            log = logging.getLogger('cache_cmd')
            cache = ProcessingCache(cache_path, log)
            stats = cache.get_stats()
            print(f"\n📊 Processing Cache Statistics")
            print(f"   Location:         {cache_path}")
            print(f"   Total books:      {stats['total']}")
            print(f"   Successful:       {stats['success']}")
            print(f"   Errors:           {stats['errors']}")
            print(f"   Avg completeness: {stats['avg_completeness']*100:.0f}%")
            cache.close()
        else:
            print(f"No cache found at {cache_path}")
        sys.exit(0)

    if args.cache_clear:
        removed = False
        for suffix in ('', '-wal', '-shm'):
            p = cache_path + suffix
            if os.path.exists(p):
                os.remove(p)
                removed = True
        if removed:
            print(f"✓ Cache cleared: {cache_path}")
        else:
            print(f"No cache found at {cache_path}")
        sys.exit(0)

    pipeline = EbookMetadataPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

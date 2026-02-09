#!/usr/bin/env python3
"""
Ebook Metadata Pipeline
=======================
Comprehensive metadata extraction, enrichment, writing, and renaming for ebook collections.

Pipeline Order:
  1. Virtual environment bootstrap & dependency check (runs once at startup)
  2. Gutenberg RDF Catalog (fastest, most reliable for PG books)
  3. Embedded metadata extraction from file
  4. Public APIs (Open Library, Google Books)
  5. AI text analysis via Claude API (last resort / unreadable fallback)

Supported Formats: .epub, .pdf, .mobi, .azw, .azw3, .fb2, .txt, .html, .htm, .djvu, .cbz, .cbr, .lit

Setup (recommended):
  # Option 1: Auto-bootstrap (creates venv automatically)
  python ebook_metadata_pipeline.py /path/to/ebooks --bootstrap-venv

  # Option 2: Manual venv
  python -m venv ~/.venvs/ebook-pipeline
  source ~/.venvs/ebook-pipeline/bin/activate
  pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4
  python ebook_metadata_pipeline.py /path/to/ebooks

  # Option 3: System-wide (not recommended)
  pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4 --break-system-packages
  python ebook_metadata_pipeline.py /path/to/ebooks

  Also: Calibre CLI tools (ebook-meta) for metadata writing
    sudo apt install calibre

Usage:
  python ebook_metadata_pipeline.py /path/to/ebooks [--rdf-catalog /path/to/cache/rdf-files] [options]

  Run with --help for full options.
"""

import argparse
import hashlib
import importlib
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

OPENLIBRARY_SEARCH_URL = "https://openlibrary.org/search.json"
OPENLIBRARY_BOOK_URL = "https://openlibrary.org/api/books"
GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"

# Text extraction configuration
DEFAULT_MAX_PAGES = 15          # Uniform page limit across all formats
DEFAULT_MAX_CHARS = 15000       # Character limit for text extraction
AI_EXTRACT_MAX_CHARS = 15000    # Max chars sent to AI
API_DELAY_SECONDS = 1.0

# What metadata fields each format can actually store (and round-trip via Calibre ebook-meta)
# Research sources: EPUB OPF/Dublin Core spec (IDPF), MOBI EXTH record definitions
# (MobileRead Wiki + Calibre source format_docs/pdb/mobi.txt + KindleUnpack),
# PDF XMP/Document Info (ISO 16684-1, pikepdf docs), FB2 FictionBook XML spec.
# See ebook_format_metadata_reference.md for full documentation.
FORMAT_WRITABLE_FIELDS = {
    # EPUB: OPF Dublin Core + <meta> extensions. Most capable format.
    '.epub': {'title', 'authors', 'publisher', 'date', 'language', 'description',
              'isbn', 'tags', 'series', 'cover_embed'},
    # PDF: Document Info dict (/Title, /Author, /Keywords) + XMP Dublin Core.
    # publisher/date/language/isbn via XMP dc:* namespace.
    # series via calibre: custom XMP namespace (Calibre-only round-trip).
    # No native cover embed mechanism.
    '.pdf':  {'title', 'authors', 'publisher', 'date', 'language', 'description',
              'isbn', 'tags', 'series'},
    # MOBI: EXTH records (100=author, 101=publisher, 103=desc, 104=isbn,
    # 105=subject, 106=date). Language in MOBI header. Cover via EXTH 201.
    # Series via EXTH 535/536 (Calibre custom, widely adopted).
    '.mobi': {'title', 'authors', 'publisher', 'date', 'language', 'description',
              'isbn', 'tags', 'series', 'cover_embed'},
    # AZW: Identical to MOBI (same format, different DRM scheme).
    '.azw':  {'title', 'authors', 'publisher', 'date', 'language', 'description',
              'isbn', 'tags', 'series', 'cover_embed'},
    # AZW3/KF8: Compiled EPUB in PDB container. Has BOTH EXTH records (MOBI compat)
    # AND internal OPF metadata (EPUB-like). Full EPUB capabilities.
    '.azw3': {'title', 'authors', 'publisher', 'date', 'language', 'description',
              'isbn', 'tags', 'series', 'cover_embed'},
    # FB2: XML-based. ONLY format with native <genre> element (first-class field).
    # Also has native <sequence> for series (not a Calibre extension).
    '.fb2':  {'title', 'authors', 'publisher', 'date', 'language', 'description',
              'isbn', 'genres', 'tags', 'series', 'cover_embed'},
    # Archive/image formats: no metadata container
    '.cbz':  set(),
    '.cbr':  set(),
    '.djvu': set(),
}
# Fields that NO ebook format stores natively (library/catalog concepts only):
#   - orig_date   : formats have one "date" field; no separate original pub date
#   - LCC/DDC     : Library of Congress / Dewey Decimal classification codes
#   - page_count  : physical book concept; ebook pages vary by viewer/device
#   - cover_url   : formats embed cover images, not URLs to external images
#   - subtitle    : most formats fold subtitle into title; no separate field
#   - edition     : no standard ebook field across formats
#   - genres      : no standard field EXCEPT FB2 which has native <genre> element
#   - subjects    : conceptually stored as tags/dc:subject, not a separate field

# Default venv location
DEFAULT_VENV_DIR = os.path.expanduser("~/.venvs/ebook-pipeline")


# =============================================================================
# VENV BOOTSTRAP
# =============================================================================

def bootstrap_venv(venv_dir: str = DEFAULT_VENV_DIR, force: bool = False) -> str:
    """
    Create a virtual environment and install all dependencies.
    Returns the path to the venv's Python interpreter.
    """
    venv_python = os.path.join(venv_dir, 'bin', 'python')

    if os.path.exists(venv_python) and not force:
        print(f"  ✓ Venv already exists: {venv_dir}")
        return venv_python

    print(f"\n{'='*60}")
    print("  BOOTSTRAPPING VIRTUAL ENVIRONMENT")
    print(f"{'='*60}")
    print(f"  Location: {venv_dir}")

    # Check if venv module is available
    try:
        import venv as _venv_check
    except ImportError:
        print(f"\n  ✗ python3-venv is not installed.")
        print(f"    Fix with: sudo apt install python3-venv")
        print(f"    Then re-run: python {sys.argv[0]} --bootstrap-venv")
        sys.exit(1)

    # Create venv
    print(f"  Creating virtual environment...")
    os.makedirs(os.path.dirname(venv_dir), exist_ok=True)
    result = subprocess.run(
        [sys.executable, '-m', 'venv', venv_dir],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        print(f"  ✗ Failed to create venv: {stderr}")
        if 'ensurepip' in stderr or 'No module named' in stderr:
            print(f"    Fix with: sudo apt install python3-venv python3-pip")
        sys.exit(1)
    print(f"  ✓ Venv created")

    # Upgrade pip inside venv
    print(f"  Upgrading pip...")
    subprocess.run(
        [venv_python, '-m', 'pip', 'install', '--upgrade', 'pip', '--quiet'],
        capture_output=True
    )

    # Install all dependencies
    packages = [
        'ebooklib', 'PyMuPDF', 'lxml', 'requests',
        'beautifulsoup4', 'anthropic', 'mobi',
    ]
    print(f"  Installing {len(packages)} packages...")
    failed = []
    for pkg in packages:
        print(f"    ↻ {pkg}...", end='', flush=True)
        try:
            result = subprocess.run(
                [venv_python, '-m', 'pip', 'install', pkg, '--quiet'],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode == 0:
                print(f" ✓")
            else:
                print(f" ✗")
                failed.append(pkg)
                if 'mobi' in pkg.lower():
                    print(f"      Note: mobi module often fails. MOBI files will use Calibre fallback.")
        except subprocess.TimeoutExpired:
            print(f" ✗ (timed out)")
            failed.append(pkg)

    print(f"\n  ✅ Venv ready: {venv_dir}")
    if failed:
        print(f"  ⚠ Failed to install: {', '.join(failed)} (non-critical, fallbacks available)")
    print(f"  To activate manually:")
    print(f"    source {venv_dir}/bin/activate")
    print(f"{'='*60}\n")

    return venv_python


def relaunch_in_venv(venv_dir: str = DEFAULT_VENV_DIR):
    """
    If not already running inside the venv, re-exec this script under the venv Python.
    """
    venv_python = os.path.join(venv_dir, 'bin', 'python')
    current_python = os.path.realpath(sys.executable)
    venv_python_real = os.path.realpath(venv_python) if os.path.exists(venv_python) else None

    if venv_python_real and current_python == venv_python_real:
        return  # Already inside the venv

    if not os.path.exists(venv_python):
        return  # No venv to relaunch into

    print(f"  Re-launching inside venv: {venv_dir}")
    os.execv(venv_python, [venv_python] + sys.argv)


# =============================================================================
# DEPENDENCY CHECKER
# =============================================================================

REQUIRED_DEPENDENCIES = {
    'ebooklib': 'ebooklib', 'fitz': 'PyMuPDF', 'lxml': 'lxml',
    'requests': 'requests', 'bs4': 'beautifulsoup4',
    'anthropic': 'anthropic', 'mobi': 'mobi',
}
CRITICAL_MODULES = {'lxml', 'requests', 'bs4'}
FORMAT_MODULES = {
    'ebooklib': ['.epub'], 'fitz': ['.pdf'],
    'mobi': ['.mobi', '.azw', '.azw3'],
}
REQUIRED_TOOLS = {'ebook-meta': 'calibre', 'ebook-convert': 'calibre'}


@dataclass
class DependencyStatus:
    available: Dict[str, bool] = field(default_factory=dict)
    versions: Dict[str, str] = field(default_factory=dict)
    install_attempted: Dict[str, bool] = field(default_factory=dict)
    install_failed: List[str] = field(default_factory=list)
    tools_available: Dict[str, bool] = field(default_factory=dict)
    all_critical_ok: bool = True
    warnings: List[str] = field(default_factory=list)

    def module_ok(self, module_name: str) -> bool:
        return self.available.get(module_name, False)

    def tool_ok(self, tool_name: str) -> bool:
        return self.tools_available.get(tool_name, False)

    def formats_supported(self) -> Dict[str, bool]:
        result = {}
        for mod, exts in FORMAT_MODULES.items():
            for ext in exts:
                if ext not in result:
                    result[ext] = True
                if not self.module_ok(mod):
                    result[ext] = False
        return result


class DependencyChecker:
    def __init__(self, logger: Optional[logging.Logger] = None, auto_install: bool = True):
        self.log = logger or logging.getLogger('ebook_pipeline.deps')
        self.auto_install = auto_install

    def _in_venv(self) -> bool:
        """Check if running inside a virtual environment."""
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        )

    def check_module(self, module_name: str) -> Tuple[bool, Optional[str]]:
        try:
            mod = importlib.import_module(module_name)
            version = getattr(mod, '__version__', None)
            if not version:
                version = getattr(mod, 'VERSION', None)
            if not version:
                version = getattr(mod, 'version', None)
            if isinstance(version, tuple):
                version = '.'.join(str(v) for v in version)
            elif version and not isinstance(version, str):
                version = str(version)
            return True, str(version) if version else 'unknown'
        except ImportError:
            return False, None
        except Exception as e:
            self.log.warning(f"Module {module_name} import error: {e}")
            return False, None

    def check_tool(self, tool_name: str) -> bool:
        try:
            result = subprocess.run([tool_name, '--version'],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return False

    def _pip_supports_break_system(self) -> bool:
        """Check if pip version supports --break-system-packages (pip >= 23.0.1)."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', '--version'],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                # Output like: "pip 23.1.2 from /usr/lib/..."
                match = re.search(r'pip\s+(\d+)\.(\d+)', result.stdout)
                if match:
                    major, minor = int(match.group(1)), int(match.group(2))
                    return (major > 23) or (major == 23 and minor >= 1)
        except Exception:
            pass
        return False

    def install_module(self, module_name: str, pip_name: str) -> bool:
        self.log.info(f"  ↻ Installing {pip_name}...")

        # Build install commands to try in order
        attempts = []
        base_cmd = [sys.executable, '-m', 'pip', 'install', pip_name, '--quiet']

        if self._in_venv():
            # Inside venv: never need --break-system-packages
            attempts.append(base_cmd)
        else:
            # System Python: try with --break-system-packages first if pip supports it,
            # then fall back to without it for older pip or permissive systems
            if self._pip_supports_break_system():
                attempts.append(base_cmd + ['--break-system-packages'])
                attempts.append(base_cmd)  # fallback without flag
            else:
                attempts.append(base_cmd)  # old pip, don't use the flag
                attempts.append(base_cmd + ['--user'])  # try --user as last resort

        last_error = ""
        for cmd in attempts:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    if module_name in sys.modules:
                        del sys.modules[module_name]
                    importlib.invalidate_caches()
                    ok, ver = self.check_module(module_name)
                    if ok:
                        self.log.info(f"  ✓ {pip_name} installed successfully (v{ver})")
                        return True
                    else:
                        self.log.warning(f"  ✗ {pip_name} installed but import still fails")
                        return False
                else:
                    last_error = result.stderr.strip()[:200]
                    # If pip printed usage info, the flag isn't supported — try next
                    if 'usage:' in last_error.lower():
                        continue
                    # If externally-managed error, try next attempt (with --break-system-packages or --user)
                    if 'externally-managed' in last_error.lower():
                        continue
                    # Other errors — don't retry
                    break
            except subprocess.TimeoutExpired:
                last_error = "timed out"
                break
            except Exception as e:
                last_error = str(e)
                break

        self.log.warning(f"  ✗ pip install {pip_name} failed: {last_error}")
        return False

    def check_and_install(self) -> DependencyStatus:
        status = DependencyStatus()

        G = '\033[32m'; R = '\033[31m'; Y = '\033[33m'; D = '\033[90m'
        C = '\033[36m'; B = '\033[1m'; X = '\033[0m'

        env_label = "venv" if self._in_venv() else "system"
        self.log.info(f"{C}Checking dependencies{X} {D}({env_label}: {sys.executable})...{X}")

        missing = []
        for module_name, pip_name in REQUIRED_DEPENDENCIES.items():
            ok, ver = self.check_module(module_name)
            status.available[module_name] = ok
            if ver:
                status.versions[module_name] = ver
            if ok:
                self.log.info(f"  {G}✓{X} {module_name} {D}(v{ver}){X}")
            else:
                missing.append((module_name, pip_name))
                self.log.warning(f"  {R}✗{X} {module_name} {R}— not installed{X}")

        if missing and self.auto_install:
            self.log.info(f"\n{Y}Attempting to install {len(missing)} missing package(s)...{X}")
            for module_name, pip_name in missing:
                status.install_attempted[module_name] = True
                success = self.install_module(module_name, pip_name)
                status.available[module_name] = success
                if success:
                    _, ver = self.check_module(module_name)
                    if ver:
                        status.versions[module_name] = ver
                else:
                    status.install_failed.append(pip_name)
        elif missing and not self.auto_install:
            pip_names = [p for _, p in missing]
            install_cmd = f"pip install {' '.join(pip_names)}"
            if not self._in_venv() and self._pip_supports_break_system():
                install_cmd += " --break-system-packages"
            elif not self._in_venv():
                install_cmd += " --user"
            self.log.warning(
                f"\n  {R}Missing packages:{X} {', '.join(pip_names)}\n"
                f"  {D}Install with:{X} {install_cmd}\n"
                f"  {D}Or use a venv:{X} python {sys.argv[0]} --bootstrap-venv")

        self.log.info("")
        for tool_name, package_name in REQUIRED_TOOLS.items():
            ok = self.check_tool(tool_name)
            status.tools_available[tool_name] = ok
            if ok:
                self.log.info(f"  {G}✓{X} {tool_name} {D}(from {package_name}){X}")
            else:
                self.log.warning(f"  {R}✗{X} {tool_name} {R}— not found{X} {D}(install {package_name}){X}")
                status.warnings.append(f"{tool_name} not found. Install with: sudo apt install {package_name}")

        for mod in CRITICAL_MODULES:
            if not status.module_ok(mod):
                status.all_critical_ok = False
                status.warnings.append(f"Critical module '{mod}' is missing")

        fmt_support = status.formats_supported()
        degraded = [ext for ext, ok in fmt_support.items() if not ok]
        if degraded:
            status.warnings.append(f"Degraded support for: {', '.join(degraded)} (metadata write/deep extraction will be skipped)")

        total = len(REQUIRED_DEPENDENCIES)
        ok_count = sum(1 for v in status.available.values() if v)
        tool_total = len(REQUIRED_TOOLS)
        tool_ok = sum(1 for v in status.tools_available.values() if v)

        self.log.info("")
        if ok_count == total and tool_ok == tool_total:
            self.log.info(f"  {G}✅ All {total} modules + {tool_total} tools available{X}")
        else:
            self.log.info(f"  {Y}⚠ {ok_count}/{total} modules, {tool_ok}/{tool_total} tools available{X}")
            if status.install_failed:
                install_cmd = f"pip install {' '.join(status.install_failed)}"
                if not self._in_venv() and self._pip_supports_break_system():
                    install_cmd += " --break-system-packages"
                elif not self._in_venv():
                    install_cmd += " --user"
                self.log.warning(
                    f"  {R}Failed to install:{X} {', '.join(status.install_failed)}\n"
                    f"  {D}Manual install:{X} {install_cmd}\n"
                    f"  {D}Recommended:{X}    python {sys.argv[0]} --bootstrap-venv")

        return status


# =============================================================================
# DATA MODEL
# =============================================================================

@dataclass
class BookMetadata:
    title: Optional[str] = None
    subtitle: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    editors: List[str] = field(default_factory=list)
    translators: List[str] = field(default_factory=list)
    illustrators: List[str] = field(default_factory=list)
    contributors: List[str] = field(default_factory=list)
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    original_publication_date: Optional[str] = None
    edition: Optional[str] = None
    revision: Optional[str] = None
    isbn_10: Optional[str] = None
    isbn_13: Optional[str] = None
    gutenberg_id: Optional[str] = None
    oclc: Optional[str] = None
    lccn: Optional[str] = None
    openlibrary_id: Optional[str] = None
    google_books_id: Optional[str] = None
    asin: Optional[str] = None
    doi: Optional[str] = None
    subjects: List[str] = field(default_factory=list)
    genres: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    lcc: Optional[str] = None
    ddc: Optional[str] = None
    bisac: List[str] = field(default_factory=list)
    series: Optional[str] = None
    series_index: Optional[float] = None
    language: Optional[str] = None
    description: Optional[str] = None
    long_description: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    rights: Optional[str] = None
    copyright: Optional[str] = None
    license_url: Optional[str] = None
    cover_url: Optional[str] = None
    source_file: Optional[str] = None
    source_format: Optional[str] = None
    file_hash: Optional[str] = None
    sources_used: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    processing_notes: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def merge(self, other: 'BookMetadata', source_name: str = "unknown"):
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
                combined = list(current)
                for item in incoming:
                    if item not in combined:
                        combined.append(item)
                if len(combined) > len(current):
                    # Extra dedup for authors (handles "Last, First" vs "First Last")
                    if fld == 'authors':
                        combined = deduplicate_authors(combined)
                    setattr(self, fld, combined)
                    merged_fields.append(fld)
        if merged_fields:
            self.sources_used.append(source_name)
            self.processing_notes.append(f"Merged from {source_name}: {', '.join(merged_fields)}")

    def completeness_score(self) -> float:
        weights = {
            'title': 20, 'authors': 20, 'language': 5, 'publication_date': 10,
            'description': 10, 'subjects': 8, 'isbn_13': 5, 'isbn_10': 3,
            'publisher': 5, 'genres': 5, 'series': 3, 'page_count': 3, 'cover_url': 3,
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
# READABILITY REPORT
# =============================================================================

@dataclass
class ReadabilityResult:
    """Tracks whether a file's content could be read and why/why not."""
    readable: bool = False
    text_extracted: bool = False
    text_length: int = 0
    pages_read: int = 0
    total_pages: Optional[int] = None
    method_used: Optional[str] = None       # e.g. 'epub-native', 'pdf-pymupdf', 'calibre-convert', 'ai-ocr'
    methods_tried: List[str] = field(default_factory=list)
    failure_reasons: List[str] = field(default_factory=list)
    is_scanned_pdf: bool = False             # Detected as image-only PDF
    is_encrypted: bool = False
    is_corrupted: bool = False
    ai_fallback_used: bool = False
    ai_fallback_success: bool = False

    @property
    def status_label(self) -> str:
        if self.readable and self.text_extracted:
            return 'readable'
        elif self.is_encrypted:
            return 'encrypted'
        elif self.is_corrupted:
            return 'corrupted'
        elif self.is_scanned_pdf:
            return 'scanned-image-only'
        elif self.ai_fallback_used and self.ai_fallback_success:
            return 'ai-recovered'
        elif self.failure_reasons:
            return 'unreadable'
        else:
            return 'no-text'

    @property
    def status_icon(self) -> str:
        icons = {
            'readable': '\033[32m✓ Readable\033[0m',
            'encrypted': '\033[31m🔒 Encrypted\033[0m',
            'corrupted': '\033[31m✗ Corrupted\033[0m',
            'scanned-image-only': '\033[33m📷 Scanned (image-only)\033[0m',
            'ai-recovered': '\033[35m🤖 AI-recovered\033[0m',
            'unreadable': '\033[31m✗ Unreadable\033[0m',
            'no-text': '\033[33m⚠ No text content\033[0m',
        }
        return icons.get(self.status_label, '\033[90m? Unknown\033[0m')


# =============================================================================
# LOGGING SETUP
# =============================================================================

class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[90m', 'INFO': '\033[0m', 'WARNING': '\033[33m',
        'ERROR': '\033[31m', 'CRITICAL': '\033[41m',
    }
    RESET = '\033[0m'
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        return f"{color}{record.getMessage()}{self.RESET}"

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


class BufferedLogHandler(logging.Handler):
    def __init__(self, min_level=logging.INFO, thread_id=None):
        super().__init__(min_level)
        self.buffer = []
        self.thread_id = thread_id or threading.get_ident()
        self.setFormatter(ColorFormatter())

    def emit(self, record):
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
    logger.setLevel(TRACE)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    fh = logging.FileHandler(os.path.join(log_dir, f'pipeline_{timestamp}.log'))
    fh.setLevel(TRACE)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(fh)

    eh = logging.FileHandler(os.path.join(log_dir, f'errors_{timestamp}.log'))
    eh.setLevel(logging.ERROR)
    eh.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s'))
    logger.addHandler(eh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(ColorFormatter())
    logger.addHandler(ch)

    sh = logging.FileHandler(os.path.join(log_dir, f'stats_{timestamp}.jsonl'))
    sh.setLevel(logging.INFO)
    sh.setFormatter(logging.Formatter('%(message)s'))
    stats_logger = logging.getLogger('ebook_pipeline.stats')
    stats_logger.addHandler(sh)
    stats_logger.propagate = False

    # Unreadable files log
    uh = logging.FileHandler(os.path.join(log_dir, f'unreadable_{timestamp}.log'))
    uh.setLevel(logging.WARNING)
    uh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    unreadable_logger = logging.getLogger('ebook_pipeline.unreadable')
    unreadable_logger.addHandler(uh)
    unreadable_logger.propagate = False

    return logger


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def sanitize_filename(name: str, max_length: int = 200) -> str:
    if not name:
        return "Unknown"
    name = unicodedata.normalize('NFKD', name)
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', name)
    name = re.sub(r'[\s]+', ' ', name).strip()
    name = name.strip('. ')
    if len(name) > max_length:
        name = name[:max_length].rsplit(' ', 1)[0].strip()
    return name or "Unknown"


def clean_title(title: str) -> str:
    if not title:
        return title
    title = re.sub(r'_\s+', ': ', title)
    title = re.sub(r'\s+_', ': ', title)
    title = re.sub(r'\s*\[(?:Team[- ]\w+|WWRG|scan|OCR|ebook|eBook|converted)\]\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s*\((?:True\s+)?(?:PDF|EPUB|AZW3?|MOBI|retail|scan|OCR)\)\s*', '', title, flags=re.IGNORECASE)
    title = re.sub(r'\s+WW\s*$', '', title)
    title = re.sub(r'\s+HQ\s*$', '', title)
    title = re.sub(r'\s+', ' ', title).strip()
    return title


def file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def extract_year(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    match = re.search(r'(1[4-9]\d{2}|20[0-3]\d)', str(date_str))
    return match.group(1) if match else None


def ordinal(n) -> str:
    n = int(n)
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th','st','nd','rd'][n % 10] if n % 10 < 4 else 'th'}"


def clean_author_string(author: str) -> str:
    author = re.sub(r'\s*\([^)]*\d{4}[^)]*\)\s*', ' ', author).strip()
    author = re.sub(r'\s+WW\s*$', '', author).strip()
    author = re.sub(r'\s*\[.*?\]\s*', ' ', author).strip()
    author = re.sub(r'[\[\]]', '', author).strip()
    author = re.sub(r'\s+', ' ', author).strip()
    return author


def deduplicate_authors(authors: List[str]) -> List[str]:
    if not authors or len(authors) <= 1:
        return authors
    def normalize(name):
        n = name.strip().rstrip('.')
        if ',' in n and n.count(',') == 1:
            parts = n.split(',', 1)
            n = f"{parts[1].strip()} {parts[0].strip()}"
        tokens = re.split(r'[\s.]+', n)
        return {t.lower() for t in tokens if len(t) > 1}
    result, seen_parts = [], []
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


def run_command(cmd: list, timeout: int = 60) -> Tuple[int, str, str]:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except Exception as e:
        return -1, "", str(e)


# =============================================================================
# GENRE / SUBJECT INFERENCE
# =============================================================================

DDC_TO_GENRE = {
    '0': 'Computers & General Works', '00': 'Computers & General Works',
    '001': 'Knowledge & Systems', '004': 'Computers', '005': 'Computers', '006': 'Computers',
    '1': 'Philosophy', '10': 'Philosophy', '13': 'Psychology',
    '14': 'Philosophy', '15': 'Psychology', '16': 'Philosophy',
    '17': 'Philosophy', '18': 'Philosophy', '19': 'Philosophy',
    '2': 'Religion', '20': 'Religion', '22': 'Religion', '23': 'Religion',
    '24': 'Religion', '25': 'Religion', '28': 'Religion', '29': 'Religion',
    '3': 'Social Sciences', '30': 'Social Sciences', '31': 'Social Sciences',
    '32': 'Political Science', '33': 'Business & Economics',
    '34': 'Law', '35': 'Public Administration', '36': 'Social Services',
    '37': 'Education', '38': 'Business & Economics', '39': 'Social Sciences',
    '4': 'Language', '40': 'Language', '41': 'Language',
    '5': 'Science', '50': 'Science', '51': 'Mathematics',
    '52': 'Science', '53': 'Science', '54': 'Science', '55': 'Science',
    '56': 'Science', '57': 'Science', '58': 'Science', '59': 'Science',
    '6': 'Technology & Engineering', '60': 'Technology & Engineering',
    '61': 'Medical', '62': 'Technology & Engineering', '63': 'Technology & Engineering',
    '64': 'Home & Garden', '65': 'Business & Economics', '66': 'Technology & Engineering',
    '67': 'Technology & Engineering', '68': 'Technology & Engineering', '69': 'Architecture',
    '7': 'Arts & Recreation', '70': 'Arts', '71': 'Architecture', '72': 'Architecture',
    '73': 'Arts', '74': 'Arts', '75': 'Arts', '76': 'Arts',
    '77': 'Photography', '78': 'Music', '79': 'Sports & Recreation',
    '8': 'Literature', '80': 'Literature', '81': 'Literature', '82': 'Literature',
    '83': 'Literature', '84': 'Literature', '85': 'Literature', '86': 'Literature',
    '87': 'Literature', '89': 'Literature',
    '9': 'History & Geography', '90': 'History', '91': 'Geography',
    '92': 'Biography & Autobiography', '93': 'History', '94': 'History',
    '95': 'History', '96': 'History', '97': 'History', '98': 'History', '99': 'History',
}

LCC_TO_GENRE = {
    'A': 'General Works', 'B': 'Philosophy & Religion',
    'BF': 'Psychology', 'BL': 'Religion', 'BM': 'Religion', 'BP': 'Religion',
    'BR': 'Religion', 'BS': 'Religion', 'BT': 'Religion', 'BV': 'Religion', 'BX': 'Religion',
    'C': 'History', 'D': 'History', 'E': 'History', 'F': 'History',
    'G': 'Geography', 'H': 'Social Sciences',
    'HA': 'Social Sciences', 'HB': 'Business & Economics', 'HC': 'Business & Economics',
    'HD': 'Business & Economics', 'HE': 'Business & Economics', 'HF': 'Business & Economics',
    'HG': 'Business & Economics', 'HJ': 'Business & Economics',
    'HM': 'Social Sciences', 'HN': 'Social Sciences', 'HQ': 'Social Sciences',
    'HV': 'Social Services', 'J': 'Political Science', 'K': 'Law',
    'L': 'Education', 'M': 'Music', 'N': 'Arts', 'P': 'Literature & Language',
    'PA': 'Literature', 'PB': 'Language', 'PC': 'Language', 'PD': 'Language',
    'PE': 'Language', 'PF': 'Language', 'PG': 'Literature', 'PH': 'Literature',
    'PJ': 'Literature', 'PK': 'Literature', 'PL': 'Literature', 'PM': 'Language',
    'PN': 'Literature', 'PQ': 'Literature', 'PR': 'Literature', 'PS': 'Literature',
    'PT': 'Literature', 'PZ': 'Literature',
    'Q': 'Science', 'QA': 'Mathematics', 'QB': 'Science', 'QC': 'Science',
    'QD': 'Science', 'QE': 'Science', 'QH': 'Science', 'QK': 'Science',
    'QL': 'Science', 'QM': 'Medical', 'QP': 'Medical', 'QR': 'Medical',
    'R': 'Medical', 'S': 'Agriculture',
    'T': 'Technology & Engineering', 'TA': 'Technology & Engineering',
    'TC': 'Technology & Engineering', 'TD': 'Technology & Engineering',
    'TE': 'Technology & Engineering', 'TF': 'Technology & Engineering',
    'TG': 'Technology & Engineering', 'TH': 'Architecture',
    'TJ': 'Technology & Engineering', 'TK': 'Technology & Engineering',
    'TL': 'Technology & Engineering', 'TN': 'Technology & Engineering',
    'TP': 'Technology & Engineering', 'TR': 'Photography',
    'TS': 'Technology & Engineering', 'TT': 'Crafts & Hobbies',
    'TX': 'Cooking', 'U': 'Military Science', 'V': 'Naval Science', 'Z': 'Library Science',
}

TITLE_KEYWORD_GENRES = {
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
    'circuit': 'Technology & Engineering', 'electronics': 'Technology & Engineering',
    'transistor': 'Technology & Engineering', 'microprocessor': 'Technology & Engineering',
    'embedded system': 'Technology & Engineering', 'robotics': 'Technology & Engineering',
    'arduino': 'Technology & Engineering', 'raspberry pi': 'Technology & Engineering',
    'signal processing': 'Technology & Engineering', 'FPGA': 'Technology & Engineering',
    'radar': 'Technology & Engineering', 'antenna': 'Technology & Engineering',
    'semiconductor': 'Technology & Engineering', 'VLSI': 'Technology & Engineering',
    'happiness': 'Self-Help', 'mindfulness': 'Self-Help', 'meditation': 'Self-Help',
    'emotional intelligence': 'Self-Help', 'self-help': 'Self-Help',
    'affirmation': 'Self-Help', 'gratitude': 'Self-Help', 'resilience': 'Self-Help',
    'self-love': 'Self-Help', 'confidence': 'Self-Help', 'motivation': 'Self-Help',
    'habit': 'Self-Help', 'productivity': 'Self-Help', 'wellness': 'Self-Help',
    'personal growth': 'Self-Help', 'self-improvement': 'Self-Help',
    'overthinking': 'Self-Help', 'procrastination': 'Self-Help',
    'stoicism': 'Self-Help', 'emotional agility': 'Self-Help',
    'anxiety': 'Psychology', 'therapy': 'Psychology', 'psychology': 'Psychology',
    'cognitive': 'Psychology', 'behavioral': 'Psychology', 'emotions': 'Psychology',
    'business': 'Business & Economics', 'management': 'Business & Economics',
    'leadership': 'Business & Economics', 'entrepreneur': 'Business & Economics',
    'marketing': 'Business & Economics', 'finance': 'Business & Economics',
    'investing': 'Business & Economics', 'economics': 'Business & Economics',
    'negotiation': 'Business & Economics', 'startup': 'Business & Economics',
    'nutrition': 'Health & Fitness', 'fitness': 'Health & Fitness',
    'diet': 'Health & Fitness', 'exercise': 'Health & Fitness',
    'yoga': 'Health & Fitness', 'medical': 'Medical',
    'physics': 'Science', 'chemistry': 'Science', 'biology': 'Science',
    'mathematics': 'Mathematics', 'calculus': 'Mathematics',
    'algebra': 'Mathematics', 'statistics': 'Mathematics',
    'history': 'History', 'biography': 'Biography & Autobiography',
    'memoir': 'Biography & Autobiography', 'autobiography': 'Biography & Autobiography',
    'novel': 'Fiction', 'mystery': 'Fiction', 'thriller': 'Fiction',
    'romance': 'Fiction', 'fantasy': 'Fiction', 'science fiction': 'Fiction',
    'textbook': 'Education', 'curriculum': 'Education', 'teaching': 'Education',
}

PUBLISHER_GENRES = {
    'packt': 'Computers', "o'reilly": 'Computers', 'apress': 'Computers',
    'no starch': 'Computers', 'manning': 'Computers', 'pragmatic': 'Computers',
    'wiley': None, 'springer': 'Science', 'elsevier': None,
    'zondervan': 'Religion', 'baker': 'Religion', 'bethany': 'Religion',
    'hay house': 'Self-Help', 'sounds true': 'Self-Help',
    'penguin random': None, 'harpercollins': None, 'simon & schuster': None,
    'mcgraw': None, 'oxford university': None, 'cambridge university': None,
    'harvard business': 'Business & Economics', 'mit press': 'Science',
    'artech': 'Technology & Engineering', 'newnes': 'Technology & Engineering',
}


def infer_genres_subjects(meta) -> tuple:
    inferred_genres = set()
    inferred_subjects = set()

    if meta.ddc:
        ddc_num = re.sub(r'[^\d.]', '', str(meta.ddc))
        for length in [3, 2, 1]:
            prefix = ddc_num[:length]
            if prefix in DDC_TO_GENRE:
                inferred_genres.add(DDC_TO_GENRE[prefix])
                break

    if meta.lcc:
        lcc_str = str(meta.lcc).strip()
        prefix2 = re.match(r'^([A-Z]{1,2})', lcc_str)
        if prefix2:
            p2 = prefix2.group(1)
            if p2 in LCC_TO_GENRE:
                inferred_genres.add(LCC_TO_GENRE[p2])
            elif p2[0] in LCC_TO_GENRE:
                inferred_genres.add(LCC_TO_GENRE[p2[0]])

    if meta.title:
        full_text = meta.title.lower()
        if meta.subtitle:
            full_text += ' ' + meta.subtitle.lower()
        if meta.description:
            full_text += ' ' + meta.description.lower()[:200]
        for keyword, genre in TITLE_KEYWORD_GENRES.items():
            if keyword in full_text:
                inferred_genres.add(genre)
                skip_words = {'your', 'get', 'change', 'inner', 'self-'}
                if (len(keyword) > 5
                        and keyword not in ('business', 'history', 'medical', 'thrive', 'empowering', 'novel')
                        and not any(sw in keyword for sw in skip_words)
                        and len(keyword.split()) <= 2):
                    inferred_subjects.add(keyword.title())

    if meta.publisher:
        pub_lower = meta.publisher.lower()
        for pub_key, genre in PUBLISHER_GENRES.items():
            if pub_key in pub_lower and genre:
                inferred_genres.add(genre)
                break

    if meta.tags:
        for tag in meta.tags:
            tag_clean = tag.strip()
            if tag_clean and len(tag_clean) > 2:
                if tag_clean.lower() not in {'book', 'ebook', 'pdf', 'epub', 'general',
                                             'referex', 'unknown', 'none', 'other'}:
                    inferred_subjects.add(tag_clean)

    return list(inferred_genres), list(inferred_subjects)


# =============================================================================
# PROCESSING CACHE
# =============================================================================

class ProcessingCache:
    SCHEMA_VERSION = 2  # Bumped for readability tracking

    def __init__(self, cache_path: str, logger: logging.Logger):
        self.cache_path = cache_path
        self.log = logger
        self._lock = threading.Lock()
        self._conn = None
        self._init_db()

    def _init_db(self):
        self._conn = sqlite3.connect(self.cache_path, check_same_thread=False, timeout=30.0)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        except sqlite3.OperationalError:
            self.log.warning("Could not set WAL mode. Using default journal.")
            self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS processed_books (
                filepath TEXT PRIMARY KEY, file_size INTEGER NOT NULL,
                file_mtime REAL, completeness REAL,
                status TEXT NOT NULL DEFAULT 'success', sources_used TEXT,
                processed_at TEXT NOT NULL, title TEXT, authors TEXT,
                isbn TEXT, metadata_json TEXT,
                readability_status TEXT DEFAULT NULL)""")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (key TEXT PRIMARY KEY, value TEXT)""")
        self._conn.execute("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', ?)",
                           (str(self.SCHEMA_VERSION),))
        # Add readability_status column if missing (upgrade from v1)
        try:
            self._conn.execute("SELECT readability_status FROM processed_books LIMIT 1")
        except sqlite3.OperationalError:
            self._conn.execute("ALTER TABLE processed_books ADD COLUMN readability_status TEXT DEFAULT NULL")
        self._conn.commit()

    def is_processed(self, filepath: str) -> Optional[dict]:
        abspath = os.path.normpath(os.path.abspath(filepath))
        with self._lock:
            row = self._conn.execute(
                "SELECT file_size, file_mtime, completeness, status, processed_at "
                "FROM processed_books WHERE filepath = ?", (abspath,)).fetchone()
        if not row:
            return None
        cached_size, cached_mtime, completeness, status, processed_at = row
        # Verify the file still exists at this path
        if not os.path.exists(abspath):
            return None
        return {'completeness': completeness, 'status': status, 'processed_at': processed_at}

    def mark_processed(self, filepath: str, meta, status: str = 'success',
                       readability: Optional[ReadabilityResult] = None):
        abspath = os.path.normpath(os.path.abspath(filepath))
        try:
            stat = os.stat(filepath)
            file_size, file_mtime = stat.st_size, stat.st_mtime
        except OSError:
            file_size, file_mtime = 0, 0
        metadata_json = json.dumps({
            'title': meta.title, 'authors': meta.authors,
            'isbn_13': meta.isbn_13, 'isbn_10': meta.isbn_10,
            'publisher': meta.publisher, 'genres': meta.genres,
            'subjects': meta.subjects[:10], 'language': meta.language,
            'completeness': meta.completeness_score(), 'sources_used': meta.sources_used,
        }, ensure_ascii=False)
        readability_status = readability.status_label if readability else None
        with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO processed_books
                (filepath, file_size, file_mtime, completeness, status,
                 sources_used, processed_at, title, authors, isbn, metadata_json,
                 readability_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (abspath, file_size, file_mtime, meta.completeness_score(), status,
                 ','.join(meta.sources_used), datetime.now().isoformat(),
                 meta.title, ' | '.join(meta.authors) if meta.authors else None,
                 meta.isbn_13 or meta.isbn_10, metadata_json, readability_status))
            self._conn.commit()

    def get_stats(self) -> dict:
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) FROM processed_books").fetchone()[0]
            success = self._conn.execute("SELECT COUNT(*) FROM processed_books WHERE status = 'success'").fetchone()[0]
            avg_comp = self._conn.execute("SELECT AVG(completeness) FROM processed_books WHERE status = 'success'").fetchone()[0]
            # Readability stats
            try:
                unreadable = self._conn.execute(
                    "SELECT COUNT(*) FROM processed_books WHERE readability_status IN ('unreadable', 'encrypted', 'corrupted', 'scanned-image-only', 'no-text')"
                ).fetchone()[0]
            except sqlite3.OperationalError:
                unreadable = 0
        return {
            'total': total, 'success': success, 'errors': total - success,
            'avg_completeness': round(avg_comp or 0, 2), 'unreadable': unreadable,
        }

    def remove_entry(self, filepath: str):
        abspath = os.path.normpath(os.path.abspath(filepath))
        with self._lock:
            self._conn.execute("DELETE FROM processed_books WHERE filepath = ?", (abspath,))
            self._conn.commit()

    def clear_all(self) -> int:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM processed_books").fetchone()[0]
            self._conn.execute("DELETE FROM processed_books")
            self._conn.commit()
        return count

    def list_entries(self, limit: int = 50) -> List[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT filepath, completeness, status, readability_status, processed_at, title, authors "
                "FROM processed_books ORDER BY processed_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [
            {'filepath': r[0], 'completeness': r[1], 'status': r[2],
             'readability': r[3], 'processed_at': r[4], 'title': r[5], 'authors': r[6]}
            for r in rows
        ]

    def list_problems(self) -> dict:
        """Return categorized problem entries: unreadable, low completeness, errors."""
        results = {'unreadable': [], 'low_completeness': [], 'errors': [], 'no_api_match': []}
        with self._lock:
            # Unreadable files
            rows = self._conn.execute(
                "SELECT filepath, completeness, status, readability_status, title, metadata_json "
                "FROM processed_books WHERE readability_status IN "
                "('unreadable', 'encrypted', 'corrupted', 'scanned-image-only', 'no-text') "
                "ORDER BY readability_status"
            ).fetchall()
            for r in rows:
                results['unreadable'].append({
                    'filepath': r[0], 'completeness': r[1], 'status': r[2],
                    'readability': r[3], 'title': r[4], 'metadata_json': r[5]
                })

            # Error status
            rows = self._conn.execute(
                "SELECT filepath, completeness, status, readability_status, title, metadata_json "
                "FROM processed_books WHERE status = 'error' "
                "ORDER BY processed_at DESC"
            ).fetchall()
            for r in rows:
                results['errors'].append({
                    'filepath': r[0], 'completeness': r[1], 'status': r[2],
                    'readability': r[3], 'title': r[4], 'metadata_json': r[5]
                })

            # Low completeness (below 60%)
            rows = self._conn.execute(
                "SELECT filepath, completeness, status, readability_status, title, sources_used, metadata_json "
                "FROM processed_books WHERE completeness < 0.6 AND status = 'success' "
                "ORDER BY completeness ASC"
            ).fetchall()
            for r in rows:
                results['low_completeness'].append({
                    'filepath': r[0], 'completeness': r[1], 'status': r[2],
                    'readability': r[3], 'title': r[4], 'sources': r[5],
                    'metadata_json': r[6]
                })

        return results

    def close(self):
        if self._conn:
            self._conn.close()


def detect_gutenberg_id(filepath: str) -> Optional[str]:
    basename = os.path.basename(filepath)
    match = re.search(r'pg(\d+)', basename)
    if match:
        return match.group(1)
    match = re.search(r'/(\d{1,6})/', filepath)
    if match:
        return match.group(1)
    return None


def parse_filename_metadata(filepath: str) -> BookMetadata:
    meta = BookMetadata()
    basename = Path(filepath).stem
    basename = re.sub(r'[-_\s]*\bWW\b\s*$', '', basename).strip()
    basename = re.sub(r'[-_\s]*(images|images-\d+)\s*$', '', basename, flags=re.IGNORECASE).strip()

    publisher, year = None, None
    pub_match = re.search(r'\(([^,)]+),\s*(\d{4})\)\s*$', basename)
    if pub_match:
        publisher, year = pub_match.group(1).strip(), pub_match.group(2)
        basename = basename[:pub_match.start()].strip()
    if not year:
        year_match = re.search(r'\((\d{4})\)\s*$', basename)
        if year_match:
            year = year_match.group(1)
            basename = basename[:year_match.start()].strip()

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
        score, words, wc = 0, s.split(), len(s.split())
        if wc == 2 or wc == 3: score += 4
        elif wc == 1: score += 1
        if re.search(r'\b[A-Z]\.\s', s): score += 3
        if 'et al' in s.lower(): score += 4
        if re.search(r'\w+\s*[&,]\s*[A-Z]', s) and wc <= 6: score += 2
        if all(w[0].isupper() for w in words if w and w[0].isalpha()) and wc <= 4: score += 1
        title_words = {'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'with',
                       'how', 'why', 'what', 'guide', 'introduction', 'handbook',
                       'fundamentals', 'principles', 'analysis', 'programming',
                       'mastering', 'learning', 'practical', 'complete', 'essential',
                       'step-by-step', 'comprehensive', 'beginner', 'concepts'}
        score -= len(set(s.lower().split()) & title_words) * 2
        return score

    if len(parts) == 2:
        p0w, p1w = len(parts[0].split()), len(parts[1].split())
        if p0w == 1 and p1w >= 3 and parts[0][0].isupper():
            meta.authors, meta.title = [parts[0]], parts[1]
        elif p1w == 1 and p0w >= 3 and parts[1][0].isupper():
            meta.title, meta.authors = parts[0], [parts[1]]
        else:
            s_a, s_b = author_score(parts[0]), author_score(parts[1])
            if s_b > s_a:
                meta.title, meta.authors = parts[0], [parts[1]]
            elif s_a > s_b:
                meta.authors, meta.title = [parts[0]], parts[1]
            elif len(parts[0]) <= len(parts[1]):
                meta.authors, meta.title = [parts[0]], parts[1]
            else:
                meta.title, meta.authors = parts[0], [parts[1]]
    else:
        first_score, last_score = author_score(parts[0]), author_score(parts[-1])
        if last_score >= 3 and last_score > first_score:
            meta.authors, meta.title = [parts[-1]], ' - '.join(parts[:-1])
        else:
            meta.authors, meta.title = [parts[0]], ' - '.join(parts[1:])

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
    if meta.title:
        ed = re.search(r'(\d+)(?:st|nd|rd|th)\s*(?:ed|edition)', meta.title, re.IGNORECASE)
        if ed:
            meta.edition = f"{ordinal(ed.group(1))} Edition"
        meta.title = clean_title(meta.title)
    meta.processing_notes.append("filename-parsed")
    return meta


# =============================================================================
# TEXT EXTRACTORS (with uniform page limits and readability tracking)
# =============================================================================

class TextExtractor:
    def __init__(self, logger: logging.Logger, dep_status: Optional[DependencyStatus] = None,
                 max_pages: int = DEFAULT_MAX_PAGES, max_chars: int = DEFAULT_MAX_CHARS):
        self.log = logger
        self.dep_status = dep_status
        self.max_pages = max_pages
        self.max_chars = max_chars

    def extract(self, filepath: str, max_chars: Optional[int] = None) -> Tuple[Optional[str], ReadabilityResult]:
        """
        Extract text from an ebook file.
        Returns (text, ReadabilityResult) — text may be None if unreadable.
        """
        if max_chars is None:
            max_chars = self.max_chars

        result = ReadabilityResult()
        ext = Path(filepath).suffix.lower()

        # Check basic file accessibility
        if not os.path.exists(filepath):
            result.failure_reasons.append("File does not exist")
            return None, result
        if os.path.getsize(filepath) == 0:
            result.failure_reasons.append("File is empty (0 bytes)")
            result.is_corrupted = True
            return None, result

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
        methods_to_try = []

        if extractor:
            methods_to_try.append((f"{ext[1:]}-native", extractor))
        # Always add calibre as fallback
        methods_to_try.append(('calibre-convert', self._extract_via_calibre))

        text = None
        for method_name, method_func in methods_to_try:
            result.methods_tried.append(method_name)
            try:
                text = method_func(filepath, max_chars, result)
                if text and len(text.strip()) > 50:
                    result.readable = True
                    result.text_extracted = True
                    result.text_length = len(text)
                    result.method_used = method_name
                    break
                else:
                    if text is not None and len(text.strip()) <= 50:
                        result.failure_reasons.append(
                            f"{method_name}: extracted only {len(text.strip())} chars (below 50-char threshold)")
                    text = None
            except Exception as e:
                err_msg = str(e)[:200]
                result.failure_reasons.append(f"{method_name}: {err_msg}")
                self.log.log(TRACE, f"Text extraction {method_name} failed for {filepath}: {e}")

        if not text:
            # Log to unreadable file log
            unreadable_log = logging.getLogger('ebook_pipeline.unreadable')
            reasons = '; '.join(result.failure_reasons) if result.failure_reasons else 'no text extracted'
            unreadable_log.warning(f"UNREADABLE: {filepath} | {result.status_label} | {reasons}")

        return text, result

    def _extract_epub(self, filepath, max_chars, result: ReadabilityResult):
        if self.dep_status and not self.dep_status.module_ok('ebooklib'):
            result.failure_reasons.append("epub-native: ebooklib module not available")
            return None
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup

        try:
            book = epub.read_epub(filepath, options={'ignore_ncx': True})
        except Exception as e:
            if 'encrypted' in str(e).lower() or 'drm' in str(e).lower():
                result.is_encrypted = True
                result.failure_reasons.append(f"epub-native: encrypted/DRM — {e}")
            else:
                result.is_corrupted = True
                result.failure_reasons.append(f"epub-native: failed to open — {e}")
            return None

        text_parts, total, pages_read = [], 0, 0
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        result.total_pages = len(items)

        for item in items:
            if total >= max_chars or pages_read >= self.max_pages:
                break
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            text_parts.append(text)
            total += len(text)
            pages_read += 1

        result.pages_read = pages_read
        return '\n\n'.join(text_parts)[:max_chars] if text_parts else None

    def _extract_pdf(self, filepath, max_chars, result: ReadabilityResult):
        if self.dep_status and not self.dep_status.module_ok('fitz'):
            result.failure_reasons.append("pdf-pymupdf: fitz/PyMuPDF module not available")
            return None
        import fitz

        try:
            doc = fitz.open(filepath)
        except Exception as e:
            if 'password' in str(e).lower() or 'encrypted' in str(e).lower():
                result.is_encrypted = True
                result.failure_reasons.append(f"pdf-pymupdf: password protected — {e}")
            else:
                result.is_corrupted = True
                result.failure_reasons.append(f"pdf-pymupdf: failed to open — {e}")
            return None

        result.total_pages = len(doc)
        pages_to_read = min(self.max_pages, len(doc))
        text_parts, total, pages_with_text = [], 0, 0

        try:
            for page_num in range(pages_to_read):
                if total >= max_chars:
                    break
                try:
                    text = doc[page_num].get_text()
                except Exception as e:
                    self.log.log(TRACE, f"PDF page {page_num} extraction error: {e}")
                    continue
                text_parts.append(text)
                total += len(text)
                if text.strip():
                    pages_with_text += 1
        finally:
            doc.close()
        result.pages_read = pages_to_read

        # Detect scanned/image-only PDFs
        if pages_to_read > 0 and pages_with_text == 0:
            result.is_scanned_pdf = True
            result.failure_reasons.append(
                f"pdf-pymupdf: scanned/image-only PDF — 0/{pages_to_read} pages had text")
            return None
        elif pages_to_read >= 3 and pages_with_text < pages_to_read * 0.2:
            # Less than 20% of pages have text — likely mostly scanned
            result.is_scanned_pdf = True
            result.failure_reasons.append(
                f"pdf-pymupdf: mostly scanned — only {pages_with_text}/{pages_to_read} pages had text")

        full_text = '\n\n'.join(text_parts)[:max_chars]
        return full_text if full_text.strip() else None

    def _extract_mobi(self, filepath, max_chars, result: ReadabilityResult):
        if self.dep_status and not self.dep_status.module_ok('mobi'):
            result.failure_reasons.append("mobi-native: mobi module not available")
            return self._extract_via_calibre(filepath, max_chars, result)
        try:
            import mobi
            tempdir, extracted = mobi.extract(filepath)
            for root, dirs, files in os.walk(tempdir):
                for f in files:
                    if f.endswith(('.html', '.htm')):
                        text = self._extract_html(os.path.join(root, f), max_chars, result)
                        if text:
                            return text
            for root, dirs, files in os.walk(tempdir):
                for f in files:
                    if f.endswith('.txt'):
                        text = self._extract_text(os.path.join(root, f), max_chars, result)
                        if text:
                            return text
        except Exception as e:
            result.failure_reasons.append(f"mobi-native: {e}")
        return None

    def _extract_fb2(self, filepath, max_chars, result: ReadabilityResult):
        try:
            tree = ET.parse(filepath)
        except ET.ParseError as e:
            result.is_corrupted = True
            result.failure_reasons.append(f"fb2-native: XML parse error — {e}")
            return None

        root = tree.getroot()
        ns = re.match(r'\{.*\}', root.tag)
        ns = ns.group(0) if ns else ''
        text_parts, pages_read = [], 0

        for body in root.iter(f'{ns}body'):
            for p in body.iter(f'{ns}p'):
                text = ''.join(p.itertext()).strip()
                if text:
                    text_parts.append(text)
                    pages_read += 1
                if sum(len(t) for t in text_parts) >= max_chars:
                    break

        result.pages_read = pages_read
        return '\n'.join(text_parts)[:max_chars] if text_parts else None

    def _extract_text(self, filepath, max_chars, result: ReadabilityResult):
        for enc in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    text = f.read(max_chars)
                    result.pages_read = 1
                    return text
            except (UnicodeDecodeError, UnicodeError):
                continue
        result.failure_reasons.append("text-native: all encodings failed (utf-8, latin-1, cp1252, ascii)")
        return None

    def _extract_html(self, filepath, max_chars, result: ReadabilityResult):
        from bs4 import BeautifulSoup
        raw = None
        for enc in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    raw = f.read(max_chars * 2)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if not raw:
            result.failure_reasons.append("html-native: all encodings failed")
            return None
        soup = BeautifulSoup(raw, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)[:max_chars]
        result.pages_read = 1
        return text

    def _extract_via_calibre(self, filepath, max_chars, result: ReadabilityResult):
        if self.dep_status and not self.dep_status.tool_ok('ebook-convert'):
            result.failure_reasons.append("calibre-convert: ebook-convert tool not available")
            return None

        tmp_txt = filepath + '.tmp_extract.txt'
        try:
            rc, out, err = run_command(
                ['ebook-convert', filepath, tmp_txt, '--txt-output-formatting=plain'],
                timeout=120)
            if rc == 0 and os.path.exists(tmp_txt):
                text = None
                for enc in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(tmp_txt, 'r', encoding=enc) as f:
                            text = f.read(max_chars)
                        break
                    except (UnicodeDecodeError, UnicodeError):
                        continue
                result.pages_read = 1
                return text
            else:
                err_short = (err or 'unknown error')[:150]
                if 'drm' in err_short.lower() or 'encrypted' in err_short.lower():
                    result.is_encrypted = True
                result.failure_reasons.append(f"calibre-convert: rc={rc} — {err_short}")
                return None
        finally:
            if os.path.exists(tmp_txt):
                os.remove(tmp_txt)


# =============================================================================
# EMBEDDED METADATA EXTRACTOR
# =============================================================================

class EmbeddedMetadataExtractor:
    def __init__(self, logger: logging.Logger, dep_status: Optional[DependencyStatus] = None):
        self.log = logger
        self.dep_status = dep_status

    def extract(self, filepath: str, timeout: int = 60) -> BookMetadata:
        ext = Path(filepath).suffix.lower()
        meta = BookMetadata()
        calibre_meta = self._extract_calibre(filepath, timeout=timeout)
        if calibre_meta:
            meta.merge(calibre_meta, "calibre-embedded")
        extractors = {'.epub': self._extract_epub_meta, '.pdf': self._extract_pdf_meta, '.fb2': self._extract_fb2_meta}
        specific = extractors.get(ext)
        if specific:
            try:
                specific_meta = specific(filepath)
                if specific_meta:
                    meta.merge(specific_meta, f"{ext}-embedded")
            except Exception as e:
                self.log.log(TRACE, f"Format-specific extraction failed for {filepath}: {e}")
        return meta

    def _extract_calibre(self, filepath: str, timeout: int = 60) -> Optional[BookMetadata]:
        rc, out, err = run_command(['ebook-meta', filepath], timeout=timeout)
        if rc != 0:
            return None
        meta = BookMetadata()
        field_map = {
            'Title': 'title', 'Author(s)': '_authors', 'Publisher': 'publisher',
            'Language': 'language', 'Published': 'publication_date',
            'Identifiers': '_identifiers', 'Tags': '_tags', 'Series': '_series',
            'Comments': 'description', 'Rights': 'rights',
        }
        for line in out.strip().split('\n'):
            if ':' not in line: continue
            key, _, value = line.partition(':')
            key, value = key.strip(), value.strip()
            if not value or value.lower() == 'unknown': continue
            mapped = field_map.get(key)
            if not mapped: continue
            if mapped == '_authors':
                raw = re.sub(r'\s*\[.*?\]\s*', ' ', value)
                raw = re.sub(r'[\[\]]', '', raw)
                raw = re.sub(r'\s+', ' ', raw).strip()
                raw_authors = [a.strip() for a in re.split(r'\s*[;&]\s*', raw) if a.strip()]
                meta.authors = deduplicate_authors(raw_authors)
            elif mapped == '_identifiers':
                for ident in value.split(','):
                    ident = ident.strip()
                    if ':' in ident:
                        id_type, _, id_val = ident.partition(':')
                        id_type, id_val = id_type.strip().lower(), id_val.strip()
                        if id_type == 'isbn' and len(id_val) == 13: meta.isbn_13 = id_val
                        elif id_type == 'isbn' and len(id_val) == 10: meta.isbn_10 = id_val
                        elif id_type == 'isbn10': meta.isbn_10 = id_val
                        elif id_type == 'google': meta.google_books_id = id_val
                        elif id_type == 'amazon': meta.asin = id_val
                        elif id_type == 'doi': meta.doi = id_val
                        elif id_type == 'gutenberg': meta.gutenberg_id = id_val
                        elif id_type == 'openlibrary': meta.openlibrary_id = id_val
            elif mapped == '_tags':
                raw_tags = [t.strip() for t in value.split(',') if t.strip()]
                meta.tags = list(dict.fromkeys(raw_tags))  # Dedupe preserving order
            elif mapped == '_series':
                m = re.match(r'(.+?)(?:\s*#\s*(\d+(?:\.\d+)?))?$', value)
                if m:
                    meta.series = m.group(1).strip()
                    if m.group(2): meta.series_index = float(m.group(2))
            else:
                setattr(meta, mapped, value)
        return meta

    def _extract_epub_meta(self, filepath: str) -> Optional[BookMetadata]:
        if self.dep_status and not self.dep_status.module_ok('ebooklib'):
            return None
        import ebooklib
        from ebooklib import epub
        meta = BookMetadata()
        book = epub.read_epub(filepath, options={'ignore_ncx': True})
        titles = book.get_metadata('DC', 'title')
        if titles: meta.title = titles[0][0]
        for creator in book.get_metadata('DC', 'creator'):
            name = creator[0]
            role = creator[1].get('{http://www.idpf.org/2007/opf}role', 'aut')
            if role == 'aut': meta.authors.append(name)
            elif role == 'edt': meta.editors.append(name)
            elif role == 'trl': meta.translators.append(name)
            elif role == 'ill': meta.illustrators.append(name)
            else: meta.contributors.append(name)
        langs = book.get_metadata('DC', 'language')
        if langs: meta.language = langs[0][0]
        dates = book.get_metadata('DC', 'date')
        if dates: meta.publication_date = dates[0][0]
        pubs = book.get_metadata('DC', 'publisher')
        if pubs: meta.publisher = pubs[0][0]
        descs = book.get_metadata('DC', 'description')
        if descs: meta.description = descs[0][0]
        for subj in book.get_metadata('DC', 'subject'):
            meta.subjects.append(subj[0])
        rights = book.get_metadata('DC', 'rights')
        if rights: meta.rights = rights[0][0]
        for ident in book.get_metadata('DC', 'identifier'):
            val = ident[0]
            scheme = ident[1].get('{http://www.idpf.org/2007/opf}scheme', '').lower()
            if scheme == 'isbn' or re.match(r'^97[89]\d{10}$', val):
                if len(val) == 13: meta.isbn_13 = val
                elif len(val) == 10: meta.isbn_10 = val
            elif 'gutenberg' in val.lower() or 'gutenberg' in scheme:
                gid = re.search(r'(\d+)', val)
                if gid: meta.gutenberg_id = gid.group(1)
        return meta

    def _extract_pdf_meta(self, filepath: str) -> Optional[BookMetadata]:
        if self.dep_status and not self.dep_status.module_ok('fitz'):
            return None
        import fitz
        meta = BookMetadata()
        doc = fitz.open(filepath)
        pdf_meta = doc.metadata
        if pdf_meta:
            meta.title = pdf_meta.get('title') or None
            author = pdf_meta.get('author')
            if author:
                raw_authors = [a.strip() for a in re.split(r'[,;&]', author) if a.strip()]
                meta.authors = deduplicate_authors(raw_authors)
            meta.description = pdf_meta.get('subject') or None
            if pdf_meta.get('keywords'):
                raw_tags = [k.strip() for k in pdf_meta['keywords'].split(',') if k.strip()]
                meta.tags = list(dict.fromkeys(raw_tags))  # Dedupe preserving order
            meta.publisher = pdf_meta.get('producer') or pdf_meta.get('creator') or None
            if pdf_meta.get('creationDate'):
                dm = re.search(r'(\d{4})', pdf_meta['creationDate'])
                if dm: meta.publication_date = dm.group(1)
        meta.page_count = len(doc)
        doc.close()
        return meta

    def _extract_fb2_meta(self, filepath: str) -> Optional[BookMetadata]:
        meta = BookMetadata()
        tree = ET.parse(filepath)
        root = tree.getroot()
        ns_match = re.match(r'\{(.*)\}', root.tag)
        ns = ns_match.group(1) if ns_match else ''
        nsmap = {'fb': ns} if ns else {}
        def find(parent, tag):
            return parent.find(f'fb:{tag}', nsmap) if ns else parent.find(tag)
        def findall(parent, tag):
            return parent.findall(f'fb:{tag}', nsmap) if ns else parent.findall(tag)
        def findtext(parent, tag):
            el = find(parent, tag)
            return el.text.strip() if el is not None and el.text else None
        desc = find(root, 'description')
        if desc is None: return meta
        title_info = find(desc, 'title-info')
        if title_info is not None:
            bt = find(title_info, 'book-title')
            if bt is not None and bt.text: meta.title = bt.text.strip()
            for author_el in findall(title_info, 'author'):
                parts = []
                for tag in ['first-name', 'middle-name', 'last-name']:
                    el = find(author_el, tag)
                    if el is not None and el.text: parts.append(el.text.strip())
                if parts: meta.authors.append(' '.join(parts))
            lang = findtext(title_info, 'lang')
            if lang: meta.language = lang
            for genre_el in findall(title_info, 'genre'):
                if genre_el.text: meta.genres.append(genre_el.text.strip())
            ann = find(title_info, 'annotation')
            if ann is not None: meta.description = ''.join(ann.itertext()).strip()
            date_el = find(title_info, 'date')
            if date_el is not None:
                meta.publication_date = date_el.get('value') or (date_el.text or '').strip()
            seq = find(title_info, 'sequence')
            if seq is not None:
                meta.series = seq.get('name')
                num = seq.get('number')
                if num:
                    try: meta.series_index = float(num)
                    except ValueError: pass
        publish_info = find(desc, 'publish-info')
        if publish_info is not None:
            meta.publisher = findtext(publish_info, 'publisher')
            meta.isbn_13 = findtext(publish_info, 'isbn')
            year = findtext(publish_info, 'year')
            if year and not meta.publication_date: meta.publication_date = year
        return meta


# =============================================================================
# GUTENBERG RDF CATALOG
# =============================================================================

class GutenbergRDFCatalog:
    def __init__(self, rdf_dir: str, logger: logging.Logger):
        self.rdf_dir = rdf_dir
        self.log = logger
        self._cache: Dict[str, BookMetadata] = {}

    def lookup(self, gutenberg_id: str) -> Optional[BookMetadata]:
        if gutenberg_id in self._cache:
            return self._cache[gutenberg_id]
        rdf_path = os.path.join(self.rdf_dir, 'cache', 'epub', gutenberg_id, f'pg{gutenberg_id}.rdf')
        if not os.path.exists(rdf_path):
            for alt in [
                os.path.join(self.rdf_dir, gutenberg_id, f'pg{gutenberg_id}.rdf'),
                os.path.join(self.rdf_dir, f'pg{gutenberg_id}.rdf'),
                os.path.join(self.rdf_dir, 'epub', gutenberg_id, f'pg{gutenberg_id}.rdf'),
            ]:
                if os.path.exists(alt):
                    rdf_path = alt
                    break
            else:
                return None
        try:
            meta = self._parse_rdf(rdf_path, gutenberg_id)
            self._cache[gutenberg_id] = meta
            return meta
        except Exception as e:
            self.log.error(f"Failed to parse RDF for PG#{gutenberg_id}: {e}")
            return None

    def _parse_rdf(self, rdf_path: str, gutenberg_id: str) -> BookMetadata:
        tree = ET.parse(rdf_path)
        root = tree.getroot()
        meta = BookMetadata()
        meta.gutenberg_id = gutenberg_id
        ebook = root.find('.//pgterms:ebook', RDF_NS)
        if ebook is None:
            for child in root:
                if 'ebook' in child.tag.lower():
                    ebook = child
                    break
        if ebook is None:
            return meta

        title_el = ebook.find('dcterms:title', RDF_NS)
        if title_el is not None and title_el.text:
            full_title = title_el.text.strip()
            if '\n' in full_title:
                parts = full_title.split('\n', 1)
                meta.title, meta.subtitle = parts[0].strip(), parts[1].strip()
            elif ': ' in full_title and len(full_title.split(': ', 1)[0]) > 3:
                parts = full_title.split(': ', 1)
                meta.title, meta.subtitle = parts[0].strip(), parts[1].strip()
            else:
                meta.title = full_title

        for creator in ebook.findall('dcterms:creator', RDF_NS):
            agent = creator.find('pgterms:agent', RDF_NS)
            if agent is not None:
                name_el = agent.find('pgterms:name', RDF_NS)
                if name_el is not None and name_el.text:
                    meta.authors.append(name_el.text.strip())

        for role_ns, field_name in {'marcrel:edt': 'editors', 'marcrel:trl': 'translators',
                                     'marcrel:ill': 'illustrators', 'marcrel:ctb': 'contributors'}.items():
            prefix, tag = role_ns.split(':')
            for el in ebook.findall(f'{prefix}:{tag}', RDF_NS):
                agent = el.find('pgterms:agent', RDF_NS)
                if agent is not None:
                    name_el = agent.find('pgterms:name', RDF_NS)
                    if name_el is not None and name_el.text:
                        getattr(meta, field_name).append(name_el.text.strip())

        lang_el = ebook.find('.//dcterms:language//rdf:value', RDF_NS)
        if lang_el is not None and lang_el.text:
            meta.language = lang_el.text.strip()

        for subject in ebook.findall('dcterms:subject', RDF_NS):
            desc = subject.find('rdf:Description', RDF_NS)
            if desc is not None:
                member_of = desc.find('dcam:memberOf', RDF_NS)
                value = desc.find('rdf:value', RDF_NS)
                if value is not None and value.text:
                    val = value.text.strip()
                    if member_of is not None:
                        resource = member_of.get(f'{{{RDF_NS["rdf"]}}}resource', '')
                        if 'LCSH' in resource: meta.subjects.append(val)
                        elif 'LCC' in resource: meta.lcc = val
                    else:
                        meta.subjects.append(val)

        rights_el = ebook.find('dcterms:rights', RDF_NS)
        if rights_el is not None and rights_el.text:
            meta.rights = rights_el.text.strip()
        license_el = ebook.find('cc:license', RDF_NS)
        if license_el is not None:
            meta.license_url = license_el.get(f'{{{RDF_NS["rdf"]}}}resource', '')
        issued = ebook.find('dcterms:issued', RDF_NS)
        if issued is not None and issued.text:
            meta.publication_date = issued.text.strip()
        publisher_el = ebook.find('dcterms:publisher', RDF_NS)
        if publisher_el is not None and publisher_el.text:
            meta.publisher = publisher_el.text.strip()
        desc_el = ebook.find('dcterms:description', RDF_NS)
        if desc_el is not None and desc_el.text:
            meta.description = desc_el.text.strip()
        return meta


# =============================================================================
# PUBLIC API LOOKUPS
# =============================================================================

class PublicAPILookup:
    GOOGLE_RATE_LIMIT = 1.0
    OL_RATE_LIMIT = 0.3
    MAX_RETRIES = 2
    BACKOFF_BASE = 1.5
    CIRCUIT_BREAK_THRESHOLD = 3
    REQUEST_TIMEOUT = 8

    def __init__(self, logger, google_api_keys=None):
        self.log = logger
        self.google_api_keys = google_api_keys or []
        self._google_key_index = 0
        self._google_key_lock = threading.Lock()
        self._disabled_google_keys = set()
        self._session = None
        self._session_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._last_google_call = 0.0
        self._last_ol_call = 0.0
        self._google_backoff = 0
        self._ol_backoff = 0
        self._google_consecutive_429s = 0
        self._ol_consecutive_429s = 0
        self._google_disabled = False
        self._ol_disabled = False
        self._google_disable_reason = ""

    @property
    def session(self):
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    import requests
                    self._session = requests.Session()
                    self._session.headers.update({'User-Agent': 'EbookMetadataPipeline/2.0'})
        return self._session

    def _next_google_key(self):
        if not self.google_api_keys: return None
        with self._google_key_lock:
            total = len(self.google_api_keys)
            for _ in range(total):
                key = self.google_api_keys[self._google_key_index % total]
                self._google_key_index += 1
                if key not in self._disabled_google_keys:
                    return key
            return None

    def _disable_google_key(self, key):
        with self._google_key_lock:
            self._disabled_google_keys.add(key)
            remaining = len(self.google_api_keys) - len(self._disabled_google_keys)
            self.log.warning(f"Disabled Google API key ...{key[-8:]} -- {remaining} remaining")
            if remaining == 0:
                self._google_disabled = True
                self._google_disable_reason = "all API keys exhausted (403)"

    def _clean_author(self, author):
        if not author: return author
        author = re.sub(r'\s*\[.*?\]\s*', ' ', author).strip()
        author = re.sub(r'\s+', ' ', author).strip()
        if ',' in author and author.count(',') == 1:
            parts = author.split(',', 1)
            author = f"{parts[1].strip()} {parts[0].strip()}"
        return author

    def _rate_limit(self, api):
        with self._rate_lock:
            now = time.time()
            if api == 'google':
                delay = self.GOOGLE_RATE_LIMIT + (self.BACKOFF_BASE ** self._google_backoff - 1 if self._google_backoff else 0)
                elapsed = now - self._last_google_call
                if elapsed < delay: time.sleep(delay - elapsed)
                self._last_google_call = time.time()
            elif api == 'openlibrary':
                delay = self.OL_RATE_LIMIT + (self.BACKOFF_BASE ** self._ol_backoff - 1 if self._ol_backoff else 0)
                elapsed = now - self._last_ol_call
                if elapsed < delay: time.sleep(delay - elapsed)
                self._last_ol_call = time.time()

    def _request_with_retry(self, api, method, url, **kwargs):
        import requests
        if api == 'google' and self._google_disabled: return None
        if api == 'openlibrary' and self._ol_disabled: return None
        for attempt in range(self.MAX_RETRIES + 1):
            self._rate_limit(api)
            try:
                kwargs.setdefault('timeout', self.REQUEST_TIMEOUT)
                resp = self.session.request(method, url, **kwargs)

                if resp.status_code == 429:
                    if api == 'google':
                        self._google_consecutive_429s += 1
                        if self._google_consecutive_429s >= self.CIRCUIT_BREAK_THRESHOLD:
                            self._google_disabled = True
                            self._google_disable_reason = "rate limited (too many 429s)"
                            return None
                    else:
                        self._ol_consecutive_429s += 1
                        if self._ol_consecutive_429s >= self.CIRCUIT_BREAK_THRESHOLD:
                            self._ol_disabled = True
                            return None
                    wait = self.BACKOFF_BASE ** (attempt + 1)
                    ra = resp.headers.get('Retry-After')
                    if ra:
                        try: wait = min(int(ra), 10)  # Cap retry-after at 10s
                        except ValueError: pass
                    if attempt < self.MAX_RETRIES:
                        time.sleep(wait)
                        continue
                    return None

                if resp.status_code == 403:
                    if api == 'google':
                        ck = kwargs.get('params', {}).get('key')
                        if ck:
                            self._disable_google_key(ck)
                        else:
                            # No API key — count 403s like 429s instead of instant kill
                            self._google_consecutive_429s += 1
                            if self._google_consecutive_429s >= self.CIRCUIT_BREAK_THRESHOLD:
                                self._google_disabled = True
                                self._google_disable_reason = "access denied (no API key)"
                        return None
                    return None

                # Don't retry other 4xx errors (400, 404, etc.)
                if 400 <= resp.status_code < 500:
                    return None

                resp.raise_for_status()

                # Success — reset backoff counters
                if api == 'google':
                    self._google_consecutive_429s = 0
                    self._google_backoff = max(0, self._google_backoff - 1)
                else:
                    self._ol_consecutive_429s = 0
                    self._ol_backoff = max(0, self._ol_backoff - 1)
                return resp

            except requests.exceptions.HTTPError as e:
                # 5xx server errors — worth retrying
                sc = e.response.status_code if e.response is not None else 0
                if sc >= 500 and attempt < self.MAX_RETRIES:
                    time.sleep(self.BACKOFF_BASE ** attempt)
                    continue
                return None
            except requests.exceptions.Timeout:
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.BACKOFF_BASE ** attempt)
                    continue
                return None
            except requests.exceptions.ConnectionError:
                return None
        return None

    def _simplify_title(self, title):
        s = re.split(r'\s*[:]\s*', title)[0]
        s = re.sub(r'\s*\d+(?:st|nd|rd|th)\s*(?:ed|edition)\b', '', s, flags=re.IGNORECASE)
        s = re.sub(r'\s*\([^)]*\)\s*', ' ', s)
        return s.strip()

    def _title_matches(self, result_title, search_title, author=None):
        if not result_title or not search_title: return False
        rt = re.sub(r'[^\w\s]', '', result_title.lower()).split()
        st = re.sub(r'[^\w\s]', '', search_title.lower()).split()
        if not rt or not st: return False
        stop = {'the', 'a', 'an', 'of', 'and', 'in', 'to', 'for', 'with', 'on', 'by', 'is', 'at'}
        rt_w = {w for w in rt if w not in stop and len(w) > 2}
        st_w = {w for w in st if w not in stop and len(w) > 2}
        if not rt_w or not st_w: return True
        overlap = rt_w & st_w
        sr = len(overlap) / len(st_w) if st_w else 0
        rr = len(overlap) / len(rt_w) if rt_w else 0
        return sr >= 0.4 and rr >= 0.3

    def search(self, title=None, author=None, isbn=None, current_completeness=0.0):
        meta = BookMetadata()
        meta._api_diagnostics = []
        if author: author = self._clean_author(author)
        search_titles = []
        if title:
            for t in [title, re.split(r'\s*:\s*', title)[0].strip(), self._simplify_title(title)]:
                if t and len(t) >= 4 and t not in search_titles and len(t.split()) >= 2:
                    search_titles.append(t)
            if not search_titles and title and len(title) >= 4:
                search_titles = [title]

        ol_got, ol_fields = False, []
        def _track_ol(m):
            f = []
            if m.isbn_13 or m.isbn_10: f.append('ISBN')
            if m.language: f.append('language')
            if m.description: f.append('description')
            if m.page_count: f.append('pages')
            if m.cover_url: f.append('cover')
            if m.subjects: f.append('subjects')
            if m.genres: f.append('genres')
            if m.publisher: f.append('publisher')
            if m.lcc: f.append('LCC')
            if m.ddc: f.append('DDC')
            if m.original_publication_date: f.append('orig_date')
            if m.authors: f.append('authors')
            return f

        if isbn:
            ol_m = self._openlibrary_isbn(isbn)
            if ol_m:
                ol_fields = _track_ol(ol_m)
                meta.merge(ol_m, "openlibrary-isbn")
                ol_got = True
        if search_titles and not (meta.subjects and meta.description and meta.page_count):
            for st in search_titles:
                ol_m = self._openlibrary_search(st, author)
                if ol_m and ol_m.title and self._title_matches(ol_m.title, title, author):
                    nf = _track_ol(ol_m)
                    for f in nf:
                        if f not in ol_fields: ol_fields.append(f)
                    meta.merge(ol_m, "openlibrary-search")
                    ol_got = True
                    break

        if ol_got:
            meta._api_diagnostics.append(('Open Library', 'success', ', '.join(ol_fields) if ol_fields else 'matched'))
        else:
            meta._api_diagnostics.append(('Open Library', 'no_match', 'no results'))

        est = max(current_completeness, meta.completeness_score()) if ol_got else current_completeness
        need_google, google_reason = False, ""
        if not ol_got:
            need_google, google_reason = True, "OL had no match"
        elif est < 0.7:
            need_google, google_reason = True, f"OL only reached {est:.0%}"
        else:
            meta._api_diagnostics.append(('Google Books', 'skipped', f'OL sufficient at {est:.0%}'))

        if need_google:
            gg, gf = False, []
            def _track_gg(m):
                f = []
                if m.isbn_13 or m.isbn_10: f.append('ISBN')
                if m.language: f.append('language')
                if m.description: f.append('description')
                if m.page_count: f.append('pages')
                if m.cover_url: f.append('cover')
                if m.subjects: f.append('subjects')
                if m.genres: f.append('genres')
                if m.publisher: f.append('publisher')
                if m.authors: f.append('authors')
                return f
            if isbn:
                gb = self._google_books_isbn(isbn)
                if gb:
                    gf = _track_gg(gb)
                    meta.merge(gb, "google-books-isbn")
                    gg = True
            if search_titles and not (meta.description and meta.page_count):
                for st in search_titles:
                    gb = self._google_books_search(st, author)
                    if gb and gb.title and self._title_matches(gb.title, title, author):
                        nf = _track_gg(gb)
                        for f in nf:
                            if f not in gf: gf.append(f)
                        meta.merge(gb, "google-books-search")
                        gg = True
                        break
            if gg:
                meta._api_diagnostics.append(('Google Books', 'success', ', '.join(gf) if gf else 'matched'))
            elif self._google_disabled:
                reason = self._google_disable_reason or 'disabled'
                meta._api_diagnostics.append(('Google Books', 'error', reason))
            else:
                meta._api_diagnostics.append(('Google Books', 'no_match', f'no results ({google_reason})'))
        return meta

    def _openlibrary_isbn(self, isbn):
        resp = self._request_with_retry('openlibrary', 'GET', OPENLIBRARY_SEARCH_URL,
                                        params={'isbn': isbn, 'fields': '*', 'limit': 1})
        if resp:
            data = resp.json()
            if data.get('docs'): return self._parse_openlibrary(data['docs'][0])
        return None

    def _openlibrary_search(self, title, author=None):
        params = {'title': title, 'fields': '*', 'limit': 3}
        if author: params['author'] = author
        resp = self._request_with_retry('openlibrary', 'GET', OPENLIBRARY_SEARCH_URL, params=params)
        if resp:
            data = resp.json()
            if data.get('docs'):
                best = self._best_match(data['docs'], title, author)
                if best: return self._parse_openlibrary(best)
        return None

    def _parse_openlibrary(self, doc):
        meta = BookMetadata()
        meta.title = doc.get('title')
        meta.subtitle = doc.get('subtitle')
        meta.authors = doc.get('author_name', [])
        meta.publisher = (doc.get('publisher') or [None])[0]
        meta.language = (doc.get('language') or [None])[0]
        fp = doc.get('first_publish_year')
        if fp:
            meta.original_publication_date = str(fp)
            if not meta.publication_date: meta.publication_date = str(fp)
        for isbn in doc.get('isbn', []):
            if len(isbn) == 13 and not meta.isbn_13: meta.isbn_13 = isbn
            elif len(isbn) == 10 and not meta.isbn_10: meta.isbn_10 = isbn
        meta.oclc = (doc.get('oclc') or [None])[0]
        meta.lccn = (doc.get('lccn') or [None])[0]
        ol_key = doc.get('key')
        if ol_key: meta.openlibrary_id = ol_key.replace('/works/', '')
        meta.subjects = doc.get('subject', [])[:20]
        meta.lcc = (doc.get('lcc') or [None])[0]
        meta.ddc = (doc.get('ddc') or [None])[0]
        pages = doc.get('number_of_pages_median')
        if pages: meta.page_count = int(pages)
        cover_id = doc.get('cover_i')
        if cover_id: meta.cover_url = f"https://covers.openlibrary.org/b/id/{cover_id}-L.jpg"
        if ol_key:
            try:
                resp = self._request_with_retry('openlibrary', 'GET', f"https://openlibrary.org{ol_key}.json")
                if resp:
                    work = resp.json()
                    desc = work.get('description')
                    if isinstance(desc, dict): meta.description = desc.get('value', '')
                    elif isinstance(desc, str): meta.description = desc
                    if not meta.subjects:
                        meta.subjects = [s for s in work.get('subjects', [])[:20]]
            except Exception: pass
        return meta

    def _google_books_isbn(self, isbn):
        if self._google_disabled:
            return None
        params = {'q': f'isbn:{isbn}', 'maxResults': 1}
        gk = self._next_google_key()
        if gk: params['key'] = gk
        resp = self._request_with_retry('google', 'GET', GOOGLE_BOOKS_URL, params=params)
        if resp:
            data = resp.json()
            if data.get('items'): return self._parse_google_books(data['items'][0])
        return None

    def _google_books_search(self, title, author=None):
        if self._google_disabled:
            return None
        st = title[:100]
        words = st.split()
        query = f'intitle:"{st}"' if len(words) <= 6 else f'intitle:"{" ".join(words[:5])}" {" ".join(words[5:])}'
        if author: query += f' inauthor:"{author}"'
        params = {'q': query, 'maxResults': 3}
        gk = self._next_google_key()
        if gk: params['key'] = gk
        resp = self._request_with_retry('google', 'GET', GOOGLE_BOOKS_URL, params=params)
        if resp:
            data = resp.json()
            if data.get('items'): return self._parse_google_books(data['items'][0])
            # Got a valid response but no items — try without author only if we had one
            if author and not self._google_disabled:
                params2 = {'q': f'intitle:"{st[:60]}"', 'maxResults': 3}
                gk2 = self._next_google_key()
                if gk2: params2['key'] = gk2
                resp2 = self._request_with_retry('google', 'GET', GOOGLE_BOOKS_URL, params=params2)
                if resp2:
                    data2 = resp2.json()
                    if data2.get('items'): return self._parse_google_books(data2['items'][0])
        # If resp was None (error/timeout/disabled), don't retry with a variant
        return None

    def _parse_google_books(self, item):
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
        meta.genres = vol.get('categories', [])[:10]
        for ident in vol.get('industryIdentifiers', []):
            if ident['type'] == 'ISBN_13': meta.isbn_13 = ident['identifier']
            elif ident['type'] == 'ISBN_10': meta.isbn_10 = ident['identifier']
        images = vol.get('imageLinks', {})
        meta.cover_url = images.get('thumbnail') or images.get('smallThumbnail')
        return meta

    def _best_match(self, docs, title, author):
        if not docs: return None
        tl = title.lower().strip()
        for doc in docs:
            dt = (doc.get('title') or '').lower().strip()
            if dt == tl:
                if author:
                    da = [a.lower() for a in doc.get('author_name', [])]
                    if any(author.lower() in a for a in da): return doc
                else: return doc
        return docs[0]


# =============================================================================
# AI TEXT ANALYSIS (with fallback for unreadable files)
# =============================================================================

class AIMetadataExtractor:
    def __init__(self, logger, api_key=None, dep_status=None):
        self.log = logger
        self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.dep_status = dep_status

    def available(self) -> bool:
        if not self.api_key:
            return False
        if self.dep_status and not self.dep_status.module_ok('anthropic'):
            return False
        return True

    def analyze(self, text, filename, existing_meta=None):
        if not self.available():
            self.log.warning("No Anthropic API key or module available, skipping AI analysis")
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            existing_info = ""
            if existing_meta and existing_meta.title:
                existing_info = f"\nPartially known: title='{existing_meta.title}', authors={existing_meta.authors}"
            prompt = f"""Analyze this text from ebook "{filename}" and extract metadata.{existing_info}

Return JSON only: {{"title": "...", "subtitle": null, "authors": [], "editors": [], "translators": [],
"illustrators": [], "publisher": null, "publication_date": null, "original_publication_date": null,
"edition": null, "language": "ISO code", "description": "2-3 sentences", "subjects": [], "genres": [],
"series": null, "series_index": null, "isbn_10": null, "isbn_13": null}}

TEXT ({len(text)} chars):
{text}"""
            response = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=2000,
                messages=[{"role": "user", "content": prompt}])
            rt = response.content[0].text.strip()
            rt = re.sub(r'^```json\s*', '', rt)
            rt = re.sub(r'\s*```$', '', rt)
            data = json.loads(rt)
            meta = BookMetadata()
            for fld in ['title', 'subtitle', 'publisher', 'publication_date',
                        'original_publication_date', 'edition', 'language',
                        'description', 'series', 'isbn_10', 'isbn_13']:
                val = data.get(fld)
                if val: setattr(meta, fld, val)
            for fld in ['authors', 'editors', 'translators', 'illustrators', 'subjects', 'genres']:
                val = data.get(fld)
                if val and isinstance(val, list):
                    setattr(meta, fld, [v for v in val if v])
            si = data.get('series_index')
            if si is not None:
                try: meta.series_index = float(si)
                except (ValueError, TypeError): pass
            return meta
        except json.JSONDecodeError as e:
            self.log.error(f"AI returned invalid JSON: {e}")
            return None
        except Exception as e:
            self.log.error(f"AI analysis failed: {e}")
            return None

    def analyze_from_filename(self, filename: str, file_size: int,
                               existing_meta: Optional[BookMetadata] = None) -> Optional[BookMetadata]:
        """
        AI fallback when text extraction completely fails.
        Uses filename, file size, and any embedded metadata to infer what we can.
        """
        if not self.available():
            return None

        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            existing_info = ""
            if existing_meta:
                known = {}
                if existing_meta.title: known['title'] = existing_meta.title
                if existing_meta.authors: known['authors'] = existing_meta.authors
                if existing_meta.publisher: known['publisher'] = existing_meta.publisher
                if existing_meta.isbn_13: known['isbn_13'] = existing_meta.isbn_13
                if existing_meta.isbn_10: known['isbn_10'] = existing_meta.isbn_10
                if known:
                    existing_info = f"\nPartially known metadata: {json.dumps(known)}"

            ext = Path(filename).suffix
            prompt = f"""I have an ebook file that I cannot extract text from (possibly scanned PDF, encrypted, or corrupted).
Based on the filename and any known metadata, identify this book and provide metadata.

Filename: {filename}
Extension: {ext}
File size: {file_size / 1024:.0f} KB{existing_info}

Return JSON only: {{"title": "...", "subtitle": null, "authors": [], "publisher": null,
"publication_date": null, "language": "ISO code", "description": "2-3 sentences",
"subjects": [], "genres": [], "series": null, "series_index": null,
"isbn_10": null, "isbn_13": null, "confidence": "low|medium|high"}}

If you cannot confidently identify the book, return {{"title": null, "confidence": "low"}}."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514", max_tokens=1500,
                messages=[{"role": "user", "content": prompt}])
            rt = response.content[0].text.strip()
            rt = re.sub(r'^```json\s*', '', rt)
            rt = re.sub(r'\s*```$', '', rt)
            data = json.loads(rt)

            if data.get('confidence') == 'low' and not data.get('title'):
                return None

            meta = BookMetadata()
            for fld in ['title', 'subtitle', 'publisher', 'publication_date',
                        'language', 'description', 'series', 'isbn_10', 'isbn_13']:
                val = data.get(fld)
                if val: setattr(meta, fld, val)
            for fld in ['authors', 'subjects', 'genres']:
                val = data.get(fld)
                if val and isinstance(val, list):
                    setattr(meta, fld, [v for v in val if v])
            si = data.get('series_index')
            if si is not None:
                try: meta.series_index = float(si)
                except (ValueError, TypeError): pass
            return meta
        except Exception as e:
            self.log.error(f"AI filename analysis failed: {e}")
            return None


# =============================================================================
# METADATA WRITER
# =============================================================================

class MetadataWriter:
    def __init__(self, logger, dep_status=None):
        self.log = logger
        self._calibre_available = None
        self.dep_status = dep_status

    def calibre_available(self):
        if self._calibre_available is None:
            if self.dep_status and not self.dep_status.tool_ok('ebook-meta'):
                self._calibre_available = False
            else:
                rc, _, _ = run_command(['ebook-meta', '--version'])
                self._calibre_available = (rc == 0)
        return self._calibre_available

    def write(self, filepath, meta, timeout: int = 60):
        ext = Path(filepath).suffix.lower()
        success = False
        if self.calibre_available():
            success = self._write_calibre(filepath, meta, timeout=timeout)
        if ext == '.epub':
            success = self._write_epub_extra(filepath, meta) or success
        elif ext == '.pdf':
            success = self._write_pdf_meta(filepath, meta) or success
        return success

    def _write_calibre(self, filepath, meta, timeout: int = 60):
        cmd = ['ebook-meta', filepath]
        if meta.title: cmd.extend(['--title', meta.title])
        if meta.authors: cmd.extend(['--authors', ' & '.join(meta.authors)])
        if meta.publisher: cmd.extend(['--publisher', meta.publisher])
        if meta.publication_date:
            date = meta.publication_date
            if re.match(r'^\d{4}$', date): date = f'{date}-01-01'
            cmd.extend(['--date', date])
        if meta.language: cmd.extend(['--language', meta.language])
        if meta.description: cmd.extend(['--comments', meta.description])
        if meta.subjects or meta.genres or meta.tags:
            all_tags = list(set(meta.subjects + meta.genres + meta.tags))
            cmd.extend(['--tags', ', '.join(all_tags[:50])])
        if meta.series: cmd.extend(['--series', meta.series])
        if meta.series_index is not None: cmd.extend(['--index', str(meta.series_index)])
        if meta.isbn_13: cmd.extend(['--isbn', meta.isbn_13])
        elif meta.isbn_10: cmd.extend(['--isbn', meta.isbn_10])

        identifiers = []
        if meta.isbn_13: identifiers.append(f'isbn:{meta.isbn_13}')
        if meta.isbn_10: identifiers.append(f'isbn10:{meta.isbn_10}')
        if meta.gutenberg_id: identifiers.append(f'gutenberg:{meta.gutenberg_id}')
        if meta.google_books_id: identifiers.append(f'google:{meta.google_books_id}')
        if meta.openlibrary_id: identifiers.append(f'openlibrary:{meta.openlibrary_id}')
        if meta.asin: identifiers.append(f'amazon:{meta.asin}')
        if meta.doi: identifiers.append(f'doi:{meta.doi}')
        for ident in identifiers:
            cmd.extend(['--identifier', ident])

        cover_path = None
        if meta.cover_url:
            cover_path = self._download_cover(meta.cover_url, filepath)
            if cover_path: cmd.extend(['--cover', cover_path])

        rc, out, err = run_command(cmd, timeout=timeout)
        if cover_path and os.path.exists(cover_path):
            try: os.remove(cover_path)
            except OSError: pass
        if rc == 0: return True
        if '--identifier' in cmd:
            return self._write_calibre_minimal(filepath, meta)
        return False

    def _download_cover(self, url, filepath):
        import tempfile
        try:
            import requests
            resp = requests.get(url, timeout=15, stream=True)
            resp.raise_for_status()
            ct = resp.headers.get('content-type', '')
            ext = '.png' if 'png' in ct else ('.gif' if 'gif' in ct else '.jpg')
            fd, tmp = tempfile.mkstemp(suffix=ext, prefix='.cover_', dir=os.path.dirname(filepath))
            with os.fdopen(fd, 'wb') as f:
                for chunk in resp.iter_content(8192): f.write(chunk)
            if os.path.getsize(tmp) < 500:
                os.remove(tmp)
                return None
            return tmp
        except Exception:
            return None

    def _write_calibre_minimal(self, filepath, meta):
        cmd = ['ebook-meta', filepath]
        if meta.title: cmd.extend(['--title', meta.title])
        if meta.authors: cmd.extend(['--authors', ' & '.join(meta.authors)])
        if meta.publisher: cmd.extend(['--publisher', meta.publisher])
        if meta.language: cmd.extend(['--language', meta.language])
        if meta.description: cmd.extend(['--comments', meta.description])
        if meta.isbn_13: cmd.extend(['--isbn', meta.isbn_13])
        rc, _, _ = run_command(cmd, timeout=60)
        return rc == 0

    def _write_epub_extra(self, filepath, meta):
        if self.dep_status and not self.dep_status.module_ok('ebooklib'):
            return False
        try:
            import ebooklib
            from ebooklib import epub
            book = epub.read_epub(filepath, options={'ignore_ncx': True})
            for subject in meta.subjects:
                if subject:  # Skip None/empty
                    book.add_metadata('DC', 'subject', subject)
            for editor in meta.editors:
                if editor:
                    book.add_metadata('DC', 'contributor', editor, {'{http://www.idpf.org/2007/opf}role': 'edt'})
            for translator in meta.translators:
                if translator:
                    book.add_metadata('DC', 'contributor', translator, {'{http://www.idpf.org/2007/opf}role': 'trl'})
            for illustrator in meta.illustrators:
                if illustrator:
                    book.add_metadata('DC', 'contributor', illustrator, {'{http://www.idpf.org/2007/opf}role': 'ill'})
            if meta.rights: book.add_metadata('DC', 'rights', meta.rights)
            if meta.gutenberg_id:
                book.add_metadata('DC', 'source', f'https://www.gutenberg.org/ebooks/{meta.gutenberg_id}')
            epub.write_epub(filepath, book)
            return True
        except Exception as e:
            self.log.warning(f"EPUB extra metadata write failed: {e}")
            return False

    def _write_pdf_meta(self, filepath, meta):
        if self.dep_status and not self.dep_status.module_ok('fitz'):
            return False
        try:
            import fitz
            doc = fitz.open(filepath)
            doc.set_metadata({
                'title': meta.title or '', 'author': ', '.join(meta.authors) if meta.authors else '',
                'subject': meta.description or '',
                'keywords': ', '.join(meta.subjects[:10] + meta.genres[:5] + meta.tags[:5]),
                'creator': 'EbookMetadataPipeline', 'producer': meta.publisher or '',
            })
            doc.saveIncr()
            doc.close()
            return True
        except Exception as e:
            self.log.warning(f"PDF metadata write failed: {e}")
            return False


# =============================================================================
# FILE RENAMER
# =============================================================================

class FileRenamer:
    def __init__(self, logger, dry_run=False):
        self.log = logger
        self.dry_run = dry_run

    def _display_author(self, author):
        if ',' in author and author.count(',') == 1:
            parts = author.split(',', 1)
            last, first = parts[0].strip(), parts[1].strip().rstrip('.')
            if first and last and len(last.split()) <= 2 and len(first.split()) <= 3 and first[0].isupper() and last[0].isupper():
                return f"{first} {last}"
        return author

    def _build_name(self, meta, ext):
        title = sanitize_filename(meta.title, max_length=120)
        year = extract_year(meta.publication_date) or extract_year(meta.original_publication_date)
        ed = ""
        if meta.edition: ed = f" [{sanitize_filename(meta.edition, 30)}]"
        elif meta.revision: ed = f" [rev {sanitize_filename(meta.revision, 20)}]"
        if meta.authors:
            da = [self._display_author(a) for a in meta.authors]
            if len(da) == 1: author_str = sanitize_filename(da[0], 60)
            elif len(da) == 2: author_str = sanitize_filename(f"{da[0]} & {da[1]}", 80)
            else: author_str = sanitize_filename(f"{da[0]} et al.", 60)
        else:
            author_str = "Unknown Author"
        sp = ""
        if meta.series:
            ss = sanitize_filename(meta.series, 50)
            if meta.series_index is not None:
                idx = int(meta.series_index) if meta.series_index == int(meta.series_index) else meta.series_index
                sp = f" ({ss} #{idx})"
            else: sp = f" ({ss})"
        yp = f" ({year})" if year else ""
        return f"{title}{yp}{ed}{sp} - {author_str}{ext}"

    def compute_new_path(self, filepath, meta):
        if not meta.title: return None
        new_name = self._build_name(meta, Path(filepath).suffix)
        new_path = os.path.join(os.path.dirname(filepath), new_name)
        return new_path if os.path.abspath(new_path) != os.path.abspath(filepath) else filepath

    def rename(self, filepath, meta):
        if not meta.title: return None
        ext = Path(filepath).suffix
        directory = os.path.dirname(filepath)
        new_name = self._build_name(meta, ext)
        new_path = os.path.join(directory, new_name)
        if os.path.abspath(new_path) == os.path.abspath(filepath):
            return filepath
        counter = 1
        base = new_path
        while os.path.exists(new_path):
            stem = Path(base).stem
            new_path = os.path.join(directory, f"{stem} ({counter}){ext}")
            counter += 1
        if self.dry_run: return new_path
        try:
            os.rename(filepath, new_path)
            return new_path
        except OSError as e:
            self.log.error(f"Rename failed: {e}")
            return None


# =============================================================================
# REPORT GENERATOR
# =============================================================================

class ReportGenerator:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.results = []
        self._lock = threading.Lock()
        self._readability_stats = defaultdict(int)

    def add_result(self, filepath, meta, new_path, elapsed, status,
                   readability: Optional[ReadabilityResult] = None):
        with self._lock:
            entry = {
                'original_path': filepath, 'new_path': new_path,
                'title': meta.title, 'authors': meta.authors,
                'completeness': meta.completeness_score(),
                'sources_used': meta.sources_used, 'errors': meta.errors,
                'elapsed_seconds': round(elapsed, 2), 'status': status,
            }
            if readability:
                entry['readability'] = {
                    'status': readability.status_label,
                    'method': readability.method_used,
                    'text_length': readability.text_length,
                    'pages_read': readability.pages_read,
                    'total_pages': readability.total_pages,
                    'is_scanned': readability.is_scanned_pdf,
                    'is_encrypted': readability.is_encrypted,
                    'is_corrupted': readability.is_corrupted,
                    'failure_reasons': readability.failure_reasons,
                    'ai_fallback': readability.ai_fallback_used,
                }
                self._readability_stats[readability.status_label] += 1
            self.results.append(entry)

    def save(self):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(self.log_dir, f'report_{timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump({
                'generated_at': datetime.now().isoformat(),
                'total_processed': len(self.results),
                'successful': sum(1 for r in self.results if r['status'] == 'success'),
                'failed': sum(1 for r in self.results if r['status'] == 'error'),
                'skipped': sum(1 for r in self.results if r['status'] == 'skipped'),
                'avg_completeness': (sum(r['completeness'] for r in self.results) / len(self.results) if self.results else 0),
                'readability_summary': dict(self._readability_stats),
                'results': self.results,
            }, f, indent=2)
        G = '\033[32m'; R = '\033[31m'; Y = '\033[33m'; D = '\033[90m'
        C = '\033[36m'; B = '\033[1m'; X = '\033[0m'

        total = len(self.results)
        success = sum(1 for r in self.results if r['status'] == 'success')
        errors = sum(1 for r in self.results if r['status'] == 'error')
        skipped = sum(1 for r in self.results if r['status'] == 'skipped')

        print(f"\n{C}{'═'*60}{X}")
        print(f"{B}PROCESSING COMPLETE{X}")
        print(f"{C}{'═'*60}{X}")
        print(f"  {D}Total files:{X}      {B}{total}{X}")
        print(f"  {D}Successful:{X}       {G}{success}{X}")
        print(f"  {D}Failed:{X}           {R if errors else D}{errors}{X}")
        print(f"  {D}Skipped:{X}          {Y if skipped else D}{skipped}{X}")
        if self.results:
            avg = sum(r['completeness'] for r in self.results) / total
            avg_color = G if avg >= 0.8 else Y if avg >= 0.5 else R
            print(f"  {D}Avg completeness:{X} {avg_color}{avg:.1%}{X}")

        if self._readability_stats:
            print(f"\n  {D}Readability:{X}")
            status_colors = {
                'readable': G, 'ai-recovered': C,
                'encrypted': Y, 'corrupted': R, 'scanned-image-only': Y,
                'unreadable': R, 'no-text': R,
            }
            for status, count in sorted(self._readability_stats.items()):
                sc = status_colors.get(status, D)
                print(f"    {sc}{status:<22}{X} {count}")

        print(f"\n  {D}Report saved:{X}     {json_path}")
        print(f"{C}{'═'*60}{X}\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class EbookMetadataPipeline:
    RDF_DOWNLOAD_URL = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"
    DEFAULT_RDF_DIR = os.path.expanduser("~/gutenberg-rdf")

    def __init__(self, args):
        self.args = args
        self.log_dir = args.log_dir or os.path.join(args.ebook_dir, '.metadata_logs')
        self.log = setup_logging(self.log_dir, args.verbose)
        self.stats = logging.getLogger('ebook_pipeline.stats')

        # Dependency check
        checker = DependencyChecker(self.log, auto_install=not getattr(args, 'skip_dep_install', False))
        self.dep_status = checker.check_and_install()
        if not self.dep_status.all_critical_ok:
            self.log.error("Critical dependencies missing — cannot proceed.")
            self.log.error("Try: python ebook_metadata_pipeline.py --bootstrap-venv /path/to/ebooks")
            sys.exit(1)

        max_pages = getattr(args, 'max_pages', DEFAULT_MAX_PAGES)
        self.text_extractor = TextExtractor(self.log, self.dep_status, max_pages=max_pages)
        self.embedded_extractor = EmbeddedMetadataExtractor(self.log, self.dep_status)

        rdf_path = args.rdf_catalog or self.DEFAULT_RDF_DIR
        self.rdf_catalog = None
        if not args.skip_rdf:
            self.rdf_catalog = self._setup_rdf_catalog(rdf_path)

        self.api_lookup = PublicAPILookup(self.log, args.google_api_key)
        self.ai_extractor = AIMetadataExtractor(self.log, args.anthropic_api_key, self.dep_status)
        self.writer = MetadataWriter(self.log, self.dep_status)
        self.renamer = FileRenamer(self.log, args.dry_run)
        self.report = ReportGenerator(self.log_dir)

        if os.path.exists("/tmp"):
            path_hash = hashlib.md5(args.ebook_dir.encode()).hexdigest()
            cache_path = os.path.join("/tmp", f"ebook_metadata_{path_hash}.db")
        else:
            cache_path = os.path.join(args.ebook_dir, '.metadata_cache.db')
        self.cache = ProcessingCache(cache_path, self.log)
        self.force_reprocess = getattr(args, 'force', False)

    def _setup_rdf_catalog(self, rdf_dir):
        for d in [os.path.join(rdf_dir, 'cache', 'epub'), os.path.join(rdf_dir, 'epub'), rdf_dir]:
            if os.path.isdir(d):
                sample = [x for x in os.listdir(d) if x.isdigit()][:1]
                if sample:
                    self.log.info(f"RDF catalog found at {d}")
                    return GutenbergRDFCatalog(rdf_dir, self.log)
        tarball = os.path.join(rdf_dir, 'rdf-files.tar.bz2')
        if os.path.exists(tarball):
            return self._extract_rdf(rdf_dir, tarball)
        self.log.warning(f"Gutenberg RDF catalog not found at {rdf_dir}")
        if self.args.auto_download_rdf or self._confirm_download():
            return self._download_and_extract_rdf(rdf_dir)
        return None

    def _confirm_download(self):
        if not sys.stdin.isatty(): return False
        try:
            resp = input("\nGutenberg RDF catalog not found. Download? (~300MB) [Y/n]: ").strip().lower()
            return resp in ('', 'y', 'yes')
        except (EOFError, KeyboardInterrupt):
            return False

    def _download_and_extract_rdf(self, rdf_dir):
        import urllib.request
        os.makedirs(rdf_dir, exist_ok=True)
        tarball = os.path.join(rdf_dir, 'rdf-files.tar.bz2')
        self.log.info("Downloading Gutenberg RDF catalog (~300MB)...")
        try:
            def hook(bn, bs, ts):
                if ts > 0:
                    print(f"\r  Downloading: {bn*bs/(1024*1024):.0f}/{ts/(1024*1024):.0f} MB ({min(100,bn*bs*100//ts)}%)", end='', flush=True)
            urllib.request.urlretrieve(self.RDF_DOWNLOAD_URL, tarball, hook)
            print()
            return self._extract_rdf(rdf_dir, tarball)
        except Exception as e:
            self.log.error(f"RDF download failed: {e}")
            return None

    def _extract_rdf(self, rdf_dir, tarball):
        self.log.info("Extracting RDF catalog...")
        rc, _, err = run_command(['tar', 'xjf', tarball, '-C', rdf_dir], timeout=600)
        if rc == 0:
            self.log.info("RDF catalog extracted successfully")
            return GutenbergRDFCatalog(rdf_dir, self.log)
        self.log.error(f"Extraction failed: {err}")
        return None

    def discover_files(self):
        self.log.info(f"\033[36mScanning\033[0m {self.args.ebook_dir}...")
        files = []
        for root, dirs, filenames in os.walk(self.args.ebook_dir):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            for fname in filenames:
                if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append(os.path.join(root, fname))
        self.log.info(f"\033[32mFound\033[0m \033[1m{len(files)}\033[0m ebook files")
        return sorted(files)


    def process_file(self, filepath):
        start_time = time.time()
        basename = os.path.basename(filepath)
        ext = Path(filepath).suffix.lower()
        file_size = os.path.getsize(filepath)
        file_size_mb = file_size / (1024 * 1024)
        self.log.info(f"")
        self.log.info(f"\033[36m┌───\033[0m \033[1m📖 {basename}\033[0m")
        size_color = '\033[33m' if file_size_mb > 20 else '\033[90m'
        self.log.info(f"\033[36m│\033[0m  {size_color}{ext}  •  {file_size/1024:.0f} KB\033[0m")

        # Skip very large files if --max-file-size is set
        max_mb = getattr(self.args, 'max_file_size', 0)
        if max_mb and file_size_mb > max_mb:
            self.log.info(f"\033[36m│\033[0m  \033[33m⊘ Skipped: {file_size_mb:.0f}MB exceeds --max-file-size {max_mb}MB\033[0m")
            self.log.info(f"\033[36m└{'─'*60}\033[0m")
            meta = BookMetadata(source_file=filepath, source_format=ext)
            meta.title = parse_filename_metadata(filepath).title
            return meta, filepath, 'skipped', ReadabilityResult()

        meta = BookMetadata(source_file=filepath, source_format=ext)
        meta.file_hash = file_sha256(filepath) if file_size < 50_000_000 else None
        original_meta = BookMetadata()
        readability = ReadabilityResult()

        # Stage 1: Gutenberg RDF
        pg_id = detect_gutenberg_id(filepath)
        if pg_id and self.rdf_catalog:
            rdf_meta = self.rdf_catalog.lookup(pg_id)
            if rdf_meta:
                meta.merge(rdf_meta, "gutenberg-rdf")

        # Stage 2: Embedded metadata
        if self.args.verbose:
            self.log.info(f"\033[36m│\033[0m  \033[90m↻ Reading embedded metadata...\033[0m")
        # Scale calibre timeout for large files
        calibre_read_timeout = 30 if file_size_mb < 20 else 60 if file_size_mb < 100 else 90
        try:
            embedded = self.embedded_extractor.extract(filepath, timeout=calibre_read_timeout)
            for fld in embedded.__dataclass_fields__:
                val = getattr(embedded, fld)
                if val and val != [] and val != 0 and val != 0.0:
                    try:
                        if isinstance(val, list): setattr(original_meta, fld, list(val))
                        else: setattr(original_meta, fld, val)
                    except Exception: pass
            meta.merge(embedded, "embedded")
        except Exception as e:
            meta.errors.append(f"embedded: {e}")

        # Normalize original_meta so that deduplication/cleanup doesn't appear as a "change"
        if original_meta.authors:
            original_meta.authors = deduplicate_authors(
                [clean_author_string(a) for a in original_meta.authors if a])
        if original_meta.tags:
            original_meta.tags = list(dict.fromkeys(
                t.strip() for t in original_meta.tags if t and t.strip()))
        if original_meta.title:
            original_meta.title = clean_title(original_meta.title)

        # Stage 2.5: Filename parsing
        fn_meta = parse_filename_metadata(filepath)
        if meta.authors:
            expanded = []
            for a in meta.authors:
                if ';' in a: expanded.extend([s.strip() for s in a.split(';') if s.strip()])
                else: expanded.append(a)
            meta.authors = [clean_author_string(a) for a in expanded if a]

        # Garbage detection
        if fn_meta.title and meta.title:
            t = (meta.title or '').lower().strip()
            garbage_indicators = (
                bool(re.search(r'\.\w{2,4}$', meta.title)) or
                t in ('frontmatter', 'prelims', 'preface', 'contents', 'table of contents',
                       'copyright', 'copyright page', 'title page', 'cover', 'half title') or
                bool(re.match(r'^module\s', t)) or bool(re.match(r'^chapter\s*\d', t)) or
                'microsoft word' in t or 'untitled' in t or '.indd' in t or 'adobe' in t
            )
            if garbage_indicators:
                meta.title = fn_meta.title
                if fn_meta.authors: meta.authors = fn_meta.authors
                if fn_meta.publisher: meta.publisher = fn_meta.publisher
                if fn_meta.publication_date: meta.publication_date = fn_meta.publication_date
                meta.processing_notes.append("corrected-from-filename")

        if not meta.title and fn_meta.title: meta.title = fn_meta.title
        if not meta.authors and fn_meta.authors: meta.authors = fn_meta.authors
        if not meta.publisher and fn_meta.publisher: meta.publisher = fn_meta.publisher
        if not meta.publication_date and fn_meta.publication_date: meta.publication_date = fn_meta.publication_date
        if not meta.edition and fn_meta.edition: meta.edition = fn_meta.edition

        if meta.authors:
            meta.authors = [clean_author_string(a) for a in meta.authors if a]
            meta.authors = deduplicate_authors(meta.authors)

        # Stage 2.75: Text extraction & readability check
        if self.args.verbose:
            self.log.info(f"\033[36m│\033[0m  \033[90m↻ Extracting text...\033[0m")
        text, readability = self.text_extractor.extract(filepath)
        self.log.info(f"\033[36m│\033[0m  {readability.status_icon}"
                      f"  \033[90m({readability.pages_read or 0} pages"
                      f"{f'/{readability.total_pages}' if readability.total_pages else ''}"
                      f", {readability.text_length} chars"
                      f", method: {readability.method_used or 'none'})\033[0m")

        if readability.failure_reasons:
            for reason in readability.failure_reasons[:3]:  # Show first 3 reasons
                self.log.info(f"\033[36m│\033[0m    \033[90m└ {reason}\033[0m")

        # Stage 3: Public APIs
        needs_api = False
        if not self.args.skip_api:
            missing = []
            if not meta.subjects and not meta.genres: missing.append('subjects')
            if not meta.language: missing.append('language')
            if not meta.description: missing.append('description')
            if not meta.page_count: missing.append('pages')
            if not meta.cover_url: missing.append('cover')
            if not meta.isbn_13 and not meta.isbn_10: missing.append('ISBN')
            if not meta.publisher: missing.append('publisher')
            if not meta.lcc and not meta.ddc: missing.append('classification')
            needs_api = len(missing) >= 2

        if needs_api:
            self.log.info(f"\033[36m│\033[0m  \033[36m↻ Searching APIs...\033[0m")
            try:
                api_meta = self.api_lookup.search(
                    title=meta.title,
                    author=meta.authors[0] if meta.authors else None,
                    isbn=meta.isbn_13 or meta.isbn_10,
                    current_completeness=meta.completeness_score())
                meta.merge(api_meta, "public-api")
                for src in api_meta.sources_used:
                    if src not in meta.sources_used: meta.sources_used.append(src)
                for api_name, status, detail in getattr(api_meta, '_api_diagnostics', []):
                    if status == 'success':
                        self.log.info(f"\033[36m│\033[0m    \033[32m✓ {api_name}: {detail}\033[0m")
                    elif status == 'skipped':
                        self.log.info(f"\033[36m│\033[0m    \033[90m⊘ {api_name}: skipped ({detail})\033[0m")
                    elif status == 'no_match':
                        self.log.info(f"\033[36m│\033[0m    \033[33m✗ {api_name}: {detail}\033[0m")
                    else:
                        self.log.info(f"\033[36m│\033[0m    \033[31m✗ {api_name}: {detail}\033[0m")
            except Exception as e:
                meta.errors.append(f"api: {e}")

        # Stage 4: AI Analysis
        ai_used = False
        if not self.args.skip_ai:
            if text and len(text.strip()) > 100 and meta.completeness_score() < self.args.ai_threshold:
                # Normal AI analysis with extracted text
                try:
                    ai_meta = self.ai_extractor.analyze(text, basename, meta)
                    if ai_meta:
                        meta.merge(ai_meta, "ai-claude")
                        ai_used = True
                except Exception as e:
                    meta.errors.append(f"ai: {e}")

            elif not readability.readable and self.ai_extractor.available():
                # AI FALLBACK: file was unreadable, try filename-based identification
                self.log.info(f"\033[36m│\033[0m  \033[35m🤖 AI fallback (unreadable file)...\033[0m")
                readability.ai_fallback_used = True
                try:
                    ai_meta = self.ai_extractor.analyze_from_filename(
                        basename, os.path.getsize(filepath), meta)
                    if ai_meta and ai_meta.title:
                        meta.merge(ai_meta, "ai-claude-filename")
                        readability.ai_fallback_success = True
                        ai_used = True
                        self.log.info(f"\033[36m│\033[0m    \033[35m✓ AI identified: {ai_meta.title}\033[0m")
                    else:
                        self.log.info(f"\033[36m│\033[0m    \033[33m✗ AI could not identify book\033[0m")
                except Exception as e:
                    meta.errors.append(f"ai-fallback: {e}")
                    self.log.info(f"\033[36m│\033[0m    \033[31m✗ AI fallback failed: {e}\033[0m")

        # Stage 4.5: Genre/Subject Inference
        if not meta.genres or not meta.subjects:
            ig, isub = infer_genres_subjects(meta)
            if ig and not meta.genres:
                meta.genres = ig
                if "inferred" not in meta.sources_used: meta.sources_used.append("inferred")
            if isub and not meta.subjects:
                meta.subjects = isub
                if "inferred" not in meta.sources_used: meta.sources_used.append("inferred")

        # Final cleanup
        if meta.authors:
            meta.authors = deduplicate_authors([clean_author_string(a) for a in meta.authors if a])
        if not meta.title:
            fn = parse_filename_metadata(filepath)
            if fn.title: meta.title = fn.title
            if not meta.authors and fn.authors: meta.authors = fn.authors
        if meta.title: meta.title = clean_title(meta.title)

        # Compute rename
        new_path = filepath
        new_basename = None
        if not self.args.skip_rename:
            renamed = self.renamer.compute_new_path(filepath, meta)
            if renamed and renamed != filepath:
                new_basename = os.path.basename(renamed)
        if new_basename:
            self.log.info(f"\033[36m│\033[0m  \033[36m→ {new_basename}\033[0m")

        # Display: FOUND IN FILE
        self.log.info(f"\033[36m│\033[0m")
        self.log.info(f"\033[36m│\033[0m  \033[1;90m─── FOUND IN FILE ───\033[0m")
        found = []
        if original_meta.title: found.append(f"\033[36m│\033[0m  \033[90m  Title:       {original_meta.title}\033[0m")
        if original_meta.authors: found.append(f"\033[36m│\033[0m  \033[90m  Authors:     {', '.join(original_meta.authors)}\033[0m")
        if original_meta.publisher: found.append(f"\033[36m│\033[0m  \033[90m  Publisher:   {original_meta.publisher}\033[0m")
        if original_meta.publication_date: found.append(f"\033[36m│\033[0m  \033[90m  Date:        {original_meta.publication_date}\033[0m")
        if original_meta.isbn_13 or original_meta.isbn_10: found.append(f"\033[36m│\033[0m  \033[90m  ISBN:        {original_meta.isbn_13 or original_meta.isbn_10}\033[0m")
        if original_meta.tags: found.append(f"\033[36m│\033[0m  \033[90m  Tags:        {', '.join(original_meta.tags[:5])}\033[0m")
        if original_meta.description:
            desc = re.sub(r'<[^>]+>', '', original_meta.description)[:80].strip()
            found.append(f"\033[36m│\033[0m  \033[90m  Description: {desc}...\033[0m")
        if found:
            for line in found: self.log.info(line)
        else:
            self.log.info(f"\033[36m│\033[0m  \033[90m  (no embedded metadata)\033[0m")

        # Display: CHANGES
        # Three categories:
        #   changes[]    — New data that CAN be written to this file format → triggers write
        #   unstorable[] — New data from APIs but format can't hold it → cache only
        #   derived[]    — Inferred from existing embedded data → display only
        writable = FORMAT_WRITABLE_FIELDS.get(ext, set())
        changes = []
        unstorable = []
        derived = []

        # Map each metadata field to its FORMAT_WRITABLE_FIELDS key (None = never writable)
        FIELD_TO_FORMAT_KEY = {
            'title': 'title', 'subtitle': None, 'authors': 'authors',
            'publisher': 'publisher', 'date': 'date', 'orig_date': None,
            'edition': None, 'language': 'language',
            'isbn_13': 'isbn', 'isbn_10': 'isbn',
            'subjects': None, 'genres': 'genres',
            'lcc': None, 'ddc': None, 'pages': None, 'cover': 'cover_embed',
            'description': 'description',
        }

        def chk(label, old, new, is_list=False, is_derived=False, field_key=None):
            if new is None or new == [] or new == 0: return
            if is_list and isinstance(new, list):
                display = ', '.join(str(x) for x in new[:8])
                if len(new) > 8: display += f" (+{len(new)-8} more)"
            else:
                display = str(new)
                if len(display) > 120: display = re.sub(r'<[^>]+>', '', display)[:120] + "..."
            old_empty = old is None or old == [] or old == 0
            same = (old == new) or (is_list and not old_empty and set(old or []) == set(new or []))
            if same: return

            # Decide which bucket
            fmt_key = FIELD_TO_FORMAT_KEY.get(field_key)
            can_write = fmt_key is not None and fmt_key in writable

            if is_derived:
                target = derived
            elif not can_write and old_empty:
                target = unstorable
            else:
                target = changes

            if old_empty:
                if target is changes:
                    target.append(f"\033[36m│\033[0m  \033[32m+ {label:<14}{display}\033[0m")
                elif target is unstorable:
                    target.append(f"\033[36m│\033[0m  \033[36m◆ {label:<14}{display}\033[0m")
                else:
                    target.append(f"\033[36m│\033[0m  \033[90m~ {label:<14}{display}\033[0m")
            else:
                if target is changes:
                    target.append(f"\033[36m│\033[0m  \033[33m✎ {label:<14}{display}\033[0m")
                else:
                    target.append(f"\033[36m│\033[0m  \033[90m✎ {label:<14}{display}\033[0m")

        # Determine which fields are truly new vs derived from existing embedded data
        genres_from_api = any(s in meta.sources_used for s in ('openlibrary', 'google-books', 'ai-claude'))
        subjects_from_api = genres_from_api

        orig_tags_lower = {t.lower().strip() for t in (original_meta.tags or [])}
        subjects_are_from_tags = (
            not subjects_from_api
            and not original_meta.subjects
            and meta.subjects
            and all(s.lower().strip() in orig_tags_lower for s in meta.subjects)
        )
        genres_are_inferred = (not genres_from_api and not original_meta.genres)

        chk("Title", original_meta.title, meta.title, field_key='title')
        chk("Subtitle", original_meta.subtitle, meta.subtitle, field_key='subtitle')
        chk("Authors", original_meta.authors, meta.authors, True, field_key='authors')
        chk("Publisher", original_meta.publisher, meta.publisher, field_key='publisher')
        chk("Date", original_meta.publication_date, meta.publication_date, field_key='date')
        chk("Orig. Date", original_meta.original_publication_date, meta.original_publication_date, field_key='orig_date')
        chk("Edition", original_meta.edition, meta.edition, field_key='edition')
        chk("Language", original_meta.language, meta.language, field_key='language')
        chk("ISBN-13", original_meta.isbn_13, meta.isbn_13, field_key='isbn_13')
        chk("ISBN-10", original_meta.isbn_10, meta.isbn_10, field_key='isbn_10')
        chk("Subjects", original_meta.subjects, meta.subjects, True,
            is_derived=subjects_are_from_tags, field_key='subjects')
        chk("Genres", original_meta.genres, meta.genres, True,
            is_derived=genres_are_inferred, field_key='genres')
        chk("LCC", original_meta.lcc, meta.lcc, field_key='lcc')
        chk("DDC", original_meta.ddc, meta.ddc, field_key='ddc')
        chk("Pages", original_meta.page_count, meta.page_count, field_key='pages')
        if meta.cover_url and not original_meta.cover_url:
            try:
                from urllib.parse import urlparse
                domain = urlparse(meta.cover_url).netloc
            except Exception: domain = meta.cover_url[:40]
            chk("Cover", None, f"✓ {domain}", field_key='cover')

        has_changes = bool(changes)  # Only writable changes trigger file writes

        fmt_name = ext.lstrip('.').upper()
        if changes or unstorable or derived:
            self.log.info(f"\033[36m│\033[0m")
            if changes:
                self.log.info(f"\033[36m│\033[0m  \033[1;32m─── CHANGES ───\033[0m")
                for line in changes: self.log.info(line)
            if unstorable:
                self.log.info(f"\033[36m│\033[0m  \033[90m─── API FOUND (not storable in {fmt_name}) ───\033[0m")
                for line in unstorable: self.log.info(line)
            if derived:
                self.log.info(f"\033[36m│\033[0m  \033[90m─── DERIVED (from existing tags) ───\033[0m")
                for line in derived: self.log.info(line)
        else:
            self.log.info(f"\033[36m│\033[0m")
            self.log.info(f"\033[36m│\033[0m  \033[32m✓ No changes needed\033[0m")

        # Stage 5: Write metadata (skip if file is corrupted/encrypted or no changes)
        if not self.args.dry_run and not self.args.skip_write and has_changes:
            if readability.is_corrupted:
                self.log.info(f"\033[36m│\033[0m  \033[90m⊘ Skipping metadata write (file corrupted)\033[0m")
            elif readability.is_encrypted:
                self.log.info(f"\033[36m│\033[0m  \033[90m⊘ Skipping metadata write (file encrypted)\033[0m")
            else:
                try:
                    write_timeout = 30 if file_size_mb < 20 else 60 if file_size_mb < 100 else 120
                    self.writer.write(filepath, meta, timeout=write_timeout)
                except Exception as e: meta.errors.append(f"write: {e}")

        # Stage 6: Rename
        if not self.args.skip_rename and new_basename:
            renamed = self.renamer.rename(filepath, meta)
            if renamed and renamed != filepath: new_path = renamed

        # Footer
        elapsed = time.time() - start_time
        meta.confidence_score = meta.completeness_score()
        status = 'success' if meta.title else 'error'
        score = meta.completeness_score()
        sc = '\033[32m' if score >= 0.7 else ('\033[33m' if score >= 0.5 else '\033[31m')
        icon = '✅' if status == 'success' else '❌'

        sp = []
        sources = meta.sources_used
        if any('gutenberg' in s for s in sources): sp.append('\033[32m☑\033[90m Catalog')
        if 'embedded' in sources: sp.append('\033[32m☑\033[90m Embedded')
        if any('openlibrary' in s for s in sources): sp.append('\033[32m☑\033[90m OpenLibrary')
        if any('google' in s for s in sources): sp.append('\033[32m☑\033[90m Google')
        if any('ai' in s for s in sources): sp.append('\033[35m☑\033[90m AI')
        sd = '  '.join(sp) if sp else '\033[90mnone'

        # Determine what's missing vs what the format can't store
        writable = FORMAT_WRITABLE_FIELDS.get(ext, set())
        missing_writable = []    # Could be added to this format
        missing_format = []      # Format can't store this

        if not meta.title:
            (missing_writable if 'title' in writable else missing_format).append('title')
        if not meta.authors:
            (missing_writable if 'authors' in writable else missing_format).append('authors')
        if not meta.publisher:
            (missing_writable if 'publisher' in writable else missing_format).append('publisher')
        if not meta.isbn_13 and not meta.isbn_10:
            (missing_writable if 'isbn' in writable else missing_format).append('ISBN')
        if not meta.language:
            (missing_writable if 'language' in writable else missing_format).append('language')
        if not meta.description:
            (missing_writable if 'description' in writable else missing_format).append('description')
        if not meta.page_count:
            missing_format.append('pages')        # No ebook format stores page count
        if not meta.lcc and not meta.ddc:
            missing_format.append('classification')  # Library catalog concept
        if not meta.cover_url:
            missing_format.append('cover')         # Formats embed covers, not URLs

        self.log.info(f"\033[36m│\033[0m")
        self.log.info(f"\033[36m│\033[0m  {icon} {sc}{score:.0%} complete\033[0m  \033[90m•  {elapsed:.1f}s\033[0m  •  {sd}\033[0m")
        if missing_writable:
            self.log.info(f"\033[36m│\033[0m  \033[33mMissing: {', '.join(missing_writable)}\033[0m")
        if missing_format:
            fmt_name = ext.lstrip('.').upper()
            self.log.info(f"\033[36m│\033[0m  \033[90mNot in {fmt_name}: {', '.join(missing_format)}\033[0m")
        self.log.info(f"\033[36m└{'─'*60}\033[0m")

        self.stats.info(json.dumps({
            'file': filepath, 'status': status, 'title': meta.title,
            'completeness': score, 'sources': meta.sources_used,
            'elapsed': round(elapsed, 2),
            'readability': readability.status_label,
        }))
        return meta, new_path, status, readability


    def run(self):
        # ANSI colors
        C = '\033[36m'    # cyan
        G = '\033[32m'    # green
        Y = '\033[33m'    # yellow
        R = '\033[31m'    # red
        D = '\033[90m'    # dim/gray
        B = '\033[1m'     # bold
        W = '\033[97m'    # bright white
        X = '\033[0m'     # reset

        self.log.info("")
        self.log.info(f"{C}╔{'═'*60}╗{X}")
        self.log.info(f"{C}║{B}{W}{'  EBOOK METADATA PIPELINE v2.0'.center(60)}{X}{C}║{X}")
        self.log.info(f"{C}╚{'═'*60}╝{X}")
        self.log.info(f"  {D}Target:{X}       {B}{self.args.ebook_dir}{X}")
        self.log.info(f"  {D}RDF Catalog:{X}  {G + '✓ Loaded' if self.rdf_catalog else R + '✗ Not available'}{X}")
        self.log.info(f"  {D}Dry run:{X}      {Y + 'True' if self.args.dry_run else D + 'False'}{X}")
        self.log.info(f"  {D}Skip API:{X}     {Y + 'True' if self.args.skip_api else D + 'False'}{X}")
        kc = len(self.args.google_api_key) if self.args.google_api_key else 0
        self.log.info(f"  {D}Google API:{X}   {G + '✓ ' + str(kc) + ' key(s)' if kc else R + '✗ None'}{X}")
        self.log.info(f"  {D}Skip AI:{X}      {Y + 'True' if self.args.skip_ai else D + 'False'}{X}")
        ai_ok = self.ai_extractor.available()
        self.log.info(f"  {D}AI available:{X} {G + '✓' if ai_ok else R + '✗ (no key or module)'}{X}")
        self.log.info(f"  {D}Skip write:{X}   {Y + 'True' if self.args.skip_write else D + 'False'}{X}")
        self.log.info(f"  {D}Skip rename:{X}  {Y + 'True' if self.args.skip_rename else D + 'False'}{X}")
        self.log.info(f"  {D}Max pages:{X}    {getattr(self.args, 'max_pages', DEFAULT_MAX_PAGES)}")
        threads = getattr(self.args, 'threads', 1) or 1
        if threads > 1: self.log.info(f"  {D}Threads:{X}      {threads}")
        self.log.info(f"  {D}Log dir:{X}      {self.log_dir}")
        self.log.info(f"  {D}Cache DB:{X}     {D}{self.cache.cache_path}{X}")
        cs = self.cache.get_stats()
        if cs['total'] > 0:
            self.log.info(f"  {D}Cache stats:{X}  {C}{cs['total']}{X} books tracked, avg {G}{cs['avg_completeness']*100:.0f}%{X} complete, {Y if cs.get('unreadable', 0) else D}{cs.get('unreadable', 0)}{X} unreadable")

        fmt_support = self.dep_status.formats_supported()
        degraded = [ext for ext, ok in fmt_support.items() if not ok]
        if degraded:
            self.log.warning(f"  {Y}⚠ Degraded:{X}   {', '.join(degraded)} {D}(missing modules){X}")
        else:
            self.log.info(f"  {D}Deps:{X}         {G}✓ All modules available{X}")
        for warn in self.dep_status.warnings:
            self.log.warning(f"  {Y}⚠ {warn}{X}")
        self.log.info("")

        files = self.discover_files()
        if not files:
            self.log.warning("No ebook files found!")
            return

        # Dedup
        seen_paths, seen_keys, unique, dup = set(), set(), [], 0
        for f in files:
            ap, rp = os.path.abspath(f), os.path.realpath(f)
            if ap in seen_paths or rp in seen_paths:
                dup += 1; continue
            try:
                fk = (os.path.basename(f), os.path.getsize(f))
                if fk in seen_keys: dup += 1; continue
                seen_keys.add(fk)
            except OSError: pass
            seen_paths.update([ap, rp])
            unique.append(f)
        if dup: self.log.info(f"Skipped {dup} duplicate(s)")
        files = unique

        limit = self.args.limit or len(files)
        files = files[:limit]

        # Cache check
        if not self.force_reprocess:
            to_process, cached = [], 0
            for f in files:
                result = self.cache.is_processed(f)
                if result:
                    cached += 1
                    if self.args.verbose:
                        self.log.debug(f"  \033[90mCache hit: {os.path.basename(f)} ({result['status']}, {result['completeness']:.0%})\033[0m")
                else:
                    to_process.append(f)
            if cached: self.log.info(f"  \033[90mCache:\033[0m \033[32m{cached}\033[0m already processed, \033[33m{len(to_process)}\033[0m remaining")
            elif self.args.verbose:
                # Show why zero cache hits — check if DB has entries vs just stale
                cs = self.cache.get_stats()
                if cs['total'] > 0:
                    self.log.info(f"  \033[33mCache:\033[0m {cs['total']} entries in DB but 0 matched current files")
                    # Show first cached entry vs first scanned file for comparison
                    sample_cached = self.cache.list_entries(3)
                    if sample_cached:
                        self.log.info(f"  \033[90m  DB sample:   {sample_cached[0]['filepath'][:100]}\033[0m")
                    if to_process:
                        norm = os.path.normpath(os.path.abspath(to_process[0]))
                        self.log.info(f"  \033[90m  Scan sample: {norm[:100]}\033[0m")
            files = to_process
        else:
            self.log.info("  \033[33mCache:\033[0m --force enabled, reprocessing all")

        if not files:
            self.log.info("\033[32mAll files already processed!\033[0m Use --force to reprocess.")
            cs = self.cache.get_stats()
            self.log.info(f"  \033[90mCache:\033[0m {cs['total']} books, avg \033[32m{cs['avg_completeness']*100:.0f}%\033[0m complete, {cs.get('unreadable', 0)} unreadable")
            return

        self.log.info(f"\033[36mProcessing\033[0m \033[1m{len(files)}\033[0m files...\n")

        if threads <= 1:
            self._run_sequential(files)
        else:
            self._run_threaded(files, threads)

        self.report.save()
        cs = self.cache.get_stats()
        self.log.info(f"\n  \033[90mCache:\033[0m \033[36m{cs['total']}\033[0m books tracked, avg \033[32m{cs['avg_completeness']*100:.0f}%\033[0m complete, \033[33m{cs.get('unreadable', 0)}\033[0m unreadable")

    def _run_sequential(self, files):
        for i, filepath in enumerate(files, 1):
            try:
                self.log.info(f"\n\033[36m{'━'*62}\033[0m")
                self.log.info(f"  [{i}/{len(files)}] ({i/len(files)*100:.0f}%)")
                meta, new_path, status, readability = self.process_file(filepath)
                self.report.add_result(filepath, meta, new_path, 0, status, readability)
                # Cache under the final path (after rename) so next scan matches
                cache_path = new_path if new_path and new_path != filepath else filepath
                self.cache.mark_processed(cache_path, meta, status, readability)
                # If file was renamed, also remove stale cache entry for old path
                if new_path and new_path != filepath:
                    self.cache.remove_entry(filepath)
            except KeyboardInterrupt:
                self.log.warning("\nInterrupted!")
                break
            except Exception as e:
                self.log.error(f"FATAL: {filepath}: {e}")
                em = BookMetadata(source_file=filepath)
                em.errors.append(str(e))
                self.report.add_result(filepath, em, None, 0, 'error')
                self.cache.mark_processed(filepath, em, 'error')

    def _run_threaded(self, files, max_workers):
        print_lock = threading.Lock()
        console_handler = None
        for h in self.log.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                console_handler = h
                break

        def process_one(index, filepath):
            buf = BufferedLogHandler(thread_id=threading.get_ident())
            self.log.addHandler(buf)
            try:
                self.log.info(f"\n\033[36m{'━'*62}\033[0m")
                self.log.info(f"  [{index}/{len(files)}] ({index/len(files)*100:.0f}%)")
                meta, new_path, status, readability = self.process_file(filepath)
                return (filepath, meta, new_path, status, readability, buf.get_output(), None)
            except Exception as e:
                em = BookMetadata(source_file=filepath)
                em.errors.append(str(e))
                return (filepath, em, None, 'error', ReadabilityResult(), buf.get_output(), e)
            finally:
                self.log.removeHandler(buf)

        if console_handler: self.log.removeHandler(console_handler)
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_one, i, f): (i, f) for i, f in enumerate(files, 1)}
                for future in as_completed(futures):
                    try:
                        fp, meta, np, status, readability, output, error = future.result()
                        with print_lock:
                            print(output)
                            sys.stdout.flush()
                        self.report.add_result(fp, meta, np, 0, status, readability)
                        cache_path = np if np and np != fp else fp
                        self.cache.mark_processed(cache_path, meta, status, readability)
                        if np and np != fp:
                            self.cache.remove_entry(fp)
                    except KeyboardInterrupt:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as e:
                        idx, fp = futures[future]
                        with print_lock:
                            print(f"\033[31mFATAL: {fp}: {e}\033[0m")
                        em = BookMetadata(source_file=fp)
                        em.errors.append(str(e))
                        self.report.add_result(fp, em, None, 0, 'error')
                        self.cache.mark_processed(fp, em, 'error')
        finally:
            if console_handler: self.log.addHandler(console_handler)


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ebook metadata extraction, enrichment, and renaming pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Setup Examples:
  # First run — auto-create venv and install everything:
  python ebook_metadata_pipeline.py --bootstrap-venv /data/ebooks

  # Subsequent runs — auto-uses venv if it exists:
  python ebook_metadata_pipeline.py /data/ebooks

  # Manual venv setup:
  python -m venv ~/.venvs/ebook-pipeline
  source ~/.venvs/ebook-pipeline/bin/activate
  pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4
  python ebook_metadata_pipeline.py /data/ebooks

Usage Examples:
  python ebook_metadata_pipeline.py /data/ebooks --rdf-catalog /data/gutenberg-rdf
  python ebook_metadata_pipeline.py /data/ebooks --dry-run --verbose
  python ebook_metadata_pipeline.py /data/ebooks --skip-api --skip-ai
  python ebook_metadata_pipeline.py /data/ebooks --limit 10 --verbose --max-pages 20
        """)

    parser.add_argument('ebook_dir', help='Directory containing ebook files')
    parser.add_argument('--rdf-catalog', help='Path to Gutenberg RDF catalog directory')
    parser.add_argument('--anthropic-api-key', help='Anthropic API key (or set ANTHROPIC_API_KEY)')
    parser.add_argument('--google-api-key', action='append', default=None,
                        help='Google Books API key (repeat for multiple)')
    parser.add_argument('--log-dir', help='Directory for logs')
    parser.add_argument('--limit', type=int, help='Process only first N files')
    parser.add_argument('--max-pages', type=int, default=DEFAULT_MAX_PAGES,
                        help=f'Max pages to read per file for text extraction (default: {DEFAULT_MAX_PAGES})')
    parser.add_argument('--max-file-size', type=int, default=0,
                        help='Skip files larger than N megabytes (0 = no limit, default: 0)')
    parser.add_argument('--dry-run', action='store_true', help='Preview without modifying')
    parser.add_argument('--skip-api', action='store_true', help='Skip public API lookups')
    parser.add_argument('--skip-ai', action='store_true', help='Skip AI text analysis')
    parser.add_argument('--skip-rdf', action='store_true', help='Skip Gutenberg RDF catalog')
    parser.add_argument('--skip-write', action='store_true', help='Skip writing metadata')
    parser.add_argument('--skip-rename', action='store_true', help='Skip file renaming')
    parser.add_argument('--skip-dep-install', action='store_true',
                        help='Do not auto-install missing Python packages')
    parser.add_argument('--ai-threshold', type=float, default=0.4,
                        help='Completeness threshold below which AI is used (default: 0.4)')
    parser.add_argument('--threads', type=int, default=1, help='Parallel processing threads')
    parser.add_argument('--force', action='store_true', help='Reprocess all files ignoring cache')
    parser.add_argument('--show-cache', action='store_true',
                        help='Show cache statistics and exit without processing')
    parser.add_argument('--show-problems', action='store_true',
                        help='Show unreadable, low-completeness, and error files from cache')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear the processing cache and exit')
    parser.add_argument('--auto-download-rdf', action='store_true',
                        help='Auto-download Gutenberg RDF catalog without prompt')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # Venv options
    parser.add_argument('--bootstrap-venv', action='store_true',
                        help='Create a virtual environment and install all dependencies, then run')
    parser.add_argument('--venv-dir', default=DEFAULT_VENV_DIR,
                        help=f'Virtual environment directory (default: {DEFAULT_VENV_DIR})')
    parser.add_argument('--recreate-venv', action='store_true',
                        help='Force recreate the virtual environment')

    args = parser.parse_args()

    # Handle venv bootstrap
    if args.bootstrap_venv or args.recreate_venv:
        venv_python = bootstrap_venv(args.venv_dir, force=args.recreate_venv)
        # Re-launch this script inside the venv with bootstrap flag removed
        new_args = [a for a in sys.argv if a not in ('--bootstrap-venv', '--recreate-venv')]
        os.execv(venv_python, [venv_python] + new_args)

    # Auto-detect and use existing venv
    if os.path.exists(os.path.join(args.venv_dir, 'bin', 'python')):
        relaunch_in_venv(args.venv_dir)

    if not os.path.isdir(args.ebook_dir):
        print(f"Error: {args.ebook_dir} is not a directory")
        sys.exit(1)

    # Handle cache-only operations early (no dependency check needed)
    if args.show_cache or args.clear_cache or args.show_problems:
        path_hash = hashlib.md5(args.ebook_dir.encode()).hexdigest()
        cache_path = os.path.join("/tmp", f"ebook_metadata_{path_hash}.db") if os.path.exists("/tmp") \
            else os.path.join(args.ebook_dir, '.metadata_cache.db')
        logger = logging.getLogger('ebook_pipeline')
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO)
        if not os.path.exists(cache_path):
            print(f"No cache found at: {cache_path}")
            sys.exit(0)
        cache = ProcessingCache(cache_path, logger)

        if args.show_problems:
            G = '\033[32m'; R = '\033[31m'; Y = '\033[33m'; D = '\033[90m'
            C = '\033[36m'; B = '\033[1m'; X = '\033[0m'
            problems = cache.list_problems()

            print(f"\n{C}{'═'*70}{X}")
            print(f"{B}  PROBLEM REPORT{X}")
            print(f"{C}{'═'*70}{X}")

            # Unreadable files
            unreadable = problems['unreadable']
            if unreadable:
                print(f"\n  {R}{B}📕 UNREADABLE FILES ({len(unreadable)}){X}")
                print(f"  {D}{'─'*66}{X}")
                for e in unreadable:
                    fname = os.path.basename(e['filepath'])
                    rdbl = e.get('readability') or 'unknown'
                    rdbl_colors = {
                        'encrypted': Y, 'corrupted': R, 'scanned-image-only': Y,
                        'unreadable': R, 'no-text': R,
                    }
                    rc = rdbl_colors.get(rdbl, D)
                    comp = f"{e['completeness']*100:.0f}%" if e['completeness'] else '?'

                    print(f"    {rc}■{X} {B}{fname[:65]}{X}")
                    print(f"      {D}Reason:{X}  {rc}{rdbl}{X}")
                    print(f"      {D}Score:{X}   {comp}")
                    print(f"      {D}Path:{X}    {D}{e['filepath']}{X}")

                    # Try to show what metadata we do have
                    if e.get('metadata_json'):
                        try:
                            md = json.loads(e['metadata_json'])
                            has = [k for k, v in md.items()
                                   if v and k not in ('completeness', 'sources_used')
                                   and v != [] and v != 0]
                            if has:
                                print(f"      {D}Has:{X}     {', '.join(has)}")
                        except Exception:
                            pass
                    print()
            else:
                print(f"\n  {G}✓ No unreadable files{X}")

            # Error files
            errors = problems['errors']
            if errors:
                print(f"  {R}{B}❌ PROCESSING ERRORS ({len(errors)}){X}")
                print(f"  {D}{'─'*66}{X}")
                for e in errors:
                    fname = os.path.basename(e['filepath'])
                    print(f"    {R}■{X} {B}{fname[:65]}{X}")
                    print(f"      {D}Path:{X}    {D}{e['filepath']}{X}")
                    print()
            else:
                print(f"  {G}✓ No processing errors{X}")

            # Low completeness
            low = problems['low_completeness']
            if low:
                print(f"\n  {Y}{B}⚠ LOW COMPLETENESS ({len(low)} files below 60%){X}")
                print(f"  {D}{'─'*66}{X}")
                for e in low:
                    fname = os.path.basename(e['filepath'])
                    comp = f"{e['completeness']*100:.0f}%" if e['completeness'] else '?'
                    comp_color = R if (e['completeness'] or 0) < 0.4 else Y
                    title = e.get('title') or '(no title)'
                    sources = e.get('sources') or ''

                    print(f"    {comp_color}{comp:>4}{X}  {B}{title[:55]}{X}")
                    print(f"          {D}{fname[:60]}{X}")

                    # Show what's missing by parsing metadata
                    if e.get('metadata_json'):
                        try:
                            md = json.loads(e['metadata_json'])
                            fext = Path(fname).suffix.lower()
                            writable = FORMAT_WRITABLE_FIELDS.get(fext, set())
                            missing_w, missing_f = [], []
                            if not md.get('title'):
                                (missing_w if 'title' in writable else missing_f).append('title')
                            if not md.get('authors'):
                                (missing_w if 'authors' in writable else missing_f).append('authors')
                            if not md.get('publisher'):
                                (missing_w if 'publisher' in writable else missing_f).append('publisher')
                            if not md.get('isbn_13') and not md.get('isbn_10'):
                                (missing_w if 'isbn' in writable else missing_f).append('ISBN')
                            if not md.get('language'):
                                (missing_w if 'language' in writable else missing_f).append('language')
                            if not md.get('description'):
                                (missing_w if 'description' in writable else missing_f).append('description')
                            if missing_w:
                                print(f"          {Y}Missing: {', '.join(missing_w)}{X}")
                            if missing_f:
                                fmt = fext.lstrip('.').upper()
                                print(f"          {D}Not in {fmt}: {', '.join(missing_f)}{X}")
                        except Exception:
                            pass

                    if sources:
                        print(f"          {D}Sources: {sources}{X}")
                    print()
            else:
                print(f"\n  {G}✓ All processed files above 60% completeness{X}")

            # Summary
            stats = cache.get_stats()
            total_problems = len(unreadable) + len(errors) + len(low)
            print(f"{C}{'═'*70}{X}")
            print(f"  {D}Total tracked:{X} {stats['total']}  |  "
                  f"{D}Problems:{X} {R if total_problems else G}{total_problems}{X}  |  "
                  f"{D}Avg completeness:{X} {G}{stats['avg_completeness']*100:.0f}%{X}")
            print(f"{C}{'═'*70}{X}\n")

            cache.close()
            sys.exit(0)

        if args.show_cache:
            stats = cache.get_stats()
            print(f"\n{'='*60}")
            print(f"  CACHE: {cache_path}")
            print(f"{'='*60}")
            print(f"  Total tracked:    {stats['total']}")
            print(f"  Successful:       {stats['success']}")
            print(f"  Errors:           {stats['errors']}")
            print(f"  Unreadable:       {stats.get('unreadable', 0)}")
            print(f"  Avg completeness: {stats['avg_completeness']*100:.0f}%")
            entries = cache.list_entries(30)
            if entries:
                print(f"\n  Recent entries:")
                for e in entries:
                    title = (e['title'] or 'untitled')[:50]
                    comp = f"{e['completeness']*100:.0f}%" if e['completeness'] else '?'
                    rdbl = e.get('readability') or ''
                    print(f"    {comp:>4}  {e['status']:<8} {rdbl:<18} {title}")
                    print(f"          \033[90m{e['filepath']}\033[0m")
            print(f"{'='*60}\n")
        elif args.clear_cache:
            count = cache.clear_all()
            print(f"Cleared {count} entries from cache: {cache_path}")
        cache.close()
        sys.exit(0)

    # Flatten comma-separated Google API keys
    if args.google_api_key:
        flat = []
        for k in args.google_api_key:
            flat.extend([x.strip() for x in k.split(',') if x.strip()])
        args.google_api_key = flat if flat else None

    pipeline = EbookMetadataPipeline(args)
    pipeline.run()


if __name__ == '__main__':
    main()

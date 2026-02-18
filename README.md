# Ebook Metadata Pipeline — User Guide

## What It Does

This script scans a directory of ebook files, extracts and enriches their
metadata from multiple sources, writes improved metadata back into the files,
and renames them using a clean, consistent pattern. It handles encrypted files,
scanned PDFs, corrupted archives, and filenames in dozens of languages.

**Supported formats:** `.epub`, `.pdf`, `.mobi`, `.azw`, `.azw3`, `.fb2`,
`.txt`, `.html`, `.htm`, `.djvu`, `.cbz`, `.cbr`, `.lit`, `.doc`, `.docx`,
`.rtf`, `.odt`

---

## Installation

### Option 1: Auto-bootstrap (recommended)

```bash
python ebook_metadata_pipeline.py /path/to/ebooks --bootstrap-venv
```

This creates a virtual environment at `~/.venvs/ebook-pipeline`, installs all
Python dependencies, and re-launches the script inside that venv. Subsequent
runs auto-detect the venv and use it without the flag.

### Option 2: Manual venv

```bash
python -m venv ~/.venvs/ebook-pipeline
source ~/.venvs/ebook-pipeline/bin/activate
pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4
python ebook_metadata_pipeline.py /path/to/ebooks
```

### Option 3: System-wide (not recommended)

```bash
pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4 --break-system-packages
python ebook_metadata_pipeline.py /path/to/ebooks
```

### External tool (required for metadata writing)

```bash
sudo apt install calibre    # provides ebook-meta and ebook-convert
```

Calibre's `ebook-meta` writes metadata into files; `ebook-convert` is used as a
fallback text extractor. Without Calibre the script still runs but skips writing
and may fail to extract text from some formats.

---

## Dependencies

| Dependency | pip name | Role | Required? |
|---|---|---|---|
| `ebooklib` | `ebooklib` | EPUB read/write | For EPUB |
| `PyMuPDF` (fitz) | `PyMuPDF` | PDF text extraction + metadata write | For PDF |
| `lxml` | `lxml` | XML parsing (RDF, FB2, HTML) | **Critical** |
| `requests` | `requests` | API calls + cover downloads | **Critical** |
| `beautifulsoup4` | `beautifulsoup4` | HTML text extraction | **Critical** |
| `anthropic` | `anthropic` | Claude AI analysis | Optional |
| `mobi` | `mobi` | MOBI/AZW text extraction | For MOBI |
| `calibre` | system package | Metadata writing + fallback extraction | Recommended |

Missing non-critical modules cause graceful degradation — those formats process
with reduced capability, and the script warns you at startup.

---

## How the Pipeline Works

Each file passes through up to 7 stages. The pipeline short-circuits
intelligently — if enough metadata is already known, later stages are skipped.

### Stage-by-Stage

1. **Gutenberg RDF Catalog** — If the filename contains a Project Gutenberg ID
   (e.g. `pg12345`), the script looks it up in a local copy of the Gutenberg RDF
   catalog (~70K books). This is the fastest and most reliable source for public
   domain texts. Extracts title, authors, editors, translators, subjects (LCSH),
   LCC classification, language, rights, and publication date.

2. **Embedded Metadata** — Reads whatever metadata already exists inside the
   file. Uses `ebook-meta` (Calibre) plus format-native libraries (ebooklib for
   EPUB, PyMuPDF for PDF, mobi for MOBI/AZW). Records the original state so
   changes can be displayed as a diff later.

3. **Filename Parsing** — Parses the filename to extract author, title,
   publisher, year, edition, and series info. Uses an author-scoring heuristic to
   decide which part of `Title - Author Name.epub` is the title vs the author.
   Detects garbage embedded titles (like "Frontmatter" or "Microsoft Word") and
   replaces them from the filename.

4. **Text Extraction + Readability Check** — Extracts the first N pages of text
   (default 15) for two purposes: feeding to the AI analyzer, and detecting
   whether the file is actually readable. Reports encrypted, corrupted,
   scanned-image-only, and no-text files. Tries format-native extraction first,
   falls back to `ebook-convert` (Calibre).

5. **Public API Lookups** — Queries Open Library and Google Books to fill missing
   fields. Only fires if 2+ metadata fields are still missing after embedded +
   filename parsing. Implements rate limiting, exponential backoff, API key
   rotation, and circuit-breaker logic. Results are merged without overwriting
   existing data.

6. **AI Text Analysis** — If the completeness score is still below the threshold
   (default 40%), sends the extracted text to Claude (Sonnet) and asks it to
   identify the book and return structured metadata as JSON. There is also an AI
   fallback mode for completely unreadable files that analyzes the filename alone.

7. **Genre/Subject Inference** — If no genres or subjects were found from any
   source, infers them from existing tags, title keywords, and other metadata
   using pattern matching (no API call).

After all stages, the script:

- **Displays a diff** showing what was found in the file vs what changed, split
  into three categories: writable changes, API-found-but-unstorable fields, and
  derived-from-existing data.
- **Writes metadata** back into the file using Calibre's `ebook-meta` plus
  format-specific writers (ebooklib for EPUB extras, PyMuPDF for PDF).
- **Renames the file** using the pattern:
  `Title (Year) [Edition] (Series #N) - Author.ext`
- **Caches results** in a SQLite database so re-runs skip already-processed files.

---

## ASCII Workflow Tree

```
START
  │
  ├─── Discover files (recursive walk, filter by extension)
  │    ├── Dedup by path + (filename, size)
  │    ├── Apply --limit N
  │    └── Check cache (skip already-processed unless --force)
  │
  │  ┌──────────── FOR EACH FILE ────────────┐
  │  │                                        │
  │  │  [0] Pre-checks                        │
  │  │   ├── --max-file-size? ──YES──► SKIP   │
  │  │   └── Compute SHA-256 hash             │
  │  │                                        │
  │  │  [1] Gutenberg RDF Catalog             │
  │  │   ├── --skip-rdf? ──YES──► skip        │
  │  │   ├── Detect PG ID from filename       │
  │  │   │   └── Found? ──YES──► Parse RDF    │
  │  │   │       └── Merge: title, authors,   │
  │  │   │           subjects, LCC, language   │
  │  │   └── No ID? ──► continue              │
  │  │                                        │
  │  │  [2] Embedded Metadata                 │
  │  │   ├── ebook-meta (Calibre CLI)         │
  │  │   ├── Format-native (ebooklib/fitz/    │
  │  │   │   mobi module)                     │
  │  │   ├── Snapshot "original_meta"         │
  │  │   └── Merge into working metadata      │
  │  │                                        │
  │  │  [2.5] Filename Parsing                │
  │  │   ├── Extract: author, title, year,    │
  │  │   │   publisher, edition               │
  │  │   ├── Garbage title detection           │
  │  │   │   └── Bad title? ──► replace from  │
  │  │   │       filename                     │
  │  │   └── Fill any still-empty fields      │
  │  │                                        │
  │  │  [2.75] Text Extraction                │
  │  │   ├── Try native extractor             │
  │  │   │   └── epub-native / pdf-pymupdf /  │
  │  │   │       mobi-native / fb2-native     │
  │  │   ├── Fail? ──► Try calibre-convert    │
  │  │   ├── Detect: encrypted, corrupted,    │
  │  │   │   scanned-image-only, no-text      │
  │  │   └── ReadabilityResult recorded       │
  │  │                                        │
  │  │  [3] Public APIs                       │
  │  │   ├── --skip-api? ──YES──► skip        │
  │  │   ├── Missing ≥2 fields? ──NO──► skip  │
  │  │   ├── Open Library search              │
  │  │   │   ├── By ISBN (if known)           │
  │  │   │   └── By title + author            │
  │  │   ├── Google Books search              │
  │  │   │   ├── By ISBN (if known)           │
  │  │   │   └── By title + author            │
  │  │   │   └── Requires --google-api-key    │
  │  │   └── Merge new fields (never          │
  │  │       overwrite existing)              │
  │  │                                        │
  │  │  [4] AI Analysis                       │
  │  │   ├── --skip-ai? ──YES──► skip         │
  │  │   ├── Has text AND score < threshold?  │
  │  │   │   └── YES ──► Send text to Claude  │
  │  │   │       └── Parse JSON response      │
  │  │   │       └── Merge results            │
  │  │   ├── Unreadable file + AI available?  │
  │  │   │   └── YES ──► AI filename fallback │
  │  │   │       └── Identify book from name  │
  │  │   └── Neither? ──► skip                │
  │  │                                        │
  │  │  [4.5] Genre/Subject Inference         │
  │  │   └── Still no genres/subjects?        │
  │  │       └── Infer from tags + title      │
  │  │                                        │
  │  │  [5] Metadata Write                    │
  │  │   ├── --skip-write? ──YES──► skip      │
  │  │   ├── --dry-run? ──YES──► skip         │
  │  │   ├── No writable changes? ──► skip    │
  │  │   ├── Corrupted/encrypted? ──► skip    │
  │  │   ├── Calibre ebook-meta               │
  │  │   │   └── title, authors, publisher,   │
  │  │   │       date, language, description, │
  │  │   │       tags, series, ISBN, cover,   │
  │  │   │       identifiers                  │
  │  │   ├── EPUB extra (ebooklib)            │
  │  │   │   └── subjects, editors,           │
  │  │   │       translators, illustrators,   │
  │  │   │       rights, source               │
  │  │   └── PDF extra (PyMuPDF)              │
  │  │       └── fitz metadata dict           │
  │  │                                        │
  │  │  [6] Rename                            │
  │  │   ├── --skip-rename? ──YES──► skip     │
  │  │   ├── --dry-run? ──YES──► show only    │
  │  │   ├── Build new name:                  │
  │  │   │   Title (Year) [Edition]           │
  │  │   │   (Series #N) - Author.ext         │
  │  │   ├── Collision? ──► append (1), (2)   │
  │  │   └── os.rename()                      │
  │  │                                        │
  │  │  [7] Report + Cache                    │
  │  │   ├── Display: completeness score,     │
  │  │   │   sources used, missing fields     │
  │  │   ├── Write to SQLite cache            │
  │  │   └── Append to JSON report            │
  │  │                                        │
  │  └────────────────────────────────────────┘
  │
  ├─── Print summary statistics
  └─── Save report to log directory
```

---

## Completeness Scoring

Each file gets a 0–100% score based on weighted fields:

| Field | Weight | Field | Weight |
|---|---|---|---|
| `title` | 20 | `isbn_13` | 5 |
| `authors` | 20 | `isbn_10` | 3 |
| `publication_date` | 10 | `publisher` | 5 |
| `description` | 10 | `genres` | 5 |
| `subjects` | 8 | `series` | 3 |
| `language` | 5 | `page_count` | 3 |
| | | `cover_url` | 3 |

The `--ai-threshold` option controls when AI analysis kicks in. If the score is
below this threshold after stages 1–3, the text is sent to Claude.

---

## Rename Pattern

Files are renamed to:

```
Title (Year) [Edition] (Series #Index) - Author.ext
```

**Examples:**

| Before | After |
|---|---|
| `pg1342.epub` | `Pride and Prejudice (1813) - Jane Austen.epub` |
| `9780134685991.pdf` | `Effective Java (2018) [3rd Edition] - Joshua Bloch.pdf` |
| `unknown_scifi_book.mobi` | `Dune (1965) (Dune #1) - Frank Herbert.mobi` |
| `GoF patterns.pdf` | `Design Patterns (1994) - Erich Gamma et al..pdf` |

Multiple authors show as `Author1 & Author2` for two, or `Author1 et al.` for
three or more. Collision handling appends `(1)`, `(2)`, etc.

---

## Change Display Categories

When processing each file, the pipeline shows what's new in three sections:

- **CHANGES** (green `+` or yellow `✎`) — New or updated data that can be
  written to this file format. These trigger the metadata write stage.
- **API FOUND** (cyan `◆`) — Data discovered from APIs but the file format
  doesn't support it (e.g. LCC classification in a PDF, page count in any
  ebook). Saved to cache only.
- **DERIVED** (gray `~`) — Inferred from data already embedded in the file
  (e.g. genres inferred from existing tags). Display only, not written.

---

## Caching

Results are stored in a SQLite database at `/tmp/ebook_metadata_<hash>.db`
(keyed to the target directory). On re-runs, already-processed files are skipped
unless `--force` is used. The cache tracks:

- File path, size, and modification time
- Completeness score and processing status
- Readability status (encrypted, corrupted, scanned, etc.)
- Full metadata JSON snapshot
- Sources used and processing timestamp

If a file is renamed, the cache entry is updated to the new path and the old
entry is removed.

---

## All Options

### Required Argument

| Argument | Description |
|---|---|
| `ebook_dir` | Directory to scan (recursive). |

```bash
python ebook_metadata_pipeline.py /data/ebooks
```

---

### Data Source Options

| Option | Default | Description |
|---|---|---|
| `--rdf-catalog PATH` | `~/gutenberg-rdf` | Path to Gutenberg RDF catalog directory. |
| `--auto-download-rdf` | off | Download the RDF catalog (~300MB) without prompting. |
| `--anthropic-api-key KEY` | `$ANTHROPIC_API_KEY` | API key for Claude AI analysis. |
| `--google-api-key KEY` | none | Google Books API key. Repeat for multiple keys (rotation). |

**Gutenberg RDF Catalog:** The script will prompt to download it on first run
if not found. Use `--auto-download-rdf` for unattended setups. The catalog
is a ~300MB tarball that extracts to ~2GB of individual RDF/XML files.

```bash
# First run — auto-download catalog
python ebook_metadata_pipeline.py /data/ebooks --auto-download-rdf

# Use a custom catalog location
python ebook_metadata_pipeline.py /data/ebooks --rdf-catalog /mnt/data/gutenberg-rdf

# Multiple Google API keys for quota rotation
python ebook_metadata_pipeline.py /data/ebooks \
  --google-api-key AIzaSyA...key1 \
  --google-api-key AIzaSyB...key2 \
  --google-api-key AIzaSyC...key3

# AI via environment variable
export ANTHROPIC_API_KEY="sk-ant-..."
python ebook_metadata_pipeline.py /data/ebooks
```

---

### Skip Flags

| Option | Effect |
|---|---|
| `--skip-rdf` | Skip Gutenberg RDF catalog lookups entirely. |
| `--skip-api` | Skip Open Library and Google Books API calls. |
| `--skip-ai` | Skip Claude AI text analysis. |
| `--skip-write` | Skip writing metadata back into files. |
| `--skip-rename` | Skip renaming files. |
| `--skip-dep-install` | Don't auto-install missing Python packages. |
| `--dry-run` | Preview all changes without modifying any files (no writes, no renames). |

These can be combined freely:

```bash
# Read-only audit — see what the pipeline would do
python ebook_metadata_pipeline.py /data/ebooks --dry-run

# Offline mode — embedded + filename only, no network
python ebook_metadata_pipeline.py /data/ebooks --skip-api --skip-ai --skip-rdf

# Enrich metadata but don't rename files
python ebook_metadata_pipeline.py /data/ebooks --skip-rename

# Only extract and display — no writes at all
python ebook_metadata_pipeline.py /data/ebooks --skip-write --skip-rename

# API lookups only, no AI spending
python ebook_metadata_pipeline.py /data/ebooks --skip-ai
```

---

### Processing Controls

| Option | Default | Description |
|---|---|---|
| `--limit N` | all | Process only the first N files. |
| `--max-pages N` | `15` | Max pages of text to extract per file. |
| `--max-file-size N` | `0` (unlimited) | Skip files larger than N megabytes. |
| `--ai-threshold F` | `0.4` | Completeness score (0.0–1.0) below which AI analysis triggers. |
| `--threads N` | `1` | Number of parallel processing threads. |
| `--force` | off | Reprocess all files, ignoring the cache. |
| `--verbose` / `-v` | off | Show detailed extraction and debug info. |

```bash
# Test with 5 files first
python ebook_metadata_pipeline.py /data/ebooks --limit 5 --verbose

# Process a huge library in parallel
python ebook_metadata_pipeline.py /data/ebooks --threads 4

# More aggressive AI usage (trigger at 60% instead of 40%)
python ebook_metadata_pipeline.py /data/ebooks --ai-threshold 0.6

# Skip giant PDFs (over 200MB)
python ebook_metadata_pipeline.py /data/ebooks --max-file-size 200

# Force reprocess everything (ignore cache)
python ebook_metadata_pipeline.py /data/ebooks --force

# Extract more text for better AI analysis
python ebook_metadata_pipeline.py /data/ebooks --max-pages 30
```

---

### Cache Management

| Option | Effect |
|---|---|
| `--show-cache` | Print cache statistics and exit. |
| `--show-problems` | Show unreadable, errored, and low-completeness files from cache. |
| `--clear-cache` | Delete the processing cache and exit. |

```bash
# Check how many books are tracked, average completeness
python ebook_metadata_pipeline.py /data/ebooks --show-cache

# Find problem files — encrypted, corrupted, scanned PDFs
python ebook_metadata_pipeline.py /data/ebooks --show-problems

# Wipe cache and start fresh
python ebook_metadata_pipeline.py /data/ebooks --clear-cache
```

---

### Virtual Environment Options

| Option | Default | Description |
|---|---|---|
| `--bootstrap-venv` | off | Create venv, install deps, re-launch inside it. |
| `--venv-dir PATH` | `~/.venvs/ebook-pipeline` | Custom venv location. |
| `--recreate-venv` | off | Destroy and rebuild the venv from scratch. |

```bash
# First-time setup
python ebook_metadata_pipeline.py /data/ebooks --bootstrap-venv

# Custom venv location
python ebook_metadata_pipeline.py /data/ebooks --bootstrap-venv --venv-dir /opt/venvs/ebooks

# Rebuild after dependency updates
python ebook_metadata_pipeline.py /data/ebooks --recreate-venv
```

---

### Logging

| Option | Default | Description |
|---|---|---|
| `--log-dir PATH` | `<ebook_dir>/.metadata_logs` | Directory for log files. |

The pipeline creates several log files:

| Log File | Contents |
|---|---|
| `pipeline.log` | Full timestamped processing log. |
| `pipeline_stats.jsonl` | One JSON line per file: path, status, score, sources, timing. |
| `unreadable.log` | Files that couldn't have text extracted, with reasons. |
| `report.json` | Comprehensive JSON report of all results. |

```bash
# Centralized logging
python ebook_metadata_pipeline.py /data/ebooks --log-dir /var/log/ebook-pipeline
```

---

## Common Recipes

### Full enrichment of a new library

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python ebook_metadata_pipeline.py /data/ebooks \
  --bootstrap-venv \
  --auto-download-rdf \
  --google-api-key AIzaSy... \
  --verbose
```

### Quick local-only pass (no network, no cost)

```bash
python ebook_metadata_pipeline.py /data/ebooks \
  --skip-api --skip-ai --skip-rdf
```

### Audit-only — see what would change without touching files

```bash
python ebook_metadata_pipeline.py /data/ebooks --dry-run --verbose
```

### Process only unhandled files from a previous partial run

```bash
# The cache automatically skips already-processed files
python ebook_metadata_pipeline.py /data/ebooks
```

### Re-process failures only

```bash
# 1. See what failed
python ebook_metadata_pipeline.py /data/ebooks --show-problems

# 2. Clear cache and reprocess with more aggressive settings
python ebook_metadata_pipeline.py /data/ebooks --force --ai-threshold 0.7
```

### Large library with rate limit protection

```bash
python ebook_metadata_pipeline.py /data/ebooks \
  --threads 2 \
  --google-api-key KEY1 \
  --google-api-key KEY2 \
  --google-api-key KEY3 \
  --max-file-size 500
```

### Project Gutenberg collection

```bash
python ebook_metadata_pipeline.py /data/gutenberg-books \
  --rdf-catalog /data/gutenberg-rdf \
  --skip-api --skip-ai
```

The RDF catalog alone provides title, authors, subjects, language, LCC, and
rights for Gutenberg texts — APIs and AI are unnecessary.

---

## Decision Logic Detail

### When does the API stage fire?

Only when 2 or more of these fields are missing after embedded + filename:
subjects/genres, language, description, page count, cover, ISBN, publisher,
LCC/DDC classification.

### When does the AI stage fire?

Two conditions (checked independently):

1. **Normal mode:** Extracted text exists (>100 chars) AND completeness score
   is below `--ai-threshold` (default 0.4 = 40%).
2. **Fallback mode:** Text extraction completely failed (unreadable file) AND
   the Anthropic API is available. Sends only the filename and file size.

### When does metadata get written?

All three conditions must be true:
- `--dry-run` is not set
- `--skip-write` is not set
- At least one field has a writable change (new data that the file format
  can actually store)

Writes are also skipped for corrupted or encrypted files even if changes exist.

### What determines "writable" vs "unstorable"?

Each format has a known set of fields it can store (documented in the companion
`ebook_format_metadata_reference.md`). For example:
- EPUB can store title, authors, publisher, date, language, description, ISBN,
  tags, series, and embedded covers.
- PDF can store all of those except embedded covers.
- CBZ/CBR can store nothing (image archives).

If the API returns page count (no ebook format stores this) or LCC
classification codes, those appear under "API FOUND" instead of "CHANGES" and
are saved to the cache but not written to the file.

---

## Readability Statuses

| Status | Icon | Meaning |
|---|---|---|
| `readable` | ✓ | Text successfully extracted. |
| `encrypted` | 🔒 | DRM or password protection detected. |
| `corrupted` | ✗ | File is damaged or empty. |
| `scanned-image-only` | 📷 | PDF contains only images (no OCR text layer). |
| `ai-recovered` | 🤖 | Text extraction failed but AI identified the book from filename. |
| `unreadable` | ✗ | All extraction methods failed. |
| `no-text` | ⚠ | File opened but contained no extractable text. |

# 📚 Ebook Metadata Filler

A comprehensive metadata extraction, enrichment, writing, and renaming tool for large ebook collections. Processes `.epub`, `.pdf`, `.mobi`, `.azw`, `.azw3`, `.fb2`, and many more formats through a multi-stage pipeline that pulls from free catalogs, public APIs, and AI analysis to build rich, complete metadata — then writes it back into files and renames them cleanly.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT: Ebook File                           │
│  (.epub, .pdf, .mobi, .azw, .azw3, .fb2, .txt, .html, .djvu, ...) │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 0 — Discovery & Deduplication                                │
│  • Recursive directory scan for supported formats                   │
│  • Dedup by path, symlink, and (filename + size)                    │
│  • Cache check: skip files already processed (unless --force)       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Gutenberg RDF Catalog                                    │
│  Fastest, most reliable for Project Gutenberg books.                │
│  • Detects PG ID from filename (pg12345) or path (/12345/)         │
│  • Parses local RDF/XML for title, authors, subjects, LCC, rights  │
│  • Auto-downloads catalog (~300MB) on first run if missing          │
│                                                                     │
│  Skip: --skip-rdf                                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Embedded Metadata Extraction                             │
│  Reads what's already inside the file.                              │
│  • Calibre CLI (ebook-meta) for universal format support            │
│  • Format-specific deep extraction:                                 │
│    ├── EPUB: OPF metadata (DC elements, roles, identifiers)        │
│    ├── PDF:  XMP/Info dict via PyMuPDF (+ page count)              │
│    └── FB2:  XML title-info, publish-info, sequences               │
│  • Garbage detection: rejects titles like "out.jpg", "Untitled",   │
│    "Frontmatter", swapped author/title fields, invalid dates       │
│  • Filename parsing fallback with author/title scoring heuristics  │
│  • Author cleanup: deduplication, credential stripping, semicolon  │
│    splitting, "Last, First" normalization                           │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Public API Enrichment                                    │
│  Fills gaps with free public book databases.                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Phase 1: Open Library (always first — free, no key needed)  │   │
│  │  • ISBN lookup → title/author search                         │   │
│  │  • Returns: ISBN, language, subjects, LCC/DDC, pages,        │   │
│  │    cover, description, publisher, original pub date          │   │
│  │  • Works API follow-up for richer descriptions               │   │
│  └──────────────────────────┬───────────────────────────────────┘   │
│                             │                                       │
│                    ┌────────┴────────┐                               │
│                    │  Decision Gate  │                               │
│                    └────────┬────────┘                               │
│              ┌──────────────┼──────────────┐                        │
│              ▼              ▼              ▼                         │
│         OL found       OL found       OL found                      │
│        nothing       completeness    completeness                   │
│                        < 70%           ≥ 70%                        │
│              │              │              │                         │
│              ▼              ▼              ▼                         │
│  ┌───────────────┐ ┌───────────────┐  ┌────────────┐               │
│  │ Google Books  │ │ Google Books  │  │   SKIP     │               │
│  │  (fallback)   │ │  (fallback)   │  │  Google ✓  │               │
│  └───────────────┘ └───────────────┘  └────────────┘               │
│                                                                     │
│  Google Books features:                                             │
│  • API key rotation (round-robin, per-key circuit breakers)         │
│  • Exponential backoff on 429s with Retry-After support             │
│  • Auto-disables after consecutive failures                         │
│                                                                     │
│  Skip: --skip-api          Threshold: --api-threshold (default 0.7) │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — AI Text Analysis (Claude API)                            │
│  Last resort for books APIs couldn't identify.                      │
│  • Extracts text from first ~15,000 chars of the book               │
│  • Sends to Claude with structured JSON prompt                      │
│  • Returns: title, authors, publisher, date, language,              │
│    description, subjects, genres, series, ISBNs                     │
│  • Only fires when completeness < ai-threshold (default 0.4)       │
│                                                                     │
│  Skip: --skip-ai          Threshold: --ai-threshold (default 0.4)  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4.5 — Genre & Subject Inference (no API needed)              │
│  Rule-based classification from existing metadata.                  │
│  • DDC code → genre mapping (Dewey Decimal Classification)          │
│  • LCC code → genre mapping (Library of Congress Classification)    │
│  • Title/subtitle keyword → genre/subject inference                 │
│  • Publisher → genre hints (e.g., Packt → Computers)                │
│  • Tag normalization and cleanup                                    │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — Metadata Writing                                         │
│  Writes enriched metadata back into the file.                       │
│  • Calibre ebook-meta: title, authors, publisher, date, language,   │
│    tags, series, ISBN, identifiers, cover image download & embed    │
│  • EPUB extras: DC subjects, contributor roles, source links        │
│  • PDF extras: XMP metadata via PyMuPDF                             │
│  • Cover download: auto-fetches from OL/Google, validates size      │
│                                                                     │
│  Skip: --skip-write         Preview: --dry-run                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — File Renaming                                            │
│  Renames files to a clean, consistent format.                       │
│                                                                     │
│  Format: Title (Year) [Edition] (Series #N) - Author.ext            │
│                                                                     │
│  Examples:                                                          │
│    Mastering Malware Analysis (2022) [2nd Edition] - Alexey K....   │
│    Foundations of Analog and Digital Electronic Circuits (2005)...   │
│    Search Inside Yourself (2012) - Chade-Meng Tan.mobi              │
│                                                                     │
│  • Collision handling (appends counter)                              │
│  • "Last, First" → "First Last" for filenames                       │
│  • Multi-author: "A & B" or "A et al."                              │
│                                                                     │
│  Skip: --skip-rename        Preview: --dry-run                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  OUTPUT                                                             │
│  • Enriched ebook file with embedded metadata + cover               │
│  • Cleanly renamed file                                             │
│  • SQLite cache entry (skip on re-run)                              │
│  • JSON processing report + JSONL stats log                         │
│  • Per-file error log                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Completeness Scoring

Every book gets a weighted completeness score (0–100%) that drives pipeline decisions:

| Field         | Weight | Notes                          |
|---------------|--------|--------------------------------|
| Title         | 20     | Most important identifier      |
| Authors       | 20     | Critical for naming/search     |
| Pub. Date     | 10     | Year or full ISO date          |
| Description   | 10     | Synopsis / back cover text     |
| Subjects      | 8      | LCSH headings, topic keywords  |
| Language      | 5      | ISO 639 code                   |
| ISBN-13       | 5      | Primary book identifier        |
| Publisher     | 5      | Publishing house               |
| Genres        | 5      | BISAC / broad categories       |
| ISBN-10       | 3      | Legacy identifier              |
| Series        | 3      | Series name + index            |
| Page Count    | 3      | Physical page count            |
| Cover URL     | 3      | Cover image source             |
| **Total**     | **100**|                                |

---

## Installation

### Prerequisites

```bash
# Core dependencies
pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4 --break-system-packages

# Calibre CLI tools (required for metadata writing)
sudo apt install calibre
```

### Gutenberg RDF Catalog (optional, auto-downloads on first run)

```bash
# Manual download if preferred
wget https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2 -O ~/gutenberg-rdf/rdf-files.tar.bz2
cd ~/gutenberg-rdf && tar xjf rdf-files.tar.bz2
```

---

## Quick Start

```bash
# Preview what would change (safe, no modifications)
python3 do.py /path/to/ebooks --dry-run --verbose

# Run for real with all defaults
python3 do.py /path/to/ebooks

# Process with 4 threads, aggressive AI enrichment
python3 do.py /path/to/ebooks --threads 4 --ai-threshold 0.7

# RDF + embedded only (no network calls)
python3 do.py /path/to/ebooks --skip-api --skip-ai
```

---

## CLI Reference

### Positional Arguments

| Argument     | Description                                   |
|-------------|-----------------------------------------------|
| `ebook_dir` | Directory containing ebook files (recursive)  |

### Source Control

| Flag                     | Default                  | Description                                                       |
|--------------------------|--------------------------|-------------------------------------------------------------------|
| `--rdf-catalog PATH`    | `~/gutenberg-rdf`        | Path to Gutenberg RDF catalog directory                           |
| `--skip-rdf`            | off                      | Skip Gutenberg RDF catalog entirely                               |
| `--auto-download-rdf`   | off                      | Auto-download RDF catalog without prompting if missing            |
| `--skip-api`            | off                      | Skip all public API lookups (Open Library + Google Books)         |
| `--skip-ai`             | off                      | Skip AI text analysis via Claude                                  |

### Threshold Tuning

| Flag                     | Default | Description                                                        |
|--------------------------|---------|--------------------------------------------------------------------|
| `--api-threshold FLOAT` | `0.7`   | Completeness below which API lookups trigger (0.0–1.0)             |
| `--ai-threshold FLOAT`  | `0.4`   | Completeness below which AI analysis triggers (0.0–1.0)            |

**Examples:**

```bash
# Always run AI (even for nearly complete books)
--ai-threshold 1.0

# Only use AI for truly empty metadata
--ai-threshold 0.2

# Conservative API usage (only for very incomplete books)
--api-threshold 0.3

# Aggressive enrichment (API + AI for anything under 90%)
--api-threshold 0.9 --ai-threshold 0.9
```

### API Keys

| Flag                       | Default          | Description                                                     |
|----------------------------|------------------|-----------------------------------------------------------------|
| `--google-api-key KEY`    | built-in default | Google Books API key. Repeat flag or comma-separate for multiple |
| `--anthropic-api-key KEY` | `$ANTHROPIC_API_KEY` env | Anthropic API key for Claude AI analysis                    |

**Multiple Google API keys (round-robin rotation):**

```bash
# Comma-separated
--google-api-key "AIza...one,AIza...two,AIza...three"

# Repeated flags
--google-api-key AIza...one --google-api-key AIza...two
```

Keys rotate round-robin per request. A key that gets 403'd is individually disabled while others continue. When all keys are exhausted, Google Books is disabled for the remainder of the run.

### Output Control

| Flag             | Default | Description                                         |
|------------------|---------|-----------------------------------------------------|
| `--dry-run`      | off     | Preview all changes without modifying any files      |
| `--skip-write`   | off     | Skip writing metadata back into files                |
| `--skip-rename`  | off     | Skip file renaming                                   |

### Processing Control

| Flag             | Default | Description                                         |
|------------------|---------|-----------------------------------------------------|
| `--threads N`    | `1`     | Parallel processing threads                          |
| `--limit N`      | all     | Process only the first N files                       |
| `--force`        | off     | Reprocess all files, ignoring cache                  |
| `--verbose`      | off     | Show debug-level output                              |
| `--log-dir PATH` | `<ebook_dir>/.metadata_logs` | Directory for log files            |

### Cache Management

| Flag             | Description                                    |
|------------------|------------------------------------------------|
| `--cache-stats`  | Show cache statistics and exit                  |
| `--cache-clear`  | Clear the processing cache and exit             |

The SQLite cache lives in `/tmp/ebook_metadata_<hash>.db` (or `<ebook_dir>/.metadata_cache.db` if `/tmp` is unavailable). It tracks processed files by path, size, and mtime — changed files are automatically reprocessed.

---

## Output Format

Each file produces a detailed processing card:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [7/273] (3%)

┌─── 📖 Axelson - Serial Port Complete (2007).pdf
│  .pdf  •  4130 KB
│  ⚠ Embedded metadata looks suspect — using filename
│  ↻ Searching APIs...
│    ✓ Open Library: ISBN, language, pages, cover, subjects, publisher, LCC
│    ⊘ Google Books: skipped (OL sufficient at 97%)
│  ⟶ Serial Port Complete (2007) - Axelson.pdf
│
│  ─── FOUND IN FILE ───
│    Title:       Axelson
│    Authors:     Serial Port Complete (2007)
│
│  ─── CHANGES (DRY RUN) ───
│  ✎ Title         Serial Port Complete
│    was:          Axelson
│  ✎ Authors       Axelson
│    was:          Serial Port Complete (2007)
│  + Publisher     Lakeview Research
│  + ISBN-13       9781931448079
│  + Language      eng
│  + Pages         343
│  + Cover         ✓ covers.openlibrary.org
│
│  ✅ 97% complete  •  2.0s  •  ☑ Embedded  ☑ OpenLibrary
└────────────────────────────────────────────────────────────
```

**Legend:**

| Symbol | Meaning |
|--------|---------|
| `+`    | New field added (was empty) |
| `✎`    | Existing field modified |
| `✓`    | API returned results |
| `✗`    | API returned no results or errored |
| `⊘`    | API intentionally skipped |
| `⚠`    | Warning (garbage metadata detected, etc.) |
| `☑`    | Source contributed to final metadata |

---

## API Fallback Strategy

The pipeline minimizes paid/rate-limited API calls:

```
         ┌──────────────┐
         │  Need data?  │
         └──────┬───────┘
                │
                ▼
       ┌────────────────┐
       │ Open Library    │  ◄── Always first (free, no key, no hard quota)
       │ ISBN → Search   │
       └───────┬────────┘
               │
         ┌─────┴─────┐
         │           │
    Got results?  No results
         │           │
         ▼           │
   Completeness      │
     ≥ 70%?          │
    ┌───┴───┐        │
   Yes      No       │
    │       │        │
    ▼       ▼        ▼
  DONE   ┌──────────────┐
  (skip  │ Google Books  │  ◄── Only when OL fails or insufficient
  Google)│ ISBN → Search │      Rate-limited, key rotation, circuit breakers
         └──────┬───────┘
                │
           Got results?
          ┌─────┴─────┐
         Yes          No
          │            │
          ▼            ▼
        DONE     Completeness
                   < 40%?
                 ┌───┴───┐
                Yes      No
                 │        │
                 ▼        ▼
           ┌──────────┐  DONE
           │ Claude AI │  ◄── Last resort: extracts metadata from book text
           │ Analysis  │      Requires --anthropic-api-key or $ANTHROPIC_API_KEY
           └──────────┘
```

---

## Supported Formats

| Format | Read Metadata | Write Metadata | Text Extraction |
|--------|:---:|:---:|:---:|
| `.epub` | ✅ OPF + Calibre | ✅ Calibre + OPF | ✅ ebooklib |
| `.pdf` | ✅ PyMuPDF + Calibre | ✅ PyMuPDF + Calibre | ✅ PyMuPDF |
| `.mobi` | ✅ Calibre | ✅ Calibre | ✅ mobi lib / Calibre |
| `.azw` / `.azw3` | ✅ Calibre | ✅ Calibre | ✅ mobi lib / Calibre |
| `.fb2` | ✅ Native XML + Calibre | ✅ Calibre | ✅ Native XML |
| `.txt` / `.html` | ✅ Calibre | ✅ Calibre | ✅ Direct read |
| `.djvu` | ✅ Calibre | ✅ Calibre | via Calibre convert |
| `.cbz` / `.cbr` | ✅ Calibre | ✅ Calibre | — |
| `.lit` | ✅ Calibre | ✅ Calibre | via Calibre convert |
| `.doc` / `.docx` / `.rtf` / `.odt` | ✅ Calibre | ✅ Calibre | via Calibre convert |

---

## Garbage Detection

Embedded metadata in ebooks is frequently wrong — especially in PDFs where tools like InDesign, Acrobat, or scanning software inject nonsense. The pipeline detects and corrects:

| Problem | Example | Action |
|---------|---------|--------|
| Filename as title | `out.jpg`, `1931448043.pdf`, `0750657847-prelims.pdf` | Replace with parsed filename |
| Structural page name | `Frontmatter`, `Preface`, `Table of Contents`, `Copyright` | Replace with parsed filename |
| App artifacts | `Microsoft Word - doc`, `Untitled`, `module tem-1` | Replace with parsed filename |
| Swapped author/title | Title: `"Barton"` Author: `"Radar Technology Encyclopedia"` | Swap from filename parsing |
| Invalid dates | `0101-01-01T00:00:00` | Replace with year from filename |
| Duplicate authors | `"Cameron Malin"` + `"Cameron H. Malin"` | Deduplicate by name overlap |
| Credential suffixes | `"Eoghan Casey BS MA"` | Strip non-name tokens |
| Title-like authors | Author: `"Comprehensive Guide To Digital Electronics"` | Remove, use filename author |

---

## Logging & Reports

All runs generate logs in `<ebook_dir>/.metadata_logs/` (or `--log-dir`):

| File | Content |
|------|---------|
| `pipeline_YYYYMMDD_HHMMSS.log` | Full trace log (all levels including TRACE) |
| `errors_YYYYMMDD_HHMMSS.log` | Errors only |
| `stats_YYYYMMDD_HHMMSS.jsonl` | One JSON line per file: status, completeness, sources, timing |
| `report_YYYYMMDD_HHMMSS.json` | Full processing report with per-file results |

---

## Common Workflows

### First run on a new collection

```bash
# Preview everything first
python3 do.py /data/ebooks --dry-run --verbose --threads 4

# If it looks good, run for real
python3 do.py /data/ebooks --threads 4
```

### Re-enrich low-quality files

```bash
# Check what's in cache
python3 do.py /data/ebooks --cache-stats

# Reprocess everything, enable AI for files under 70%
python3 do.py /data/ebooks --force --ai-threshold 0.7 --threads 4
```

### Offline mode (no network)

```bash
python3 do.py /data/ebooks --skip-api --skip-ai
```

### Test on a subset

```bash
python3 do.py /data/ebooks --limit 10 --dry-run --verbose
```

---

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `ebooklib` | EPUB reading/writing | `pip install ebooklib` |
| `PyMuPDF` (fitz) | PDF metadata + text extraction | `pip install PyMuPDF` |
| `lxml` | XML parsing | `pip install lxml` |
| `requests` | HTTP for API calls | `pip install requests` |
| `anthropic` | Claude AI API client | `pip install anthropic` |
| `mobi` | MOBI/AZW extraction | `pip install mobi` |
| `beautifulsoup4` | HTML text extraction | `pip install beautifulsoup4` |
| **Calibre** | `ebook-meta` CLI for reading/writing metadata | `sudo apt install calibre` |

```bash
# Install all Python deps at once
pip install ebooklib PyMuPDF lxml requests anthropic mobi beautifulsoup4 --break-system-packages
```

---

## License

MIT

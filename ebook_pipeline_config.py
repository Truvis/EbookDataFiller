#!/usr/bin/env python3
"""
Ebook Pipeline - Configuration Loader
======================================
Loads configuration from YAML file with CLI flag overrides.

Priority (highest wins):
  1. CLI flags (explicit only, not defaults)
  2. Environment variables (ANTHROPIC_API_KEY, GOOGLE_API_KEY)
  3. Config file (ebook_pipeline.yaml)
  4. Built-in defaults

Config file search order:
  1. --config <path>  (explicit)
  2. ./ebook_pipeline.yaml  (current directory)
  3. <ebook_dir>/ebook_pipeline.yaml  (alongside ebooks)
  4. ~/.config/ebook-pipeline/config.yaml  (user config)
  5. /etc/ebook-pipeline/config.yaml  (system config)
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def load_yaml(path: str) -> Dict[str, Any]:
    """Load YAML config file. Uses PyYAML if available, falls back to basic parser."""
    try:
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}
        return data
    except ImportError:
        # Minimal YAML parser for simple key: value files
        return _basic_yaml_parse(path)


def _basic_yaml_parse(path: str) -> Dict[str, Any]:
    """Fallback YAML parser for simple flat configs (no PyYAML dependency)."""
    import re
    data = {}
    current_section = None
    current_subsection = None

    with open(path, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue

            # Detect indentation level
            indent = len(line) - len(line.lstrip())

            # Top-level key
            if indent == 0 and ':' in stripped:
                key, _, val = stripped.partition(':')
                key = key.strip()
                val = val.strip()
                if val and not val.startswith('#'):
                    data[key] = _parse_value(val)
                    current_section = None
                else:
                    current_section = key
                    current_subsection = None
                    if key not in data:
                        data[key] = {}

            # Second-level key
            elif indent > 0 and current_section and ':' in stripped:
                key, _, val = stripped.partition(':')
                key = key.strip()
                val = val.strip()
                if key.startswith('- '):
                    # List item
                    item = key[2:].strip() or val
                    if not isinstance(data[current_section], list):
                        data[current_section] = []
                    data[current_section].append(_parse_value(item))
                elif val and not val.startswith('#'):
                    if isinstance(data[current_section], dict):
                        data[current_section][key] = _parse_value(val)
                else:
                    if isinstance(data[current_section], dict):
                        data[current_section][key] = {}
                        current_subsection = key

    return data


def _parse_value(val: str) -> Any:
    """Parse a YAML value string into Python type."""
    # Remove inline comments
    if '#' in val:
        # Careful not to strip # inside quotes
        if not (val.startswith('"') or val.startswith("'")):
            val = val[:val.index('#')].strip()

    # Strip quotes
    if (val.startswith('"') and val.endswith('"')) or \
       (val.startswith("'") and val.endswith("'")):
        return val[1:-1]

    # Booleans
    if val.lower() in ('true', 'yes', 'on'):
        return True
    if val.lower() in ('false', 'no', 'off'):
        return False

    # Null
    if val.lower() in ('null', 'none', '~', ''):
        return None

    # Numbers
    try:
        if '.' in val:
            return float(val)
        return int(val)
    except ValueError:
        pass

    # List (inline)
    if val.startswith('[') and val.endswith(']'):
        items = val[1:-1].split(',')
        return [_parse_value(i.strip()) for i in items if i.strip()]

    return val


def find_config_file(ebook_dir: Optional[str] = None, explicit_path: Optional[str] = None) -> Optional[str]:
    """Search for config file in standard locations."""
    if explicit_path:
        if os.path.isfile(explicit_path):
            return explicit_path
        print(f"Warning: Config file not found: {explicit_path}", file=sys.stderr)
        return None

    search_paths = [
        os.path.join(os.getcwd(), 'ebook_pipeline.yaml'),
        os.path.join(os.getcwd(), 'ebook_pipeline.yml'),
    ]

    if ebook_dir:
        search_paths.extend([
            os.path.join(ebook_dir, 'ebook_pipeline.yaml'),
            os.path.join(ebook_dir, 'ebook_pipeline.yml'),
        ])

    search_paths.extend([
        os.path.expanduser('~/.config/ebook-pipeline/config.yaml'),
        os.path.expanduser('~/.config/ebook-pipeline/config.yml'),
        '/etc/ebook-pipeline/config.yaml',
    ])

    for path in search_paths:
        if os.path.isfile(path):
            return path

    return None


def get_rate_limit_config(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract rate limit settings from config, with defaults."""
    defaults = {
        'openlibrary': {
            'delay': 0.5,
            'max_retries': 3,
            'backoff_base': 2.0,
            'circuit_breaker': 5,
            'daily_limit': None,
        },
        'google_books': {
            'delay': 1.5,
            'max_retries': 3,
            'backoff_base': 2.0,
            'circuit_breaker': 5,
            'daily_limit': None,
        },
        'anthropic': {
            'delay': 1.0,
            'max_retries': 2,
            'backoff_base': 3.0,
            'circuit_breaker': 3,
            'daily_limit': None,
            'max_tokens_per_run': None,
        },
        'global': {
            'files_per_minute': None,
            'files_per_hour': None,
            'pause_between_files': 0,
        },
    }

    rate_limits = config.get('rate_limits', {})
    if not isinstance(rate_limits, dict):
        return defaults

    for api, api_defaults in defaults.items():
        user_settings = rate_limits.get(api, {})
        if isinstance(user_settings, dict):
            for key, default_val in api_defaults.items():
                if key not in user_settings or user_settings[key] is None:
                    user_settings[key] = default_val
            defaults[api] = user_settings

    return defaults


class PipelineConfig:
    """Merged configuration from config file, environment, and CLI args."""

    def __init__(self, args: argparse.Namespace, config_data: Dict[str, Any], config_path: Optional[str]):
        self._args = args
        self._config = config_data
        self.config_path = config_path

        # Track which CLI args were explicitly set (not defaults)
        self._explicit_cli = set()
        if hasattr(args, '_explicit'):
            self._explicit_cli = args._explicit

    def _get(self, cli_name: str, config_key: str = None, env_var: str = None,
             default: Any = None, type_fn=None) -> Any:
        """Get config value with priority: explicit CLI > env > config > default."""
        config_key = config_key or cli_name

        # 1. Explicit CLI flag
        cli_val = getattr(self._args, cli_name, None)
        if cli_name in self._explicit_cli and cli_val is not None:
            return type_fn(cli_val) if type_fn and cli_val is not None else cli_val

        # 2. Environment variable
        if env_var:
            env_val = os.environ.get(env_var)
            if env_val is not None:
                return type_fn(env_val) if type_fn else env_val

        # 3. Config file
        # Support dotted keys like "rate_limits.global.files_per_minute"
        config_val = self._config
        for part in config_key.split('.'):
            if isinstance(config_val, dict):
                config_val = config_val.get(part)
            else:
                config_val = None
                break

        if config_val is not None:
            # Expand ~ in paths
            if isinstance(config_val, str) and ('~' in config_val or config_val.startswith('/')):
                config_val = os.path.expanduser(config_val)
            return type_fn(config_val) if type_fn and config_val is not None else config_val

        # 4. Non-explicit CLI value (argparse default)
        if cli_val is not None:
            return cli_val

        # 5. Hard default
        return default

    # --- Core directories ---
    @property
    def ebook_dir(self) -> str:
        return self._get('ebook_dir', default='.')

    @property
    def rdf_catalog(self) -> Optional[str]:
        return self._get('rdf_catalog', env_var='RDF_CATALOG')

    @property
    def log_dir(self) -> Optional[str]:
        return self._get('log_dir')

    # --- API keys ---
    @property
    def anthropic_api_key(self) -> Optional[str]:
        return self._get('anthropic_api_key', env_var='ANTHROPIC_API_KEY')

    @property
    def google_api_key(self) -> Optional[str]:
        # Support single key or rotation list
        single = self._get('google_api_key', env_var='GOOGLE_API_KEY')
        if single:
            return single
        keys = self._config.get('google_api_keys', [])
        if keys and isinstance(keys, list):
            return keys  # Return list for rotation
        return None

    # --- Stage controls ---
    @property
    def skip_rdf(self) -> bool:
        return self._get('skip_rdf', default=False)

    @property
    def skip_api(self) -> bool:
        return self._get('skip_api', default=False)

    @property
    def skip_ai(self) -> bool:
        return self._get('skip_ai', default=False)

    @property
    def skip_write(self) -> bool:
        return self._get('skip_write', default=False)

    @property
    def skip_rename(self) -> bool:
        return self._get('skip_rename', default=False)

    @property
    def auto_download_rdf(self) -> bool:
        return self._get('auto_download_rdf', default=False)

    # --- Thresholds ---
    @property
    def api_threshold(self) -> float:
        return self._get('api_threshold', default=0.7, type_fn=float)

    @property
    def ai_threshold(self) -> float:
        return self._get('ai_threshold', default=0.4, type_fn=float)

    # --- Processing ---
    @property
    def limit(self) -> Optional[int]:
        return self._get('limit', type_fn=int)

    @property
    def threads(self) -> int:
        return self._get('threads', default=1, type_fn=int)

    @property
    def dry_run(self) -> bool:
        return self._get('dry_run', default=False)

    @property
    def verbose(self) -> bool:
        return self._get('verbose', default=False)

    @property
    def force(self) -> bool:
        return self._get('force', default=False)

    # --- Rate limits ---
    @property
    def rate_limits(self) -> Dict[str, Dict[str, Any]]:
        return get_rate_limit_config(self._config)

    # --- Schedule ---
    @property
    def schedule(self) -> Dict[str, Any]:
        return self._config.get('schedule', {})

    # --- File filters ---
    @property
    def file_filters(self) -> Dict[str, Any]:
        return self._config.get('file_filters', {})

    # --- Cache ---
    @property
    def cache_config(self) -> Dict[str, Any]:
        return self._config.get('cache', {'enabled': True, 'ttl_days': 30})

    def to_dict(self) -> Dict[str, Any]:
        """Dump resolved config as dict (for logging)."""
        return {
            'ebook_dir': self.ebook_dir,
            'rdf_catalog': self.rdf_catalog,
            'log_dir': self.log_dir,
            'skip_rdf': self.skip_rdf,
            'skip_api': self.skip_api,
            'skip_ai': self.skip_ai,
            'skip_write': self.skip_write,
            'skip_rename': self.skip_rename,
            'api_threshold': self.api_threshold,
            'ai_threshold': self.ai_threshold,
            'limit': self.limit,
            'dry_run': self.dry_run,
            'verbose': self.verbose,
            'rate_limits': self.rate_limits,
            'config_path': self.config_path,
        }

    def __repr__(self):
        return f"PipelineConfig({self.config_path or 'defaults'})"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Comprehensive ebook metadata extraction, enrichment, and renaming pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Basic run with config file
  python ebook_metadata_pipeline.py /data/ebooks --config ebook_pipeline.yaml

  # CLI flags override config file
  python ebook_metadata_pipeline.py /data/ebooks --limit 10 --dry-run

  # Generate default config file
  python ebook_metadata_pipeline.py --generate-config

  # Install systemd service
  python ebook_metadata_pipeline.py --install-service
        """,
    )

    parser.add_argument('ebook_dir', nargs='?', help='Directory containing ebook files')
    parser.add_argument('--config', '-c', help='Path to config file (YAML)')
    parser.add_argument('--generate-config', action='store_true',
                        help='Generate default config file and exit')
    parser.add_argument('--install-service', action='store_true',
                        help='Install systemd service and timer')

    # Source controls
    source = parser.add_argument_group('source controls')
    source.add_argument('--rdf-catalog', help='Path to Gutenberg RDF catalog directory')
    source.add_argument('--skip-rdf', action='store_true', default=None,
                        help='Skip Gutenberg RDF catalog entirely')
    source.add_argument('--skip-api', action='store_true', default=None,
                        help='Skip public API lookups')
    source.add_argument('--skip-ai', action='store_true', default=None,
                        help='Skip AI text analysis')
    source.add_argument('--auto-download-rdf', action='store_true', default=None,
                        help='Auto-download RDF catalog if missing')

    # Thresholds
    thresh = parser.add_argument_group('thresholds')
    thresh.add_argument('--api-threshold', type=float,
                        help='Skip API if completeness >= threshold (default: 0.7)')
    thresh.add_argument('--ai-threshold', type=float,
                        help='Skip AI if completeness >= threshold (default: 0.4)')

    # API keys
    keys = parser.add_argument_group('API keys')
    keys.add_argument('--anthropic-api-key', help='Anthropic API key (or ANTHROPIC_API_KEY env)')
    keys.add_argument('--google-api-key', help='Google Books API key (or GOOGLE_API_KEY env)')

    # Rate limiting
    rate = parser.add_argument_group('rate limiting')
    rate.add_argument('--rate-delay', type=float,
                      help='Global delay between API calls (seconds)')
    rate.add_argument('--files-per-minute', type=int,
                      help='Max files to process per minute')
    rate.add_argument('--files-per-hour', type=int,
                      help='Max files to process per hour')

    # Output controls
    output = parser.add_argument_group('output controls')
    output.add_argument('--dry-run', action='store_true', default=None,
                        help='Preview changes without modifying files')
    output.add_argument('--skip-write', action='store_true', default=None,
                        help='Skip metadata writing to files')
    output.add_argument('--skip-rename', action='store_true', default=None,
                        help='Skip file renaming')

    # Processing
    proc = parser.add_argument_group('processing')
    proc.add_argument('--limit', type=int, help='Process only first N files')
    proc.add_argument('--threads', type=int, help='Concurrent processing threads')
    proc.add_argument('--force', action='store_true', default=None,
                      help='Reprocess files even if cached')
    proc.add_argument('--verbose', '-v', action='store_true', default=None,
                      help='Verbose debug output')
    proc.add_argument('--log-dir', help='Directory for logs')

    # Cache
    cache = parser.add_argument_group('cache')
    cache.add_argument('--cache-stats', action='store_true',
                       help='Show cache statistics and exit')
    cache.add_argument('--cache-clear', action='store_true',
                       help='Clear processing cache')

    return parser


def parse_args_and_config() -> PipelineConfig:
    """Parse CLI args and merge with config file."""
    parser = build_arg_parser()
    args = parser.parse_args()

    # Track which args were explicitly provided
    explicit = set()
    for action in parser._actions:
        if action.dest == 'help':
            continue
        if action.dest in sys.argv or any(
            opt in sys.argv for opt in action.option_strings
        ):
            explicit.add(action.dest)

    # More reliable: check which args differ from defaults
    defaults = parser.parse_args([])
    for key, val in vars(args).items():
        default_val = getattr(defaults, key, None)
        if val is not None and val != default_val:
            explicit.add(key)

    args._explicit = explicit

    # Find and load config
    config_path = find_config_file(
        ebook_dir=args.ebook_dir,
        explicit_path=args.config
    )

    config_data = {}
    if config_path:
        try:
            config_data = load_yaml(config_path)
            if args.verbose:
                print(f"Loaded config: {config_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Failed to load config {config_path}: {e}", file=sys.stderr)

    return PipelineConfig(args, config_data, config_path)


# ============================================================================
# Global Rate Limiter
# ============================================================================

class RateLimiter:
    """
    Configurable rate limiter with per-API delays, circuit breakers,
    exponential backoff, and global file-level throttling.
    """

    def __init__(self, config: Dict[str, Dict[str, Any]], logger=None):
        import time as _time
        self._time = _time
        self.log = logger
        self.config = config

        # Per-API state
        self._last_call = {}          # api -> timestamp
        self._backoff_level = {}      # api -> int
        self._consecutive_429s = {}   # api -> int
        self._disabled = {}           # api -> bool
        self._daily_counts = {}       # api -> int
        self._daily_reset = {}        # api -> timestamp

        # Global file-level state
        self._files_this_minute = 0
        self._files_this_hour = 0
        self._minute_start = _time.time()
        self._hour_start = _time.time()

        # Initialize per-API state
        for api in ['openlibrary', 'google_books', 'anthropic']:
            self._last_call[api] = 0.0
            self._backoff_level[api] = 0
            self._consecutive_429s[api] = 0
            self._disabled[api] = False
            self._daily_counts[api] = 0
            self._daily_reset[api] = _time.time()

    def _get_api_config(self, api: str) -> Dict[str, Any]:
        """Get config for a specific API."""
        return self.config.get(api, {
            'delay': 1.0,
            'max_retries': 3,
            'backoff_base': 2.0,
            'circuit_breaker': 5,
            'daily_limit': None,
        })

    def is_disabled(self, api: str) -> bool:
        """Check if an API is disabled by circuit breaker."""
        return self._disabled.get(api, False)

    def check_daily_limit(self, api: str) -> bool:
        """Check if daily limit has been reached. Returns True if OK to proceed."""
        cfg = self._get_api_config(api)
        limit = cfg.get('daily_limit')
        if limit is None:
            return True

        now = self._time.time()
        # Reset daily counter every 24h
        if now - self._daily_reset.get(api, 0) > 86400:
            self._daily_counts[api] = 0
            self._daily_reset[api] = now

        if self._daily_counts.get(api, 0) >= limit:
            if self.log:
                self.log.warning(f"Daily limit reached for {api} ({limit} requests)")
            return False

        return True

    def wait(self, api: str):
        """Wait for the appropriate delay before making an API call."""
        if self.is_disabled(api):
            return

        cfg = self._get_api_config(api)
        base_delay = cfg.get('delay', 1.0)
        backoff_base = cfg.get('backoff_base', 2.0)
        backoff = self._backoff_level.get(api, 0)

        total_delay = base_delay + (backoff_base ** backoff - 1 if backoff else 0)

        now = self._time.time()
        elapsed = now - self._last_call.get(api, 0)

        if elapsed < total_delay:
            wait_time = total_delay - elapsed
            if self.log:
                self.log.debug(f"Rate limit: waiting {wait_time:.1f}s for {api}")
            self._time.sleep(wait_time)

        self._last_call[api] = self._time.time()
        self._daily_counts[api] = self._daily_counts.get(api, 0) + 1

    def record_success(self, api: str):
        """Record a successful API call - reduce backoff."""
        self._consecutive_429s[api] = 0
        self._backoff_level[api] = max(0, self._backoff_level.get(api, 0) - 1)

    def record_429(self, api: str) -> bool:
        """Record a 429 response. Returns True if should retry, False if circuit broken."""
        cfg = self._get_api_config(api)
        self._consecutive_429s[api] = self._consecutive_429s.get(api, 0) + 1
        self._backoff_level[api] = min(
            self._backoff_level.get(api, 0) + 1,
            4  # max backoff level
        )

        threshold = cfg.get('circuit_breaker', 5)
        if self._consecutive_429s[api] >= threshold:
            self._disabled[api] = True
            if self.log:
                self.log.warning(
                    f"⚠ Circuit breaker OPEN for {api} after "
                    f"{self._consecutive_429s[api]} consecutive 429s"
                )
            return False

        return True

    def wait_between_files(self):
        """Apply global file-level rate limiting."""
        global_cfg = self.config.get('global', {})

        # Fixed pause between files
        pause = global_cfg.get('pause_between_files', 0)
        if pause and pause > 0:
            self._time.sleep(pause)

        # Files per minute limit
        fpm = global_cfg.get('files_per_minute')
        if fpm:
            now = self._time.time()
            if now - self._minute_start >= 60:
                self._files_this_minute = 0
                self._minute_start = now

            self._files_this_minute += 1
            if self._files_this_minute >= fpm:
                wait = 60 - (now - self._minute_start)
                if wait > 0:
                    if self.log:
                        self.log.info(f"File rate limit: {fpm}/min reached, waiting {wait:.0f}s")
                    self._time.sleep(wait)
                    self._files_this_minute = 0
                    self._minute_start = self._time.time()

        # Files per hour limit
        fph = global_cfg.get('files_per_hour')
        if fph:
            now = self._time.time()
            if now - self._hour_start >= 3600:
                self._files_this_hour = 0
                self._hour_start = now

            self._files_this_hour += 1
            if self._files_this_hour >= fph:
                wait = 3600 - (now - self._hour_start)
                if wait > 0:
                    if self.log:
                        self.log.info(f"File rate limit: {fph}/hr reached, waiting {wait:.0f}s")
                    self._time.sleep(wait)
                    self._files_this_hour = 0
                    self._hour_start = self._time.time()

    def get_retry_wait(self, api: str, attempt: int) -> float:
        """Calculate wait time for a retry attempt."""
        cfg = self._get_api_config(api)
        backoff_base = cfg.get('backoff_base', 2.0)
        return backoff_base ** (attempt + 1)

    def get_max_retries(self, api: str) -> int:
        """Get max retries for an API."""
        cfg = self._get_api_config(api)
        return cfg.get('max_retries', 3)

    def status(self) -> Dict[str, Any]:
        """Get current rate limiter status."""
        return {
            api: {
                'disabled': self._disabled.get(api, False),
                'backoff_level': self._backoff_level.get(api, 0),
                'consecutive_429s': self._consecutive_429s.get(api, 0),
                'daily_requests': self._daily_counts.get(api, 0),
            }
            for api in ['openlibrary', 'google_books', 'anthropic']
        }

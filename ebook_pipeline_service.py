#!/usr/bin/env python3
"""
Ebook Pipeline - Service Installer & Runner
=============================================
Generates and installs systemd service + timer units,
or generates crontab entries for automated pipeline runs.

Usage:
  python ebook_pipeline_service.py install   [--user] [--config PATH]
  python ebook_pipeline_service.py uninstall [--user]
  python ebook_pipeline_service.py status
  python ebook_pipeline_service.py run       [--config PATH]  # one-shot run
  python ebook_pipeline_service.py cron      [--config PATH]  # print crontab line
"""

import argparse
import os
import pwd
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path


# ============================================================================
# Defaults
# ============================================================================

SERVICE_NAME = "ebook-pipeline"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_SCRIPT = os.path.join(SCRIPT_DIR, "ebook_metadata_pipeline.py")
CONFIG_SEARCH_PATHS = [
    os.path.join(SCRIPT_DIR, "ebook_pipeline.yaml"),
    os.path.expanduser("~/.config/ebook-pipeline/config.yaml"),
    "/etc/ebook-pipeline/config.yaml",
]


def find_python() -> str:
    """Find the Python interpreter to use."""
    # Check for venv
    venv_python = os.path.expanduser("~/.venvs/ebook-pipeline/bin/python3")
    if os.path.isfile(venv_python):
        return venv_python
    # System python
    return shutil.which("python3") or sys.executable


def find_config(explicit: str = None) -> str:
    """Find the config file."""
    if explicit and os.path.isfile(explicit):
        return os.path.abspath(explicit)
    for path in CONFIG_SEARCH_PATHS:
        if os.path.isfile(path):
            return path
    return ""


def load_schedule_from_config(config_path: str) -> dict:
    """Load schedule settings from config YAML."""
    defaults = {
        'calendar': '*-*-* 02:00:00',
        'max_runtime_minutes': 240,
        'randomized_delay_sec': 300,
    }
    if not config_path or not os.path.isfile(config_path):
        return defaults

    try:
        import yaml
        with open(config_path) as f:
            data = yaml.safe_load(f) or {}
        schedule = data.get('schedule', {})
        if isinstance(schedule, dict):
            defaults.update({k: v for k, v in schedule.items() if v is not None})
    except ImportError:
        # Basic extraction without PyYAML
        pass

    return defaults


# ============================================================================
# Systemd Unit Generation
# ============================================================================

def generate_service_unit(python_path: str, config_path: str,
                          schedule: dict, user_mode: bool) -> str:
    """Generate the .service unit file content."""
    max_runtime = schedule.get('max_runtime_minutes', 240)

    wrapper = os.path.join(SCRIPT_DIR, "ebook_pipeline_run.sh")

    unit = textwrap.dedent(f"""\
    [Unit]
    Description=Ebook Metadata Pipeline
    Documentation=https://github.com/yourusername/ebook-metadata-pipeline
    After=network-online.target
    Wants=network-online.target

    [Service]
    Type=oneshot
    ExecStart={wrapper}
    TimeoutStartSec={max_runtime * 60}
    TimeoutStopSec=60

    # Resource limits
    Nice=15
    IOSchedulingClass=idle
    MemoryMax=2G

    # Logging
    StandardOutput=journal
    StandardError=journal
    SyslogIdentifier={SERVICE_NAME}

    # Security hardening
    NoNewPrivileges=true
    ProtectSystem=strict
    ProtectHome=read-only
    PrivateTmp=true
    """)

    if not user_mode:
        user = pwd.getpwuid(os.getuid()).pw_name
        unit += f"User={user}\n"
        unit += f"Group={user}\n"

    # ReadWritePaths for the ebook directory and logs
    unit += textwrap.dedent("""\

    # Allow writing to ebook dirs and logs - update these paths!
    # ReadWritePaths=/mnt/data/ebooks
    # ReadWritePaths=/var/log/ebook-pipeline

    [Install]
    WantedBy=multi-user.target
    """)

    return unit


def generate_timer_unit(schedule: dict) -> str:
    """Generate the .timer unit file content."""
    calendar = schedule.get('calendar', '*-*-* 02:00:00')
    delay = schedule.get('randomized_delay_sec', 300)

    return textwrap.dedent(f"""\
    [Unit]
    Description=Ebook Metadata Pipeline Timer
    Documentation=https://github.com/yourusername/ebook-metadata-pipeline

    [Timer]
    OnCalendar={calendar}
    RandomizedDelaySec={delay}
    Persistent=true
    Unit={SERVICE_NAME}.service

    [Install]
    WantedBy=timers.target
    """)


def generate_wrapper_script(python_path: str, config_path: str) -> str:
    """Generate the shell wrapper that the service calls."""
    return textwrap.dedent(f"""\
    #!/usr/bin/env bash
    # ==========================================================================
    # Ebook Metadata Pipeline - Service Wrapper
    # ==========================================================================
    # Called by systemd timer or cron. Handles:
    #   - Lock file (prevent concurrent runs)
    #   - Environment setup
    #   - Config file loading
    #   - Logging
    #   - Runtime limit enforcement
    # ==========================================================================

    set -euo pipefail

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------
    PYTHON="{python_path}"
    PIPELINE="{PIPELINE_SCRIPT}"
    CONFIG="{config_path}"
    LOCK_FILE="/tmp/{SERVICE_NAME}.lock"
    LOG_TAG="{SERVICE_NAME}"

    # ---------------------------------------------------------------------------
    # Lock file - prevent concurrent runs
    # ---------------------------------------------------------------------------
    if [ -f "$LOCK_FILE" ]; then
        LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null || echo "")
        if [ -n "$LOCK_PID" ] && kill -0 "$LOCK_PID" 2>/dev/null; then
            echo "[$LOG_TAG] Another instance is running (PID: $LOCK_PID). Exiting." >&2
            exit 0
        else
            echo "[$LOG_TAG] Stale lock file found (PID: $LOCK_PID). Removing." >&2
            rm -f "$LOCK_FILE"
        fi
    fi

    # Create lock
    echo $$ > "$LOCK_FILE"
    trap 'rm -f "$LOCK_FILE"' EXIT INT TERM

    # ---------------------------------------------------------------------------
    # Environment
    # ---------------------------------------------------------------------------
    # Load environment file if present (for API keys)
    ENV_FILE="$HOME/.config/ebook-pipeline/env"
    if [ -f "$ENV_FILE" ]; then
        set -a
        source "$ENV_FILE"
        set +a
    fi

    # Also check /etc location
    if [ -f "/etc/ebook-pipeline/env" ]; then
        set -a
        source "/etc/ebook-pipeline/env"
        set +a
    fi

    # ---------------------------------------------------------------------------
    # Run pipeline
    # ---------------------------------------------------------------------------
    echo "[$LOG_TAG] Starting pipeline run at $(date -Iseconds)"

    CMD=("$PYTHON" "$PIPELINE")

    # Add config if it exists
    if [ -n "$CONFIG" ] && [ -f "$CONFIG" ]; then
        CMD+=("--config" "$CONFIG")
        echo "[$LOG_TAG] Using config: $CONFIG"
    fi

    # Execute
    "${{CMD[@]}}" 2>&1

    EXIT_CODE=$?

    echo "[$LOG_TAG] Pipeline finished at $(date -Iseconds) with exit code $EXIT_CODE"
    exit $EXIT_CODE
    """)


def generate_env_file() -> str:
    """Generate template environment file for API keys."""
    return textwrap.dedent("""\
    # Ebook Pipeline - Environment Variables
    # Place at ~/.config/ebook-pipeline/env or /etc/ebook-pipeline/env
    # These are loaded by the service wrapper before running the pipeline.

    # Anthropic API key for AI metadata extraction (Stage 4)
    # ANTHROPIC_API_KEY=sk-ant-...

    # Google Books API key (optional, improves rate limits)
    # GOOGLE_API_KEY=AIza...

    # Gutenberg RDF catalog location (optional override)
    # RDF_CATALOG=~/gutenberg-rdf
    """)


def generate_crontab_line(python_path: str, config_path: str, schedule: dict) -> str:
    """Generate a crontab entry as alternative to systemd."""
    wrapper = os.path.join(SCRIPT_DIR, "ebook_pipeline_run.sh")
    calendar = schedule.get('calendar', '*-*-* 02:00:00')

    # Convert systemd calendar to cron format (approximate)
    # Default: daily at 2 AM
    cron_time = "0 2 * * *"

    # Try to parse some common patterns
    if 'Mon' in calendar:
        cron_time = "0 2 * * 1"
    elif '*:00:00' in calendar and '*-*-*' in calendar:
        # Every hour
        cron_time = "0 * * * *"

    # Extract hour if present
    import re
    hour_match = re.search(r'(\d{1,2}):(\d{2}):(\d{2})', calendar)
    if hour_match:
        hour = hour_match.group(1)
        minute = hour_match.group(2)
        cron_time = f"{minute} {hour} * * *"
        if 'Mon' in calendar:
            cron_time = f"{minute} {hour} * * 1"

    return f"{cron_time} {wrapper} >> /var/log/{SERVICE_NAME}/cron.log 2>&1"


# ============================================================================
# Install / Uninstall
# ============================================================================

def install(args):
    """Install systemd service and timer."""
    python_path = find_python()
    config_path = find_config(args.config)
    schedule = load_schedule_from_config(config_path)

    user_mode = args.user

    if user_mode:
        unit_dir = os.path.expanduser("~/.config/systemd/user")
        systemctl = ["systemctl", "--user"]
    else:
        unit_dir = "/etc/systemd/system"
        systemctl = ["sudo", "systemctl"]

    os.makedirs(unit_dir, exist_ok=True)

    # Generate files
    service_content = generate_service_unit(python_path, config_path, schedule, user_mode)
    timer_content = generate_timer_unit(schedule)
    wrapper_content = generate_wrapper_script(python_path, config_path)
    env_content = generate_env_file()

    # Write wrapper script
    wrapper_path = os.path.join(SCRIPT_DIR, "ebook_pipeline_run.sh")
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)
    print(f"  ✓ Wrapper script: {wrapper_path}")

    # Write env template
    env_dir = os.path.expanduser("~/.config/ebook-pipeline")
    os.makedirs(env_dir, exist_ok=True)
    env_path = os.path.join(env_dir, "env")
    if not os.path.exists(env_path):
        with open(env_path, 'w') as f:
            f.write(env_content)
        os.chmod(env_path, 0o600)
        print(f"  ✓ Environment file: {env_path}")
    else:
        print(f"  · Environment file exists: {env_path}")

    # Write service unit
    service_path = os.path.join(unit_dir, f"{SERVICE_NAME}.service")
    if user_mode:
        with open(service_path, 'w') as f:
            f.write(service_content)
    else:
        proc = subprocess.run(
            ["sudo", "tee", service_path],
            input=service_content.encode(), capture_output=True
        )
        if proc.returncode != 0:
            print(f"  ✗ Failed to write service: {proc.stderr.decode()}")
            return False
    print(f"  ✓ Service unit: {service_path}")

    # Write timer unit
    timer_path = os.path.join(unit_dir, f"{SERVICE_NAME}.timer")
    if user_mode:
        with open(timer_path, 'w') as f:
            f.write(timer_content)
    else:
        proc = subprocess.run(
            ["sudo", "tee", timer_path],
            input=timer_content.encode(), capture_output=True
        )
        if proc.returncode != 0:
            print(f"  ✗ Failed to write timer: {proc.stderr.decode()}")
            return False
    print(f"  ✓ Timer unit: {timer_path}")

    # Reload and enable
    subprocess.run([*systemctl, "daemon-reload"], capture_output=True)
    subprocess.run([*systemctl, "enable", f"{SERVICE_NAME}.timer"], capture_output=True)
    print(f"  ✓ Timer enabled")

    print(f"\n{'='*60}")
    print(f"Installation complete!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"  1. Edit your config: {config_path or 'ebook_pipeline.yaml'}")
    print(f"  2. Add API keys to: {env_path}")
    print(f"  3. Update ReadWritePaths in {service_path}")
    print(f"  4. Start the timer:")
    print(f"       {' '.join(systemctl)} start {SERVICE_NAME}.timer")
    print(f"  5. Or run once manually:")
    print(f"       {' '.join(systemctl)} start {SERVICE_NAME}.service")
    print(f"\nUseful commands:")
    print(f"  {' '.join(systemctl)} status {SERVICE_NAME}.timer")
    print(f"  {' '.join(systemctl)} list-timers")
    print(f"  journalctl {'--user ' if user_mode else ''}-u {SERVICE_NAME} -f")

    return True


def uninstall(args):
    """Remove systemd service and timer."""
    user_mode = args.user

    if user_mode:
        unit_dir = os.path.expanduser("~/.config/systemd/user")
        systemctl = ["systemctl", "--user"]
    else:
        unit_dir = "/etc/systemd/system"
        systemctl = ["sudo", "systemctl"]

    # Stop and disable
    subprocess.run([*systemctl, "stop", f"{SERVICE_NAME}.timer"], capture_output=True)
    subprocess.run([*systemctl, "disable", f"{SERVICE_NAME}.timer"], capture_output=True)
    subprocess.run([*systemctl, "stop", f"{SERVICE_NAME}.service"], capture_output=True)

    # Remove files
    for name in [f"{SERVICE_NAME}.service", f"{SERVICE_NAME}.timer"]:
        path = os.path.join(unit_dir, name)
        if os.path.exists(path):
            if user_mode:
                os.remove(path)
            else:
                subprocess.run(["sudo", "rm", path], capture_output=True)
            print(f"  ✓ Removed: {path}")

    subprocess.run([*systemctl, "daemon-reload"], capture_output=True)
    print("  ✓ Uninstalled")

    # Don't remove wrapper or env file - user might want those
    wrapper = os.path.join(SCRIPT_DIR, "ebook_pipeline_run.sh")
    if os.path.exists(wrapper):
        print(f"\n  Note: Wrapper script still exists: {wrapper}")
        print(f"  Note: Env file still exists: ~/.config/ebook-pipeline/env")


def show_status(args):
    """Show service/timer status."""
    user_mode = args.user
    systemctl = ["systemctl", "--user"] if user_mode else ["systemctl"]

    print(f"Timer status:")
    subprocess.run([*systemctl, "status", f"{SERVICE_NAME}.timer"], check=False)
    print(f"\nService status:")
    subprocess.run([*systemctl, "status", f"{SERVICE_NAME}.service"], check=False)
    print(f"\nRecent logs:")
    journal = ["journalctl"]
    if user_mode:
        journal.append("--user")
    subprocess.run([*journal, "-u", SERVICE_NAME, "-n", "20", "--no-pager"], check=False)


def show_cron(args):
    """Print crontab line for systems without systemd."""
    config_path = find_config(args.config)
    schedule = load_schedule_from_config(config_path)
    python_path = find_python()

    # Generate wrapper first
    wrapper_content = generate_wrapper_script(python_path, config_path)
    wrapper_path = os.path.join(SCRIPT_DIR, "ebook_pipeline_run.sh")
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_content)
    os.chmod(wrapper_path, 0o755)

    cron_line = generate_crontab_line(python_path, config_path, schedule)

    print("Add this to your crontab (crontab -e):")
    print()
    print(f"# Ebook Metadata Pipeline - automated run")
    print(cron_line)
    print()
    print(f"Wrapper script written to: {wrapper_path}")
    print(f"Create log dir: sudo mkdir -p /var/log/{SERVICE_NAME}")


def run_once(args):
    """Run the pipeline once (like the service would)."""
    python_path = find_python()
    config_path = find_config(args.config)

    cmd = [python_path, PIPELINE_SCRIPT]
    if config_path:
        cmd.extend(["--config", config_path])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ebook Pipeline service installer and manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest='command', help='Command')

    # Install
    p_install = sub.add_parser('install', help='Install systemd service + timer')
    p_install.add_argument('--user', action='store_true',
                           help='Install as user service (no sudo)')
    p_install.add_argument('--config', help='Path to config file')

    # Uninstall
    p_uninstall = sub.add_parser('uninstall', help='Remove systemd service + timer')
    p_uninstall.add_argument('--user', action='store_true')

    # Status
    p_status = sub.add_parser('status', help='Show service status')
    p_status.add_argument('--user', action='store_true')

    # Run
    p_run = sub.add_parser('run', help='Run pipeline once')
    p_run.add_argument('--config', help='Path to config file')

    # Cron
    p_cron = sub.add_parser('cron', help='Print crontab entry')
    p_cron.add_argument('--config', help='Path to config file')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    commands = {
        'install': install,
        'uninstall': uninstall,
        'status': show_status,
        'run': run_once,
        'cron': show_cron,
    }

    commands[args.command](args)


if __name__ == '__main__':
    main()

"""Logging helpers for raw API season payloads."""

import json
from datetime import datetime
from pathlib import Path


def log_raw_api_data(season_year, raw_rows, team_name, log_path=None, existing_buffer=None):
    """Write raw API rows for a season to a timestamped JSON log file."""
    if log_path is None:
        safe_name = team_name.lower().replace(' ', '_')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = Path(__file__).parent / 'logs'
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / f'{safe_name}_api_data_{timestamp}.json'
    else:
        log_path = Path(log_path)

    existing = existing_buffer or {}
    existing[str(season_year)] = {
        'fetched_at': datetime.now().isoformat(timespec='seconds'),
        'season_year': season_year,
        'team': team_name,
        'row_count': len(raw_rows),
        'data': raw_rows,
    }
    log_path.write_text(
        json.dumps({'api_log': existing}, indent=2, default=str),
        encoding='utf-8',
    )
    return existing

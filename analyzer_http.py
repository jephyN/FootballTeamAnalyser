"""HTTP/environment helpers for SportsData API access."""

import json
import os
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
from urllib.request import Request, urlopen


def build_api_headers(api_key):
    """Return the standard SportsData.io request headers dict."""
    return {'Accept': 'application/json', 'Ocp-Apim-Subscription-Key': api_key}


def load_env_file(env_path='.env'):
    """Load KEY=VALUE pairs from a .env file into the environment."""
    if not os.path.exists(env_path):
        return
    with open(env_path, 'r', encoding='utf-8') as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def extract_api_key_from_url(url):
    """Strip the 'key' query param from a URL; return (clean_url, key)."""
    if not url:
        return url, None
    parsed = urlparse(url)
    query = parse_qs(parsed.query, keep_blank_values=True)
    values = query.pop('key', None)
    extracted_key = values[0] if values else None
    cleaned_query = urlencode(query, doseq=True)
    cleaned_url = urlunparse((
        parsed.scheme, parsed.netloc, parsed.path,
        parsed.params, cleaned_query, parsed.fragment,
    ))
    return cleaned_url, extracted_key


def fetch_json(url, headers=None, params=None):
    """Fetch and return a parsed JSON payload from an HTTP endpoint."""
    if params:
        separator = '&' if '?' in url else '?'
        url = f"{url}{separator}{urlencode(params)}"
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=30) as response:
        payload = response.read().decode('utf-8', errors='ignore')
    return json.loads(payload)

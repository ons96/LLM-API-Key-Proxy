"""Report the New-API settings the sidecar's startup audit enforces.

Reads the sidecar env file for its admin token and prints only non-secret
fields: RetryTimes, and each audited channel's status, overrides and model
mapping keys. Never prints a channel key.
"""
import json
import sys
import urllib.request
from pathlib import Path

ENV_FILE = Path('/etc/newapi-sidecar/sidecar.env')
BASE = 'http://127.0.0.1:8001'


def env_values(path):
    values = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or '=' not in stripped:
            continue
        key, _, value = stripped.partition('=')
        values[key.strip()] = value.strip()
    return values


def call(path, token, user_id):
    request = urllib.request.Request(
        BASE + path,
        headers={'Authorization': 'Bearer ' + token, 'New-API-User': str(user_id)},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, {}


import urllib.error  # noqa: E402  (kept next to its use for clarity)

values = env_values(ENV_FILE)
token = values.get('NEWAPI_ADMIN_TOKEN', '')
user_id = values.get('NEWAPI_ADMIN_USER_ID', '')

status, body = call('/api/option/', token, user_id)
print('admin_auth_http', status)
if status != 200 or body.get('success') is not True:
    print('admin token is not usable; settings unreadable')
    raise SystemExit(1)

options = {row.get('key'): row.get('value') for row in body.get('data') or []
           if isinstance(row, dict)}
print('RetryTimes', options.get('RetryTimes'))
print('RetryTimesMode', options.get('RetryTimesMode'))

for channel_id in (3, 4, 10, 11):
    status, body = call('/api/channel/%d' % channel_id, token, user_id)
    if status != 200 or not isinstance(body.get('data'), dict):
        print('channel', channel_id, 'http', status)
        continue
    channel = body['data']
    mapping = channel.get('model_mapping')
    try:
        mapped = sorted(json.loads(mapping).values()) if isinstance(mapping, str) else []
    except ValueError:
        mapped = []
    print('channel', channel_id,
          'name', channel.get('name'),
          'status', channel.get('status'),
          'param_override', repr(channel.get('param_override')),
          'header_override', repr(channel.get('header_override')),
          'models', mapped)

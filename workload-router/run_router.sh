#!/bin/sh
set -eu
exec "${PYTHON:-python3}" router_server.py

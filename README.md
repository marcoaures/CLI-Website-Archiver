# CLI Website Archiver

A small Python tool to archive web pages from sitemap URLs by taking full-page screenshots (and optional HTML snapshots).

## Features

- Discovers sitemap URLs from `robots.txt` when a domain is provided.
- Traverses sitemap indexes and URL sets.
- Captures full-page screenshots via Playwright.
- Optional HTML capture and PNG optimization.
- Retry + concurrent workers.
- "Refresh" mode to update one page by URL or by existing PNG filename.

## Requirements

- Python 3.10+
- Playwright Chromium browser

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m playwright install chromium
```

## Configuration

1. Copy the template:

```bash
cp config.example.json config.json
```

2. Edit `config.json` values for your site.

If you want a different config path, set `ARCHIVE_CONFIG`, for example:

```bash
ARCHIVE_CONFIG=./my-config.json python archive.py
```

### Notes on defaults

- `domain` is used to discover sitemap sources from `robots.txt`.
- `sitemap_urls` can be set directly if you want exact control.
- Cookie selectors are site-specific and usually need updates per target site.

## Run

```bash
python archive.py
```

Then choose:

- `1` for full archive flow
- `2` for refresh flow (single URL or PNG filename)

## Output

By default, output is written to:

- `archive_out/<domain>/...png`
- `archive_out/_logs/urls.json`
- `archive_out/_logs/report.json`

## Minimal quality checks

```bash
python -m py_compile archive.py
python -m unittest discover -s tests -p 'test_*.py'
```

## License

This project is licensed under GPL-3.0. See the `LICENSE` file.

## Attribution

If you use or adapt this project, please retain the copyright notice
and license information.

CLI Website Archiver by Marco Aures | Source: https://marcoaures.ch | Licensed under GPL-3.0

#!/usr/bin/env python3
"""Inventory public files in the DSN_1k OSF project without downloading the dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import urlparse

import requests

ROOT = "https://api.osf.io/v2/nodes/t9c2a/files/osfstorage/"


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.json()


def walk(url: str, prefix: str = "") -> list[dict]:
    out: list[dict] = []
    while url:
        payload = get_json(url)
        for item in payload.get("data", []):
            attrs = item.get("attributes", {})
            links = item.get("links", {})
            name = attrs.get("name", item.get("id", "unknown"))
            kind = attrs.get("kind")
            path = f"{prefix}/{name}".lstrip("/")
            row = {
                "id": item.get("id"),
                "name": name,
                "path": path,
                "kind": kind,
                "size": attrs.get("size"),
                "content_type": attrs.get("content_type") or attrs.get("contentType"),
                "download": links.get("download"),
            }
            out.append(row)
            if kind == "folder":
                files_url = item.get("relationships", {}).get("files", {}).get("links", {}).get("related", {}).get("href")
                if not files_url:
                    files_url = links.get("new_folder")
                    if files_url:
                        files_url = files_url.split("?", 1)[0]
                if files_url:
                    out.extend(walk(files_url, path))
        url = payload.get("links", {}).get("next")
    return out


def main() -> None:
    rows = walk(ROOT)
    Path("results").mkdir(exist_ok=True)
    out = Path("results/dsn_osf_inventory.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    total = sum(int(r.get("size") or 0) for r in rows if r.get("kind") == "file")
    print(f"DSN_OSF_ITEMS={len(rows)}")
    print(f"DSN_OSF_BYTES={total}")
    for r in rows:
        if r.get("kind") == "file":
            print(f"FILE\t{r.get('size')}\t{r.get('path')}\t{r.get('download')}")


if __name__ == "__main__":
    main()

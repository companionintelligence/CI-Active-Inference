#!/usr/bin/env python3
"""Ask a package index what it already holds before this release is uploaded.

    python .github/scripts/pypi_preflight.py --index-url https://pypi.org \
        --project ci-active-inference --version 0.1.1 --dist dist [--mode enforce|inform]

Reads the public JSON API only. Decides, and writes `upload` and `project_state` to
$GITHUB_OUTPUT when that is set:

* The project does not exist (404): upload. The first upload creates it, through the
  pending trusted publisher. A pending publisher does not reserve the name, so someone
  else may still register it first; the upload is then refused.
* The project exists and holds no file of this version: upload. A warning if the
  project's metadata does not mention this repository, because it may then belong to
  someone else and the upload will be refused.
* This version exists with the same file names and SHA-256 as dist/: nothing to upload.
* This version holds some of our files, all of them identical: upload the rest. The
  publish job skips the files that are there. That is a run after an upload that
  stopped half way.
* This version holds a file of ours with another SHA-256: an error. A release needs a
  version number that differs from every earlier one, so this one needs a new version
  number. The wheel rebuilds to the same
  SHA-256 when SOURCE_DATE_EPOCH and the setuptools version are the same. The sdist's
  SHA-256 changes on every build, although its contents do not (the tar headers and the
  gzip header carry the build time), so a second run of a finished release lands here.
  Re-run the failed jobs of the first run instead.

--mode inform turns that error into a warning, for a run that uploads nothing anyway.
Standard library only. Nothing the index returns is echoed into a workflow command.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPOSITORY = "github.com/companionintelligence/ci-active-inference"


def fetch(url: str) -> tuple[int, dict | None]:
    """GET a JSON document: (status, parsed body or None). Retries 5xx and network errors."""
    last = "no attempt"
    for attempt in range(1, 4):
        request = urllib.request.Request(url, headers={"Accept": "application/json",
                                                       "User-Agent": "ci-active-inference-release-preflight"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return 404, None
            last = f"HTTP {exc.code}"
            if exc.code < 500:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last = type(exc).__name__
        time.sleep(2 * attempt)
    raise RuntimeError(f"could not read {url}: {last}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finish(outputs: dict[str, str], status: int) -> int:
    """Write the outputs once each (GitHub keeps the last of a repeated name) and return the exit status."""
    target = os.environ.get("GITHUB_OUTPUT")
    if target:
        with open(target, "a", encoding="utf-8") as handle:
            for name, value in outputs.items():
                handle.write(f"{name}={value}\n")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Check a package index before uploading.")
    parser.add_argument("--index-url", required=True, help="for example https://pypi.org")
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--mode", choices=("enforce", "inform"), default="enforce",
                        help="inform: report the error as a warning (a run that uploads nothing)")
    args = parser.parse_args()

    index = args.index_url.rstrip("/")
    local = {path.name: sha256(path) for path in sorted(args.dist.iterdir()) if path.is_file()} \
        if args.dist.is_dir() else {}
    if not local:
        print(f"::error::{args.dist}/ holds no files")
        return 1

    try:
        status, project = fetch(f"{index}/pypi/{args.project}/json")
    except RuntimeError as exc:
        print(f"::error::{exc}")
        return 1

    if status == 404:
        print(f"{args.project} is not on {index}. The first upload creates it, through the pending trusted "
              f"publisher. A pending publisher does not reserve the name: if someone registers it first, "
              f"the upload is refused.")
        return finish({"project_state": "free", "upload": "true"}, 0)

    info = (project or {}).get("info", {})
    urls = [str(info.get("home_page") or "")] + [str(value) for value in (info.get("project_urls") or {}).values()]
    mentions_us = any(REPOSITORY in url.lower() for url in urls)

    try:
        version_status, release = fetch(f"{index}/pypi/{args.project}/{args.version}/json")
    except RuntimeError as exc:
        print(f"::error::{exc}")
        return 1
    remote = {item["filename"]: (item.get("digests") or {}).get("sha256", "")
              for item in ((release or {}).get("urls") or [])} if version_status == 200 else {}

    differing = sorted(name for name in local if name in remote and remote[name] != local[name])
    identical = sorted(name for name in local if remote.get(name) == local[name])
    absent = sorted(name for name in local if name not in remote)
    outputs = {"project_state": "exists", "upload": "true"}

    if differing:
        severity = "::warning::" if args.mode == "inform" else "::error::"
        print(f"{severity}{args.project} {args.version} is already on {index} with a different SHA-256 for "
              f"{', '.join(differing)}. A release needs a version number that differs from every earlier one: "
              f"release a new version. If this is a "
              f"second run of a release that finished, that is expected, because an sdist does not rebuild to "
              f"the same SHA-256: re-run the failed jobs of the first run instead.")
        outputs["upload"] = "false"
        return finish(outputs, 0 if args.mode == "inform" else 1)
    if not absent:
        print(f"{args.project} {args.version} is already on {index} with the same files and SHA-256: nothing to upload.")
        outputs["upload"] = "false"
        return finish(outputs, 0)

    if identical:
        print(f"{args.project} {args.version} is on {index} with {', '.join(identical)} but without "
              f"{', '.join(absent)}: uploading the rest; the publish job skips what is there.")
    else:
        print(f"{args.project} exists on {index} and holds no file of {args.version}: upload.")
        if not mentions_us:
            print(f"::warning::{args.project} already exists on {index} and its metadata does not mention this "
                  f"repository. If it is not yours, the upload will be refused; see 'If the name is taken' in "
                  f"the release checklist.")
    return finish(outputs, 0)


if __name__ == "__main__":
    sys.exit(main())

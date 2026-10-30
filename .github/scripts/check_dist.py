#!/usr/bin/env python3
"""Check the built distributions before anything is installed or uploaded.

    python .github/scripts/check_dist.py --dist dist --project ci-active-inference \
        --version 0.1.1 --commit "$(git rev-parse HEAD)"

Run it from the repository root. It fails unless all of this holds:

* dist/ holds exactly one wheel and one sdist, named for the project and version.
* The wheel holds only .py files under ci_active_inference/ and ci_forage/, plus its
  dist-info. No tests/, no data files, no bytecode. Every .py file and both licence
  files are byte-identical to the file at --commit in git.
* The wheel METADATA has the right name and version, the SPDX licence expression, both
  licence files, Requires-Python, and no `License ::` classifier.
* The sdist holds only the packages, tests/ and the top-level files setuptools is
  expected to add, plus the metadata it generates. Everything else is byte-identical to
  the file at --commit in git, so the build cannot have changed or added a source file.
* The wheel and the sdist agree on name, version and licence expression.

Standard library and git only.
"""

from __future__ import annotations

import argparse
import email.parser
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

PACKAGES = ("ci_active_inference", "ci_forage")
LICENSE_FILES = ("LICENSE", "NOTICE.md")
LICENSE_EXPRESSION = "PolyForm-Noncommercial-1.0.0"

SDIST_TOP_LEVEL = {"LICENSE", "NOTICE.md", "README.md", "MANIFEST.in", "pyproject.toml"}
SDIST_GENERATED = {"PKG-INFO", "setup.cfg"}
EGG_INFO_FILES = {"PKG-INFO", "SOURCES.txt", "dependency_links.txt", "requires.txt", "top_level.txt"}
SDIST_REQUIRED = ("pyproject.toml", "LICENSE", "NOTICE.md", "README.md", "tests/conftest.py",
                  "ci_active_inference/__init__.py", "ci_forage/__init__.py")
WHEEL_GENERATED = {"METADATA", "WHEEL", "RECORD", "top_level.txt"}

errors: list[str] = []


def error(message: str) -> None:
    errors.append(message)
    print(f"::error::{message}", flush=True)


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def git_blob(commit: str, path: str) -> bytes | None:
    result = subprocess.run(["git", "cat-file", "blob", f"{commit}:{path}"], capture_output=True)
    return result.stdout if result.returncode == 0 else None


def same_as_git(commit: str, path: str, content: bytes, where: str) -> None:
    blob = git_blob(commit, path)
    if blob is None:
        error(f"{where}: {path} is not a file at {commit[:12]}")
    elif blob != content:
        error(f"{where}: {path} differs from the file at {commit[:12]}")


def parse_metadata(text: str):
    return email.parser.Parser().parsestr(text)


def check_metadata(meta, where: str, project: str, version: str) -> None:
    if normalise(meta["Name"] or "") != normalise(project):
        error(f"{where}: Name is {meta['Name']!r}, expected {project!r}")
    if meta["Version"] != version:
        error(f"{where}: Version is {meta['Version']!r}, expected {version!r}")
    if meta["License-Expression"] != LICENSE_EXPRESSION:
        error(f"{where}: License-Expression is {meta['License-Expression']!r}, expected {LICENSE_EXPRESSION!r}")
    bad = [c for c in meta.get_all("Classifier", []) if c.startswith("License ::")]
    if bad:
        error(f"{where}: licence classifiers are still declared: {bad}")
    if sorted(meta.get_all("License-File", [])) != sorted(LICENSE_FILES):
        error(f"{where}: License-File is {meta.get_all('License-File', [])}, expected {list(LICENSE_FILES)}")
    if not meta["Requires-Python"]:
        error(f"{where}: Requires-Python is missing")
    if meta["Description-Content-Type"] != "text/markdown":
        error(f"{where}: Description-Content-Type is {meta['Description-Content-Type']!r}")


def check_wheel(path: Path, project: str, version: str, commit: str):
    where = path.name
    before = len(errors)
    dist_info = f"{normalise(project).replace('-', '_')}-{version}.dist-info"
    meta = None
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        for name in names:
            if name.endswith("/"):
                error(f"{where}: directory entry {name}")
                continue
            head, _, tail = name.partition("/")
            if head in PACKAGES and name.endswith(".py"):
                same_as_git(commit, name, wheel.read(name), where)
            elif head == dist_info:
                if tail in WHEEL_GENERATED:
                    continue
                if tail.startswith("licenses/") and tail[len("licenses/"):] in LICENSE_FILES:
                    same_as_git(commit, tail[len("licenses/"):], wheel.read(name), where)
                else:
                    error(f"{where}: unexpected file {name}")
            else:
                error(f"{where}: unexpected file {name} (only .py files under {', '.join(PACKAGES)} belong in the wheel)")
        for package in PACKAGES:
            if f"{package}/__init__.py" not in names:
                error(f"{where}: {package}/__init__.py is missing")
        if f"{dist_info}/METADATA" in names:
            meta = parse_metadata(wheel.read(f"{dist_info}/METADATA").decode())
            check_metadata(meta, where, project, version)
        else:
            error(f"{where}: {dist_info}/METADATA is missing")
    if len(errors) == before:
        print(f"wheel ok: {where}, {len(names)} files")
    return meta


def check_sdist(path: Path, project: str, version: str, commit: str):
    where = path.name
    before = len(errors)
    root = f"{normalise(project).replace('-', '_')}-{version}"
    egg_info = f"{normalise(project).replace('-', '_')}.egg-info"
    meta = None
    seen = set()
    with tarfile.open(path) as sdist:
        for member in sdist.getmembers():
            if member.isdir():
                continue
            if not member.isreg():
                error(f"{where}: {member.name} is not a regular file")
                continue
            prefix, _, relative = member.name.partition("/")
            if prefix != root or not relative or relative.startswith("/") or ".." in relative.split("/"):
                error(f"{where}: {member.name} is not under {root}/")
                continue
            seen.add(relative)
            content = sdist.extractfile(member).read()
            head = relative.partition("/")[0]
            if relative in SDIST_GENERATED:
                if relative == "PKG-INFO":
                    meta = parse_metadata(content.decode())
            elif head == egg_info:
                if relative.partition("/")[2] not in EGG_INFO_FILES:
                    error(f"{where}: unexpected file {relative}")
            elif relative in SDIST_TOP_LEVEL or (head in (*PACKAGES, "tests") and relative.endswith(".py")):
                same_as_git(commit, relative, content, where)
            else:
                error(f"{where}: unexpected file {relative}")
    for required in SDIST_REQUIRED:
        if required not in seen:
            error(f"{where}: {required} is missing")
    if meta is None:
        error(f"{where}: PKG-INFO is missing")
    else:
        check_metadata(meta, where, project, version)
    if len(errors) == before:
        print(f"sdist ok: {where}, {len(seen)} files")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Check the built wheel and sdist.")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--project", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--commit", required=True, help="the commit the distributions were built from")
    args = parser.parse_args()

    resolved = subprocess.run(["git", "rev-parse", "--verify", f"{args.commit}^{{commit}}"],
                              capture_output=True, text=True)
    if resolved.returncode != 0:
        print(f"::error::{args.commit} is not a commit in this repository")
        return 1
    commit = resolved.stdout.strip()

    stem = f"{normalise(args.project).replace('-', '_')}-{args.version}"
    expected = {f"{stem}-py3-none-any.whl", f"{stem}.tar.gz"}
    found = {p.name for p in args.dist.iterdir()} if args.dist.is_dir() else set()
    if found != expected:
        error(f"{args.dist}/ holds {sorted(found)}, expected exactly {sorted(expected)}")
        return 1

    wheel_meta = check_wheel(args.dist / f"{stem}-py3-none-any.whl", args.project, args.version, commit)
    sdist_meta = check_sdist(args.dist / f"{stem}.tar.gz", args.project, args.version, commit)
    if wheel_meta is not None and sdist_meta is not None:
        for field in ("Name", "Version", "License-Expression"):
            if wheel_meta[field] != sdist_meta[field]:
                error(f"wheel and sdist disagree on {field}: {wheel_meta[field]!r} against {sdist_meta[field]!r}")

    if errors:
        print(f"\n{len(errors)} problem(s) in the distributions")
        return 1
    print("distributions ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

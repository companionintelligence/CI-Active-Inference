#!/usr/bin/env python3
"""Smoke test an INSTALLED ci-active-inference, not a source tree.

Run it with the interpreter of the virtual environment the package was installed into
(a wheel, or an sdist that pip built), from a directory that does not hold the source:

    cd "$(mktemp -d)"
    /path/to/venv/bin/python -I /path/to/checkout/.github/scripts/smoke_installed.py \
        --project ci-active-inference --version 0.1.1

What it checks, in order:

1. The installed metadata: name, version, the SPDX licence expression, no licence
   classifier, both licence files present (LICENSE must carry the PolyForm
   `Required Notice:` line), and only the two packages in top_level.txt.
2. That the packages were imported from site-packages and not from the current
   directory, that no top-level `tests` package was installed, and that __version__
   matches the metadata.
3. That every module in REQUIRED_IMPORTS imports. Every other module is imported too;
   the ones that fail are reported as ::warning:: lines and do not fail the run,
   because some modules cannot import with the dependencies the package declares
   (see the comment on REQUIRED_IMPORTS).
4. That the ci_forage demo runs and picks the beach branch, and that ci_forage.api
   serves under uvicorn (the way run_forage_api.sh starts it) and writes its trial
   snapshot to FORAGE_DATA_DIR instead of into site-packages.

There is no console script to run: main.py at the repository root is not part of the
package.

Exit status 0 means every check passed. Standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata as metadata
import importlib.util
import io
import json
import os
import pkgutil
import re
import socket
import subprocess
import sys
import sysconfig
import tempfile
import time
import urllib.request
import warnings
from pathlib import Path

PACKAGES = ("ci_active_inference", "ci_forage")
LICENSE_EXPRESSION = "PolyForm-Noncommercial-1.0.0"
LICENSE_FILES = ("LICENSE", "NOTICE.md")

# Modules that import with only the declared dependencies on every Python the workflows
# test (3.10 to 3.14). Left out on purpose, because they do not:
#   ci_active_inference.agents, .ab_testing  `from pymdp.agent import Agent`; pymdp is
#                                            not a declared dependency
#   ci_active_inference.api                  ImportError: cannot import name 'get_db'
#   ci_active_inference.models               InvalidRequestError on SQLAlchemy 2.0 (a
#                                            column named `metadata`); Python 3.10
#                                            resolves to 2.0, 3.11 and later to 2.1
REQUIRED_IMPORTS = (
    "ci_active_inference",
    "ci_active_inference.database",
    "ci_active_inference.database.dao",
    "ci_active_inference.explanation",
    "ci_active_inference.monitoring",
    "ci_active_inference.planning",
    "ci_active_inference.utils",
    "ci_forage",
    "ci_forage.api",
    "ci_forage.lake",
    "ci_forage.model",
    "ci_forage.trial",
)

failures: list[str] = []

# Loopback only, and never through a proxy that the environment may define.
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def fail(message: str) -> None:
    failures.append(message)
    print(f"::error::{message}", flush=True)


def check(condition: bool, message: str) -> bool:
    if not condition:
        fail(message)
    return bool(condition)


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def check_metadata(project: str, expected_version: str) -> metadata.Distribution:
    dist = metadata.distribution(project)
    meta = dist.metadata
    check(normalise(meta["Name"]) == normalise(project), f"installed Name is {meta['Name']!r}, expected {project!r}")
    check(dist.version == expected_version,
          f"installed version is {dist.version}, expected {expected_version}")
    check(meta.get("License-Expression") == LICENSE_EXPRESSION,
          f"License-Expression is {meta.get('License-Expression')!r}, expected {LICENSE_EXPRESSION!r}")
    licence_classifiers = [c for c in meta.get_all("Classifier", []) if c.startswith("License ::")]
    check(not licence_classifiers, f"licence classifiers are still declared: {licence_classifiers}")
    check(bool(meta.get("Requires-Python")), "Requires-Python is missing")

    files = [str(f) for f in (dist.files or [])]
    for name in LICENSE_FILES:
        found = [f for f in files if f.endswith(f".dist-info/licenses/{name}")]
        check(len(found) == 1, f"{name} is not installed under dist-info/licenses ({found})")
    licence_text = dist.read_text("licenses/LICENSE") or ""
    check("Required Notice:" in licence_text,
          "the installed LICENSE has no 'Required Notice:' line, which PolyForm requires to travel with the software")

    top_level = set((dist.read_text("top_level.txt") or "").split())
    check(top_level == set(PACKAGES), f"top_level.txt lists {sorted(top_level)}, expected {sorted(PACKAGES)}")
    print(f"metadata ok: {meta['Name']} {dist.version}, {meta.get('License-Expression')}, "
          f"Requires-Python {meta.get('Requires-Python')}")
    return dist


def check_location(dist: metadata.Distribution) -> None:
    site = {Path(sysconfig.get_paths()[key]).resolve() for key in ("purelib", "platlib")}
    for package in PACKAGES:
        path = Path(importlib.import_module(package).__file__).resolve()
        check(any(root in path.parents for root in site),
              f"{package} was imported from {path}, which is not under site-packages")
    check(importlib.util.find_spec("tests") is None,
          "a top-level `tests` package is installed (the v0.1.0 wheel did that)")
    version = importlib.import_module("ci_active_inference").__version__
    check(version == dist.version,
          f"ci_active_inference.__version__ is {version!r} but the metadata says {dist.version}")
    print(f"location ok: {PACKAGES[0]} {version} imported from site-packages")


def import_quietly(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        importlib.import_module(name)


def check_imports() -> None:
    for name in REQUIRED_IMPORTS:
        try:
            import_quietly(name)
        except BaseException as exc:  # noqa: BLE001 (any import-time failure counts)
            fail(f"required module {name} does not import: {type(exc).__name__}: {exc}".splitlines()[0])

    others: dict[str, str] = {}
    for top in PACKAGES:
        package = importlib.import_module(top)
        for info in pkgutil.walk_packages(package.__path__, top + ".", onerror=lambda _name: None):
            if info.name in REQUIRED_IMPORTS:
                continue
            try:
                import_quietly(info.name)
            except BaseException as exc:  # noqa: BLE001
                first = (str(exc).splitlines() or [""])[0][:100]
                others[info.name] = f"{type(exc).__name__}: {first}"
    for name, reason in sorted(others.items()):
        print(f"::warning::{name} does not import with the declared dependencies ({reason})")
    print(f"imports ok: {len(REQUIRED_IMPORTS)} required modules; {len(others)} other module(s) failed to import")


def check_demo() -> None:
    from ci_forage import demo_sam_saturday

    sink = io.StringIO()
    with contextlib.redirect_stdout(sink):
        result = demo_sam_saturday.run()
    pick = result["pick"]
    check(pick["branch"] == "beach", f"the demo picked {pick['branch']!r}, expected 'beach'")
    check("question: where is Sam on Saturday" in sink.getvalue(), "the demo did not print its question")
    print(f"demo ok: picked branch={pick['branch']} texture={pick['texture']}")


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"content-type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with opener.open(req, timeout=10) as response:
        return json.loads(response.read())


def check_api() -> None:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryDirectory() as data_dir:
        env = {**os.environ, "FORAGE_DATA_DIR": data_dir}
        log = Path(data_dir) / "uvicorn.log"
        with log.open("w") as sink:
            server = subprocess.Popen(
                [sys.executable, "-I", "-m", "uvicorn", "ci_forage.api:app",
                 "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                env=env, stdout=sink, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            health = None
            while time.monotonic() < deadline and server.poll() is None:
                try:
                    health = request("GET", f"{base}/health")
                    break
                except OSError:
                    time.sleep(0.25)
            if not check(health is not None and health.get("ok") is True,
                         f"ci_forage.api did not answer /health within 30 s (exit status {server.poll()})"):
                print(log.read_text())
                return
            started = request("POST", f"{base}/ask", {"seed": 0})
            check(started.get("question") == "where is Sam on Saturday", "POST /ask returned another question")
            ids = sorted(row["id"] for row in request("GET", f"{base}/branches")["branches"])
            check(ids == ["beach", "errands", "office"], f"GET /branches lists {ids}")
            picked = request("POST", f"{base}/sample")["pick"]["branch"]
            check(picked == "beach", f"POST /sample picked {picked!r}, expected 'beach'")
            written = sorted(path.name for path in Path(data_dir).glob("*.json"))
            check(len(written) == 1, f"expected one trial snapshot in FORAGE_DATA_DIR, found {written}")
            print(f"api ok: /health, /ask, /branches and /sample answered on port {port}; snapshot written to FORAGE_DATA_DIR")
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test an installed ci-active-inference.")
    parser.add_argument("--project", required=True, help="the distribution name, as installed")
    parser.add_argument("--version", required=True, help="the version the installed distribution must have")
    args = parser.parse_args()

    print(f"python {sys.version.split()[0]} at {sys.executable}")
    dist = check_metadata(args.project, args.version)
    check_location(dist)
    check_imports()
    check_demo()
    check_api()
    if failures:
        print(f"\n{len(failures)} check(s) failed:")
        for message in failures:
            print(f"  - {message}")
        return 1
    print("\nsmoke test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

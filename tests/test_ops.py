"""Automated operational tooling and container specification test suite.

Validates Dockerfile, .dockerignore, docker-compose.yml, and Makefile
programmatically without requiring an active Docker daemon.
"""

from __future__ import annotations

import re
from pathlib import Path
import yaml
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_dockerfile_multi_stage_structure():
    """Validates multi-stage Dockerfile architecture, security constraints, and uv integration."""
    dockerfile_path = REPO_ROOT / "Dockerfile"
    assert dockerfile_path.exists(), f"Dockerfile not found at {dockerfile_path}"

    content = dockerfile_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Stage 1: Builder checks
    assert any(
        re.match(r"^FROM\s+python:3\.11-slim-bookworm\s+AS\s+builder\s*$", line)
        for line in lines
    ), "Dockerfile must define builder stage: FROM python:3.11-slim-bookworm AS builder"

    assert (
        "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/" in content
    ), "Dockerfile must copy official uv binary from ghcr.io/astral-sh/uv:latest"

    assert (
        "UV_COMPILE_BYTECODE=1" in content
    ), "Dockerfile must enable bytecode compilation with UV_COMPILE_BYTECODE=1"

    assert (
        "uv sync --frozen --no-install-project --no-dev" in content
    ), "Dockerfile builder must execute locked production uv sync"

    # Stage 2: Runtime checks
    assert any(
        re.match(r"^FROM\s+python:3\.11-slim-bookworm\s+AS\s+runtime\s*$", line)
        for line in lines
    ), "Dockerfile must define runtime stage: FROM python:3.11-slim-bookworm AS runtime"

    # Non-root user creation and execution
    assert (
        "groupadd -g 1000 appgroup" in content
    ), "Dockerfile must create appgroup with GID 1000"
    assert (
        "useradd -u 1000 -g appgroup -s /bin/bash -m appuser" in content
    ), "Dockerfile must create appuser with UID 1000"
    assert (
        "USER appuser" in content
    ), "Dockerfile must switch to non-root execution with USER appuser"

    # Virtual environment copied from builder
    assert (
        "--from=builder" in content and "/app/.venv" in content
    ), "Dockerfile must copy /app/.venv from builder stage"

    # Port exposure
    assert any(
        re.match(r"^EXPOSE\s+.*8000.*8501", line) for line in lines
    ), "Dockerfile must expose ports 8000 and 8501"


def test_dockerignore_rules():
    """Validates that .dockerignore excludes transient, cached, and sensitive files."""
    dockerignore_path = REPO_ROOT / ".dockerignore"
    assert (
        dockerignore_path.exists()
    ), f".dockerignore not found at {dockerignore_path}"

    content = dockerignore_path.read_text(encoding="utf-8")
    rules = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.startswith("#")
    }

    # Core required exclusions
    required_exclusions = [
        ".venv",
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".coverage",
        ".gsd",
    ]

    for req in required_exclusions:
        assert any(
            req in rule for rule in rules
        ), f".dockerignore must exclude '{req}', found rules: {rules}"


def test_docker_compose_validity():
    """Validates docker-compose.yml structure, service definitions, ports, and healthchecks."""
    compose_path = REPO_ROOT / "docker-compose.yml"
    assert compose_path.exists(), f"docker-compose.yml not found at {compose_path}"

    content = compose_path.read_text(encoding="utf-8")
    compose_data = yaml.safe_load(content)

    assert "services" in compose_data, "docker-compose.yml must define 'services'"
    services = compose_data["services"]

    # Validate API service
    assert "api" in services, "Service 'api' must be defined in docker-compose.yml"
    api = services["api"]
    assert "ports" in api, "API service must define ports"
    assert any(
        "8000:8000" in str(p) for p in api["ports"]
    ), "API service must map port 8000:8000"
    assert "healthcheck" in api, "API service must specify a healthcheck"
    healthcheck = api["healthcheck"]
    assert "test" in healthcheck, "API healthcheck must have a test command"
    test_cmd = (
        " ".join(healthcheck["test"])
        if isinstance(healthcheck["test"], list)
        else str(healthcheck["test"])
    )
    assert (
        "/health" in test_cmd and "curl" in test_cmd
    ), "API healthcheck must verify /health endpoint via curl"

    # Validate Dashboard service
    assert (
        "dashboard" in services
    ), "Service 'dashboard' must be defined in docker-compose.yml"
    dashboard = services["dashboard"]
    assert "ports" in dashboard, "Dashboard service must define ports"
    assert any(
        "8501:8501" in str(p) for p in dashboard["ports"]
    ), "Dashboard service must map port 8501:8501"
    assert "depends_on" in dashboard, "Dashboard service must declare depends_on"
    depends_on = dashboard["depends_on"]
    assert (
        "api" in depends_on
    ), "Dashboard must depend on 'api'"
    assert (
        isinstance(depends_on["api"], dict)
        and depends_on["api"].get("condition") == "service_healthy"
    ), "Dashboard must depend on api with condition: service_healthy"

    # Validate Network configuration
    assert "networks" in compose_data, "docker-compose.yml must define 'networks'"
    networks = compose_data["networks"]
    assert (
        "stock_network" in networks
    ), "docker-compose.yml must define 'stock_network'"
    assert (
        networks["stock_network"].get("driver") == "bridge"
    ), "stock_network must use the bridge driver"


def test_makefile_targets():
    """Validates Makefile targets, hard tab indentation, and operational recipes."""
    makefile_path = REPO_ROOT / "Makefile"
    assert makefile_path.exists(), f"Makefile not found at {makefile_path}"

    content = makefile_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    # Required targets per operational specification
    required_targets = [
        "help",
        "install",
        "prepare-data",
        "train",
        "backtest",
        "test",
        "api",
        "dashboard",
        "docker-build",
        "docker-up",
        "docker-down",
        "clean",
    ]

    target_pattern = re.compile(r"^([a-zA-Z0-9_\-]+)\s*:")
    parsed_targets = set()

    for idx, line in enumerate(lines):
        match = target_pattern.match(line)
        if match:
            target_name = match.group(1)
            parsed_targets.add(target_name)

            # Check subsequent recipe lines for hard tab indentation (\t)
            for subsequent in lines[idx + 1:]:
                if not subsequent.strip():
                    continue
                if subsequent.startswith("\t"):
                    continue
                elif re.match(r"^[a-zA-Z0-9_\-]+\s*:", subsequent):
                    break
                elif subsequent.startswith("#") or "=" in subsequent:
                    break
                else:
                    pytest.fail(
                        f"Makefile recipe for target '{target_name}' contains non-tab indented line: {subsequent!r}"
                    )

    for target in required_targets:
        assert (
            target in parsed_targets
        ), f"Makefile must define target '{target}'. Found: {parsed_targets}"

    # Verify recipe toolchain invocations
    assert (
        "uv sync" in content
    ), "Makefile must invoke 'uv sync' in install recipe"
    assert (
        "uv run python" in content
    ), "Makefile must use 'uv run python' for execution"
    assert (
        "uv run pytest" in content
    ), "Makefile must use 'uv run pytest' for testing"
    assert (
        "docker compose" in content
    ), "Makefile must invoke 'docker compose' for container targets"


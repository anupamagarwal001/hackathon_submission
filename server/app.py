# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the AMC allocator environment."""

from __future__ import annotations

import os

try:
    from openenv.core.env_server.http_server import create_app
except Exception as error:  # pragma: no cover
    raise ImportError(
        "openenv-core is required for the web interface. Install dependencies with:\n"
        "  /Users/anuagar/Library/Python/3.11/bin/uv sync --python python3.11"
    ) from error

try:
    from ..models import AllocatorObservation, PortfolioAction
    from ..tasks import DEFAULT_TASK_ID
    from .amc_environment import AmcAllocatorEnvironment
except ImportError:  # pragma: no cover
    from models import AllocatorObservation, PortfolioAction
    from tasks import DEFAULT_TASK_ID
    from server.amc_environment import AmcAllocatorEnvironment


def build_environment() -> AmcAllocatorEnvironment:
    task_id = os.getenv("AMC_TASK_ID", DEFAULT_TASK_ID)
    return AmcAllocatorEnvironment(task_id=task_id)


app = create_app(
    build_environment,
    PortfolioAction,
    AllocatorObservation,
    env_name="amc_allocator_env",
    max_concurrent_envs=4,
)


def run_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    run_server(port=args.port)


if __name__ == "__main__":
    main()

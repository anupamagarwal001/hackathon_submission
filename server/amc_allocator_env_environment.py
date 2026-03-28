# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compatibility shim for the allocator environment module."""

from .amc_environment import AmcAllocatorEnvironment

__all__ = ["AmcAllocatorEnvironment"]

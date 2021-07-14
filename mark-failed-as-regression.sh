#!/bin/bash
if [[ ! -e ./Cargo.toml ]] ; then
    echo "error: must be in CARGO_MANIFEST_DIR" >&2
    exit 1
fi

cargo ltest -q --lib test_helpers::mark_failed_as_regression --features test-helpers -- --nocapture && rm tests/failures/*

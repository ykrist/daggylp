#!/bin/bash
if [[ ! -e ./Cargo.toml ]] ; then
    echo "error: must be in CARGO_MANIFEST_DIR" >&2
    exit 1
fi
cargo lrun --bin mark-failed-as-regression && rm tests/failures/*

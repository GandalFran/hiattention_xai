#!/bin/bash
# Dataset Download Script for HiAttention-XAI
# Downloads and prepares training datasets

set -e

echo "=========================================="
echo "HiAttention-XAI Dataset Download"
echo "=========================================="

# Configuration
DATA_DIR="${1:-./datasets}"
mkdir -p "$DATA_DIR"/{raw,processed}

cd "$DATA_DIR"

echo ""
echo "Downloading datasets..."

# =========================================
# Tier 1: Clean Benchmarks
# =========================================
echo ""
echo "=== Tier 1: Clean Benchmarks ==="

# CWE Juliet Test Suite (Security vulnerabilities)
echo "Downloading Juliet Test Suite..."
if [ ! -d "raw/juliet" ]; then
    wget -q https://samate.nist.gov/SARD/downloads/test-suites/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3.zip -O juliet.zip || echo "Note: Juliet requires manual download from NIST"
    if [ -f juliet.zip ]; then
        unzip -q juliet.zip -d raw/juliet
        rm juliet.zip
    fi
fi

# Defects4J (Java bug database)
echo "Downloading Defects4J metadata..."
if [ ! -d "raw/defects4j" ]; then
    mkdir -p raw/defects4j
    # Clone index only
    git clone --depth 1 https://github.com/rjust/defects4j.git raw/defects4j 2>/dev/null || echo "Defects4J clone failed, may need manual setup"
fi

# =========================================
# Tier 2: Large-Scale Repositories
# =========================================
echo ""
echo "=== Tier 2: Large-Scale Repositories ==="

# Apache Commons (Well-documented, good bug history)
echo "Downloading Apache Commons Lang..."
if [ ! -d "raw/apache-commons-lang" ]; then
    git clone --depth 1 https://github.com/apache/commons-lang.git raw/apache-commons-lang 2>/dev/null || echo "Clone failed"
fi

echo "Downloading Apache Commons IO..."
if [ ! -d "raw/apache-commons-io" ]; then
    git clone --depth 1 https://github.com/apache/commons-io.git raw/apache-commons-io 2>/dev/null || echo "Clone failed"
fi

echo "Downloading Apache Commons Collections..."
if [ ! -d "raw/apache-commons-collections" ]; then
    git clone --depth 1 https://github.com/apache/commons-collections.git raw/apache-commons-collections 2>/dev/null || echo "Clone failed"
fi

# Python projects for multi-language support
echo "Downloading Flask..."
if [ ! -d "raw/flask" ]; then
    git clone --depth 1 https://github.com/pallets/flask.git raw/flask 2>/dev/null || echo "Clone failed"
fi

echo "Downloading Django..."
if [ ! -d "raw/django" ]; then
    git clone --depth 1 https://github.com/django/django.git raw/django 2>/dev/null || echo "Clone failed"
fi

echo "Downloading Requests..."
if [ ! -d "raw/requests" ]; then
    git clone --depth 1 https://github.com/psf/requests.git raw/requests 2>/dev/null || echo "Clone failed"
fi

# =========================================
# Public Bug Datasets
# =========================================
echo ""
echo "=== Bug Datasets ==="

# BugSwarm (Python bugs)
echo "Downloading BugSwarm dataset..."
if [ ! -d "raw/bugswarm" ]; then
    mkdir -p raw/bugswarm
    wget -q https://raw.githubusercontent.com/BugSwarm/bugswarm/master/database/processed_dataset.json -O raw/bugswarm/dataset.json 2>/dev/null || echo "BugSwarm download failed"
fi

# CVE Details (for labeling)
echo "Downloading CVE data..."
if [ ! -f "raw/cve-data.json" ]; then
    # This is a placeholder - in production, you'd use the NVD API
    echo '{"note": "Use NVD API for real CVE data: https://nvd.nist.gov/developers/vulnerabilities"}' > raw/cve-data.json
fi

# =========================================
# Summary
# =========================================
echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Dataset locations:"
echo "  Raw data:       $DATA_DIR/raw/"
echo "  Processed data: $DATA_DIR/processed/"
echo ""
echo "Next steps:"
echo "  1. Run preprocessing: python preprocess_data.py --input $DATA_DIR/raw --output $DATA_DIR/processed"
echo "  2. Check statistics:  ls -la $DATA_DIR/processed/"
echo ""

# Count files
echo "Downloaded repositories:"
ls -1d raw/*/ 2>/dev/null | wc -l | xargs echo "  Total:"

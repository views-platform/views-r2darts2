#!/usr/bin/env bash
# Validates internal consistency of the views-r2darts2 governance documentation set
# (docs/ADRs, docs/CICs, docs/contributor_protocols, docs/standards).
# Exit 0 if clean, exit 1 if issues found.
#
# Adapted from base_docs/validate_docs.sh for a brownfield repo whose docs live
# under docs/ and whose ADRs extend past the constitutional 000-009 range:
#   - Cross-ADR check validates every locally-defined ADR number (000..max present)
#     rather than a hardcoded 000-009 ceiling, so it self-adjusts as ADRs are added.
#   - References to ADR numbers above the local maximum are treated as cross-repo
#     (e.g. views-pipeline-core ADR-052) and intentionally skipped.
#   - CIC contract check accepts lowercase snake_case filenames (darts_forecaster.md).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

errors=0

echo "=== views-r2darts2 docs validation ==="
echo ""

# Highest locally-defined ADR number (e.g. 015). References above this are
# considered cross-repo satellites and are not checked for a local file.
max_local_adr=$(ls ADRs 2>/dev/null | grep -oE '^[0-9]{3}' | sort -n | tail -1)
max_local_adr=${max_local_adr:-009}
echo "--- Local ADR ceiling: ADR-${max_local_adr} (higher references treated as cross-repo) ---"

# 1. Check for unfilled template placeholders in accepted/active files
#    (skip files whose names contain "template" — those are expected to have placeholders).
#    Warnings only (non-blocking).
echo "--- Checking for template placeholders in accepted/active files ---"
warnings=0
while IFS= read -r file; do
    [[ -z "$file" ]] && continue
    [[ "$file" == *template* ]] && continue
    if grep -q 'YYYY-MM-DD' "$file"; then
        echo "  WARN: Unfilled date placeholder in $file"
        warnings=$((warnings + 1))
    fi
    if grep -q '<roles / team>' "$file"; then
        echo "  WARN: Unfilled deciders placeholder in $file"
        warnings=$((warnings + 1))
    fi
    if grep -q '<ClassName>' "$file"; then
        echo "  WARN: Unfilled ClassName placeholder in $file"
        warnings=$((warnings + 1))
    fi
done < <(grep -rl 'Status:.*\(Accepted\|Active\)' --include='*.md' . 2>/dev/null || true)
if [ "$warnings" -eq 0 ]; then
    echo "  OK"
fi

# 2. Verify CIC active contracts listed in CICs/README.md exist (accept snake_case).
echo "--- Checking CIC active contract references ---"
if [ -f "CICs/README.md" ]; then
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        contract=$(echo "$line" | sed -n 's/^- `\([A-Za-z0-9_]*\.md\)`.*$/\1/p')
        if [ -n "$contract" ] && [ ! -f "CICs/$contract" ]; then
            echo "  ERROR: CIC contract listed but missing: CICs/$contract"
            errors=$((errors + 1))
        fi
    done < <(grep -E '^- `[A-Za-z0-9_]+\.md`' CICs/README.md 2>/dev/null | grep -v '>' || true)
fi

# 3. Cross-ADR reference integrity. Validate every referenced ADR number that is
#    at or below the local ceiling; skip higher numbers (cross-repo satellites).
echo "--- Checking cross-ADR references (local: 000-${max_local_adr}) ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    adr_num=$(echo "$ref" | grep -oP 'ADR-\K[0-9]{3}' | head -1)
    if [ -n "$adr_num" ]; then
        # 10# forces base-10 so leading zeros (008, 009) are not read as octal.
        if [ "$((10#$adr_num))" -le "$((10#$max_local_adr))" ]; then
            match_count=$(find ADRs -name "${adr_num}_*.md" 2>/dev/null | wc -l)
            if [ "$match_count" -eq 0 ]; then
                echo "  ERROR: $file references ADR-${adr_num} but no matching file found"
                errors=$((errors + 1))
            fi
        fi
    fi
done < <(grep -rno 'ADR-[0-9]\{3\}' --include='*.md' . 2>/dev/null || true)

# 4. Check that referenced contributor-protocol files exist.
echo "--- Checking protocol file references ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    proto=$(echo "$ref" | grep -oP 'contributor_protocols/[a-z_]+\.md' | head -1)
    if [ -n "$proto" ] && [ ! -f "$proto" ]; then
        echo "  ERROR: $file references $proto but file does not exist"
        errors=$((errors + 1))
    fi
done < <(grep -rn 'contributor_protocols/' --include='*.md' . 2>/dev/null || true)

# 5. Check that referenced standards files exist.
echo "--- Checking standards file references ---"
while IFS= read -r ref; do
    [[ -z "$ref" ]] && continue
    file=$(echo "$ref" | cut -d: -f1)
    std=$(echo "$ref" | grep -oP 'standards/[A-Za-z_]+\.md' | head -1)
    if [ -n "$std" ] && [ ! -f "$std" ]; then
        echo "  ERROR: $file references $std but file does not exist"
        errors=$((errors + 1))
    fi
done < <(grep -rn 'standards/' --include='*.md' . 2>/dev/null || true)

# 6. Report template status markers (informational).
echo "--- Checking template status markers ---"
template_count=$(grep -rl '\-\-template\-\-' --include='*.md' . 2>/dev/null | wc -l)
echo "  INFO: $template_count file(s) still have --template-- status"

echo ""
if [ "$errors" -gt 0 ]; then
    echo "=== FAILED: $errors issue(s) found ==="
    exit 1
else
    echo "=== PASSED: no issues found ==="
    exit 0
fi

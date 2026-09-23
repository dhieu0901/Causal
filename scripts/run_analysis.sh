#!/usr/bin/env bash
# Regenerate every result file, in dependency order, with no API calls.
#
#     bash scripts/run_analysis.sh
#
# This is the analysis half of the project; scripts/run_full.sh is the
# experiment half and spends API credit on anything not already cached. Every
# number in docs and in the manuscript comes from a results/*.csv that one of
# these scripts writes, from the per-response records in results/*_raw*.csv.
#
# The order is NOT kept here. It is read from ORDER in check_pipeline_order.py,
# the one canonical list, so this script, the order gate and
# verify_determinism.py --all cannot drift apart. An earlier copy of this loop
# lived outside the repository, so no reader could run it - review round 10,
# finding P1.
#
# The last three entries of ORDER are the gates: check_numbers (every quantity
# in the text against results/), check_pipeline_order (no script reads a file a
# later script writes) and verify_determinism (a re-run reproduces every file
# byte for byte). check_numbers exiting non-zero is reported, not fatal; any
# other failure is.

cd "$(dirname "$0")/.." || exit 1
export PYTHONIOENCODING=utf-8

ORDER=$(python -c "import importlib.util as u; s=u.spec_from_file_location('c','scripts/check_pipeline_order.py'); m=u.module_from_spec(s); s.loader.exec_module(m); print(' '.join(m.ORDER))") || {
  echo "could not read ORDER from scripts/check_pipeline_order.py"; exit 1; }

fail=0
for b in $ORDER; do
  f="scripts/$b.py"
  [ -f "$f" ] || { printf "  %-28s MISSING\n" "$b"; fail=1; continue; }
  s=$(date +%s)
  out=$(python "$f" 2>&1); code=$?
  d=$(( $(date +%s) - s ))
  if [ $code -eq 0 ]; then
    printf "  %-28s OK    %3ds\n" "$b" "$d"
  else
    printf "  %-28s EXIT %-3d %3ds  %s\n" "$b" "$code" "$d" \
      "$(printf '%s' "$out" | grep -E "Error|Traceback|unaccounted|differ" | tail -1 | cut -c1-70)"
    [ "$b" = "check_numbers" ] || fail=1
  fi
done
echo "=== analysis done, hard failures: $fail ==="
exit $fail

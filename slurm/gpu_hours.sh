#!/bin/bash
# GPU-hour accounting for the midterm deck's budget slide.
#
#   bash slurm/gpu_hours.sh              # everything since the start of the month
#   bash slurm/gpu_hours.sh 2026-11-01   # since a given date
#
# sacct gives elapsed time and the GPUs allocated per job. The deck's slide 5
# says a node-hour is 4 A100s for an hour, and that 1-2 GPU jobs on the shared
# QOS are charged per GPU, so one GPU for one hour costs 0.25 node-hours.
#
# This is OUR estimate from job records. The authoritative charged figure is in
# Iris (iris.nersc.gov); quote Iris in the report and use this to see the split
# by experiment.

set -euo pipefail
SINCE=${1:-$(date +%Y-%m-01)}

echo "jobs since $SINCE"
echo

sacct --starttime="$SINCE" --user="$USER" \
      --format=JobID,JobName%34,State,Elapsed,AllocTRES%46 \
      --parsable2 --noheader --allocations \
| awk -F'|' '
{
    jobid=$1; name=$2; state=$3; elapsed=$4; tres=$5;

    # Elapsed is [DD-]HH:MM:SS
    d=0; rest=elapsed;
    if (index(elapsed, "-") > 0) { split(elapsed, p, "-"); d=p[1]; rest=p[2]; }
    split(rest, t, ":");
    secs = d*86400 + t[1]*3600 + t[2]*60 + t[3];

    # GPUs from AllocTRES, e.g. "gres/gpu=4"
    g = 0;
    if (match(tres, /gres\/gpu=[0-9]+/)) {
        s = substr(tres, RSTART, RLENGTH); split(s, gg, "="); g = gg[2];
    }
    if (g == 0) next;

    gpuh = secs/3600.0 * g;
    nodeh = gpuh / 4.0;              # 4 A100s per Perlmutter GPU node

    # group by experiment prefix: asl-weak-mobilenet_v2-4 -> asl-weak
    # (awk has no backreferences, so split on "-" and rejoin the first two)
    np = split(name, parts, "-");
    n = (np >= 2) ? parts[1] "-" parts[2] : name;
    sumg[n] += gpuh; sumn[n] += nodeh; cnt[n]++;
    totg += gpuh; totn += nodeh; totc++;

    printf "  %-10s %-34s %-10s %8s  %2d GPU  %7.2f GPU-h\n", jobid, name, state, elapsed, g, gpuh;
}
END {
    printf "\n%-26s %7s %12s %12s\n", "experiment", "jobs", "GPU-hours", "node-hours";
    printf "%-26s %7s %12s %12s\n", "--------------------------", "-------", "------------", "------------";
    for (k in sumg) printf "%-26s %7d %12.2f %12.2f\n", k, cnt[k], sumg[k], sumn[k];
    printf "%-26s %7s %12s %12s\n", "--------------------------", "-------", "------------", "------------";
    printf "%-26s %7d %12.2f %12.2f\n", "TOTAL", totc, totg, totn;
    printf "\nnode-hours = GPU-hours / 4. Shared-QOS jobs (1-2 GPUs) are charged\n";
    printf "per GPU, so this estimate is right for them too.\n";
    printf "Authoritative charged total: iris.nersc.gov\n";
}'

#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t joeranbosma/dragon_roberta_large_mixed_domain:latest -t joeranbosma/dragon_roberta_large_mixed_domain:v0.2.1 "$SCRIPTPATH"

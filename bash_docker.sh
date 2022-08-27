#!/bin/bash

set -eux

docker run -d --restart=always -p 31222:31222 -v /home/zj/repos/sis/static:/Project/sis/static sis:0.1.0
#!/bin/bash

set -xe

clang -Wall -Wextra -o mult2 mult2.c -lm
clang -Wall -Wextra -o gates gates.c -lm
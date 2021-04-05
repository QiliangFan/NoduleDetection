#!/usr/bin/env bash

export IP=$(ifconfig | grep -E -o '[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+\.[[:digit:]]+'|grep -Ev '127|255|172')
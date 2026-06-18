#!/bin/bash

set -e

pip install --upgrade pip
pip install -e ".[dev,test]"
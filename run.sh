#!/bin/bash
echo "Running tests..."
poetry run pytest tests/ -v

if [ $? -eq 0 ]; then
    echo "All tests passed!"
else
    echo "Tests failed!"
    exit 1
fi
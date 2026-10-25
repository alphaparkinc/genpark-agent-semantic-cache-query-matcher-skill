# Semantic Cache Query Matcher Skill for Autonomous Agents

[![MCP Compatible](https://img.shields.io/badge/MCP-Compatible-blue.svg)](https://modelcontextprotocol.io)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-brightgreen.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero External Dependencies](https://img.shields.io/badge/dependencies-zero-success.svg)](requirements.txt)

High-performance in-memory semantic cache designed to slash LLM inference costs and latency by caching query-response pairs with fuzzy cosine term-vector matching.

## Key Features
- **Cosine Term Vector Matching**: Zero external vector database; computes term-frequency vector similarity in pure Python.
- **Configurable Similarity Threshold**: Tunable confidence threshold (e.g. 0.75 - 0.95) to prevent false-positive matches.
- **Hit Rate Telemetry**: Tracks hits, misses, and saved token estimations.

## MCP Server Integration
```bash
python mcp_server.py
```

### Available MCP Tools
- `cache_set`: Store query and generated response.
- `cache_get`: Lookup semantic match for an incoming prompt.
- `cache_stats`: Return total entries, hits, and cache utilization.

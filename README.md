# PLFactory

A Multi-Agent Framework for Diverse and High-quality NL-to-PL/SQL Data Generation.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Generation](#data-generation)
  - [Seed Generation](#seed-generation)
  - [Seed Expansion](#seed-expansion)
  - [Translation](#translation)
- [Train](#train)
  - [SFT Stage](#sft-stage)
  - [RL Stage](#rl-stage)
- [Experiments](#experiments)
  - [Natural Language to PL/SQL](#natural-language-to-plsql)
  - [Evaluation Scripts](#evaluation-scripts)

---

## Overview

PLFactory provides a complete pipeline for:
1. **Seed Generation**: Generating seed PL/SQL procedures, functions, and triggers
2. **Seed Expansion**: Expanding datasets through intelligent augmentation
3. **Translation**: Translating between database dialects
4. **Evaluation**: Evaluating model performance on PL/SQL generation tasks

---

## Environment Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable Python package management.

### Prerequisites

- Python 3.12 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installing uv

If you haven't installed uv yet, install it using one of the following methods:

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using pip
pip install uv
```

### Project Configuration

The project uses two key files for dependency management:

- **`pyproject.toml`**: Defines project metadata and dependencies
- **`uv.lock`**: Locks exact versions of all dependencies for reproducible builds

### Installation Steps

1. **Clone the repository**

```bash
git clone <repository-url>
cd plfactory
```

2. **Create and activate a virtual environment**

```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

3. **Install dependencies**

```bash
# Install all dependencies from uv.lock (recommended for exact reproducibility)
uv sync

# Or install from pyproject.toml (will resolve latest compatible versions)
uv pip install -e .
```

---

## Data Generation

### Seed Generation

The `seed_generation_main.py` script generates initial seed data for PL/SQL code generation.

#### Description
This script creates seed PL/SQL code (procedures, functions, triggers) based on database schemas. It uses an agentic workflow to generate intermediate representations (IR) and corresponding PL/SQL implementations.

#### Usage

```bash
python seed_generation_main.py [OPTIONS]
```

#### Arguments

- `--epoch`: Epoch number for training iteration
- `--target-plsql-number`: Target number of PL/SQL code samples to generate
- `--current-plsql-number`: Current number of PL/SQL samples already generated
- `--dialect`: SQL dialect to use (choices: "postgresql", "oracle")

#### Examples

```bash
# Generate 10 Oracle PL/SQL seeds
python seed_generation_main.py --dialect oracle --target-plsql-number 1000
```

---

### Seed Expansion

The `expansion_main.py` script expands existing seed data through intelligent augmentation techniques.

#### Description
This script takes existing PL/SQL seed data and generates variations and expanded versions to increase dataset diversity. It applies transformation strategies while maintaining semantic correctness.

#### Usage

```bash
python expansion_main.py [OPTIONS]
```

#### Arguments

- `--target-plsql-number` (int, default: 20): Target number of expanded PL/SQL samples
- `--dialect` (str, default: "postgresql"): SQL dialect to use (choices: "postgresql", "oracle")

#### Examples

```bash
# Expand Oracle dataset to 50 samples
python expansion_main.py --dialect oracle --target-plsql-number 50

# Expand PostgreSQL dataset
python expansion_main.py --dialect postgresql --target-plsql-number 100
```

---

### Translation

The `translation_main.py` script translates PL/SQL code between different database dialects.

#### Description
This script converts PL/SQL code from one database dialect to another (e.g., Oracle to PostgreSQL or vice versa). It handles syntax differences, built-in functions, and dialect-specific features.

#### Usage

```bash
python translation_main.py
```

---

## Train

The training pipeline consists of two phases: Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL).

### SFT Stage

The Supervised Fine-Tuning stage is implemented using the LLama-Factory framework. This stage focuses on aligning the base model with high-quality PL/SQL instruction data to ensure syntactical correctness and adherence to instructions.

### RL Stage

The Reinforcement Learning stage utilizes the verl framework to further optimize the model's performance.

The reward mechanism is designed to ensure execution correctness and semantic equivalence. The implementation includes the following key technical points (`train/verl/utils/reward_score/plfactory.py`):

- Dual-Backend Support: The system supports both PostgreSQL and Oracle environments, handling connection pooling and dialect-specific execution logic.
- Isolated Execution Environments:
  - PostgreSQL: Utilizes template databases and the Copy-On-Write (COW) mechanism to rapidly spawn isolated worker databases for each evaluation task, ensuring high performance and data safety.
  - Oracle: Manages isolated worker schemas (users), dynamically creating and dropping users to prevent state pollution between runs.
- Execution-Based Verification:
  - The system executes both the Generated Solution and the Ground Truth in fresh, restored database environments.
  - It runs a series of "call statements" (test cases) against both versions.
- Data-Driven Comparison:
  - The results of the call statements are captured as Pandas DataFrames.
  - The reward score (0.0 to 1.0) is calculated based on the ratio of test cases where the generated code's output exactly matches the ground truth's output.
- Robustness: Includes mechanisms for connection retries (handling Oracle network instability), transaction rollbacks on failure, and automatic resource cleanup (dropping temporary databases/schemas) to prevent leaks.

---

## Experiments

### Natural Language to PL/SQL

Two scripts enable testing LLM models on NL-to-PL/SQL generation tasks.

#### Oracle NL2PL/SQL (`oracle_nl2plsql.py`)

Generates Oracle PL/SQL code from natural language descriptions using various LLM models.

**Usage:**

```bash
python experiments/codes/oracle_nl2plsql.py --dataset DATASET_NAME --model MODEL_NAME [OPTIONS]
```

**Arguments:**

- `--dataset` (required): Dataset name (e.g., `oracle_spider_function_test`)
- `--model` (required): Model name or local model path
  - API models: `gpt-5.1`, `gpt-4o`, `o4-mini`
  - Local models: `/path/to/model` (e.g., `/home/user/models/starcoder2-3b`)
- `--max_tokens` (int, default: 8192): Maximum tokens for generation
- `--input-key`: Input field to use
  - `ir`: PLIR
  - `sum`: Summary
  - `nl`: Natural Language
  - `gr`: Generated IR

**Examples:**

```bash
# Using API model (GPT-5.1)
python experiments/codes/oracle_nl2plsql.py \
  --dataset oracle_spider_function_test \
  --model gpt-5.1

# Using local model
python experiments/codes/oracle_nl2plsql.py \
  --dataset oracle_omni_procedure_test \
  --model /home/user/models/starcoder2-7b \
  --max_tokens 4096

# Using summary as input
python experiments/codes/oracle_nl2plsql.py \
  --dataset oracle_spider_trigger_test \
  --model gpt-5.1 \
  --input-key sum
```

---

#### PostgreSQL NL2PL/pgSQL (`postgres_nl2plsql.py`)

Generates PostgreSQL PL/pgSQL code from natural language descriptions.

**Usage:**

```bash
python experiments/codes/postgres_nl2plsql.py --dataset DATASET_NAME --model MODEL_NAME [OPTIONS]
```

**Arguments:**

Same as `oracle_nl2plsql.py`, but for PostgreSQL datasets.

- `--dataset` (required): Dataset name (e.g., `postgres_spider_function_test`)
- `--model` (required): Model name or local model path
- `--max_tokens` (int, default: 8192): Maximum tokens for generation
- `--input-key`: Input field to use (ir/sum/nl/gr)

**Examples:**

```bash
# Using API model (GPT-5.1)
python experiments/codes/postgres_nl2plsql.py \
  --dataset postgres_spider_function_test \
  --model gpt-5.1

# Using local model with custom settings
python experiments/codes/postgres_nl2plsql.py \
  --dataset postgres_omni_procedure_test \
  --model /home/user/models/codellama-13b \
  --max_tokens 8192 \
  --input-key nl
```

---

### Evaluation Scripts

Evaluate the accuracy of generated PL/SQL code against gold standards.

#### Oracle Evaluation (`eval_oracle.py`)

Evaluates Oracle PL/SQL generation results using two metrics:
- **EM (Exact Match)**: Semantic AST-based matching
- **EX (Execution)**: Actual database execution comparison

**Usage:**

```bash
python experiments/codes/eval_oracle.py FILENAME [OPTIONS]
```

**Arguments:**

- `FILENAME` (required): Result filename without `.json` extension
  - Format: `oracle_{db}_{type}_test-{model}-{shot}_shot`
  - Example: `oracle_spider_function_test-gpt-5.1-0_shot`
- `--host` (optional): Database host address
- `--port` (optional): Database port

**Metrics:**

- **EM Score**: Measures semantic equivalence using AST comparison
- **EX Score**: Measures execution correctness by comparing actual database outputs

**Examples:**

```bash
# Evaluate results with default database connection
python experiments/codes/eval_oracle.py oracle_spider_function_test-gpt-5.1-0_shot

# Evaluate with custom database connection
python experiments/codes/eval_oracle.py oracle_omni_procedure_test-gpt-5.1-0_shot \
  --host localhost \
  --port 1521

# Evaluate trigger results
python experiments/codes/eval_oracle.py oracle_spider_trigger_test-gpt-5.1-0_shot
```

---

#### PostgreSQL Evaluation (`eval_postgres.py`)

Evaluates PostgreSQL PL/pgSQL generation results.

**Usage:**

```bash
python experiments/codes/eval_postgres.py FILENAME [OPTIONS]
```

**Arguments:**

- `FILENAME` (required): Result filename without `.json` extension
  - Format: `postgres_{db}_{type}_test-{model}-{shot}_shot`
  - Example: `postgres_spider_function_test-gpt-5.1-0_shot`
- `--host` (optional): Database host address
- `--port` (optional): Database port

**Metrics:**

Same as Oracle evaluation:
- **EM Score**: AST-based semantic matching
- **EX Score**: Execution-based correctness

**Examples:**

```bash
# Evaluate PostgreSQL function results
python experiments/codes/eval_postgres.py postgres_spider_function_test-gpt-5.1-0_shot

# Evaluate with custom connection
python experiments/codes/eval_postgres.py postgres_omni_procedure_test-gpt-5.1-0_shot \
  --host localhost \
  --port 5432

# Evaluate trigger generation
python experiments/codes/eval_postgres.py postgres_spider_trigger_test-gpt-5.1-0_shot
```
---

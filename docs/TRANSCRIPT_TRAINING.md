# Transcript Training Pipeline Implementation

This document describes the implementation of the Bob Loukas transcript training pipeline as specified in the plan.

## Overview

The system downloads Bob Loukas video transcripts from Box.com and creates training datasets:
- **Dataset A**: Transcripts only
- **Dataset B**: Transcripts + Textbook (combined)
- **Baseline**: Existing textbook-only (for comparison)

## Implementation Summary

### 1. Download Infrastructure ✅

**File**: `scripts/download_transcripts.py`

- Implements Box JWT authentication (reuses logic from mindprint-agent)
- Uses folder ID: `319911287224` (from bob-loukas/config.yaml)
- Downloads transcripts to `data/bob_loukas/transcripts/raw/`
- Generates metadata file: `transcripts_metadata.json`
- Supports date parsing from various filename formats:
  - `YYYY-MM-DD.txt` (standard format)
  - `YYYY-MM-DDTHH-MM-SSZ.txt` (ISO timestamp format)
  - `YYYY-MM-DDTHH-MM-SSZ_F.txt` (timestamp with _F suffix)
  - `session_YYYY-MM-DD.txt` (session prefix format)
  - `YYYYMMDD.txt` (compact format)

**Usage**:
```bash
python scripts/download_transcripts.py --count 50 --output-dir data/bob_loukas/transcripts/raw
```

**Environment Setup**:
- **Option 1 (Recommended)**: Run the setup script:
  ```bash
  ./scripts/setup_box_config.sh
  ```
  This automatically extracts `BOX_CONFIG_JSON` from mindprint-agent's `.env.local` and sets it up.

- **Option 2 (Manual)**: Copy `BOX_CONFIG_JSON` from mindprint-agent's `.env.local`:
  ```bash
  export BOX_CONFIG_JSON='{"boxAppSettings":{...},"enterpriseID":"..."}'
  ```

- Install Box SDK: `pip install boxsdk` (already added to requirements.txt)

### 2. Transcript Processing Module ✅

**File**: `src/data_prep/transcript_processor.py`

- `TranscriptProcessor` class (parallel to `TextbookParser`)
- Methods:
  - `parse_transcript(file_path)` → raw text
  - `load_episode_summary(date)` → structured summary (from mindprint-agent summaries.json)
  - `generate_qa_pairs(transcript, summary)` → List[Question]

**Question Generation Strategies**:
1. **Summary-based** (high-level):
   - "What was Bob's market thesis on {date}?"
   - "What risks did Bob identify on {date}?"
   - "What opportunities was Bob watching on {date}?"

2. **Topic-extraction** (mid-level):
   - Uses `summary.topics` field
   - "Explain Bob's view on {topic} from {date}"

3. **Chunk-based** (detailed, fallback):
   - Splits transcript into semantic chunks when no summary available

### 3. Question Generator for Transcripts ✅

**File**: `src/data_prep/transcript_question_generator.py`

- Generates 10-20 questions per episode
- Mix of summary-based (3-5) and topic-based (7-15)
- Uses episode summaries from mindprint-agent as context
- Preserves Bob's terminology and phrasing in answers
- Reuses `GenerationConfig` for consistency

### 4. Preference Pair Generation ✅

**File**: `src/data_prep/preference_generator.py` (modified)

**Added Rejection Strategies for Transcripts**:
- **Overly formal**: Bob speaks casually → make rejected answers academic
- **Missing terminology**: Remove "DCL", "right-translated", "Bressert bands"
- **Generic advice**: Replace specific insights with platitudes
- **Missing time context**: Remove episode date references

### 5. Pipeline Integration ✅

**File**: `src/data_prep/pipeline.py` (modified)

**Changes**:
1. Added `TranscriptProcessor` alongside `TextbookParser`
2. Added config options:
   - `transcript_dir: Optional[str]`
   - `use_transcripts: bool`
   - `combine_with_textbook: bool`
   - `textbook_ratio: float` (default 0.6 = 60% textbook, 40% transcripts)
3. Created three output modes:
   - **transcripts_only**: Use only transcript data
   - **combined**: Merge transcripts + textbook
   - **textbook_only**: Existing behavior (baseline)

**New Directory Structure**:
```
data/bob_loukas/
├── textbook/              # Existing
│   ├── sft_data.jsonl
│   └── preference_data.jsonl
├── transcripts/           # NEW
│   ├── raw/*.txt
│   ├── sft_data.jsonl
│   └── preference_data.jsonl
└── combined/              # NEW
    ├── sft_data.jsonl
    └── preference_data.jsonl
```

**Combining Logic**:
- Merges JSONL files with specified ratio
- Shuffles to mix sources
- Preserves `source` field to track origin

### 6. Training Configuration ✅

**Created**:
- `configs/bob_loukas_transcripts.yaml` (Dataset A)
- `configs/bob_loukas_combined.yaml` (Dataset B)

**Key Config Fields**:
```yaml
paths:
  data_dir: ./data/bob_loukas/transcripts  # or combined/
  checkpoint_dir: ./checkpoints/transcripts  # Separate dirs
```

### 7. Model Training

**Training Runs**:

1. **Dataset A** (Transcripts only):
   ```bash
   ./scripts/train_on_mac_studio.sh --config configs/bob_loukas_transcripts.yaml
   ```

2. **Dataset B** (Combined):
   ```bash
   ./scripts/train_on_mac_studio.sh --config configs/bob_loukas_combined.yaml
   ```

3. **Dataset C** (Textbook baseline):
   - Already exists (previous run)
   - Rerun if needed for fair comparison

### 8. Evaluation & Comparison ✅

**File**: `scripts/compare_models.py`

**Evaluation Metrics**:
- **Knowledge accuracy**: Semantic similarity to reference answers
- **Voice fidelity**: Semantic similarity to Bob's style
- **Terminology usage**: Count Bob-specific terms
- **Conversational tone**: Human evaluation (1-5 scale)

**Usage**:
```bash
python scripts/compare_models.py \
  --transcripts-model ./checkpoints/transcripts \
  --combined-model ./checkpoints/combined \
  --textbook-model ./checkpoints/textbook \
  --test-questions ./data/test_questions.json \
  --output ./reports/model_comparison.md
```

## Usage Examples

### Download Transcripts
```bash
# Option 1: Use setup script (recommended)
./scripts/setup_box_config.sh
export $(grep -v '^#' .env | xargs)

# Option 2: Manual export
export BOX_CONFIG_JSON='{"boxAppSettings": {...}, "enterpriseID": "..."}'

# Download 50 most recent transcripts
python scripts/download_transcripts.py --count 50
```

### Prepare Transcript Data
```bash
# Transcripts only (without summaries)
python scripts/run_data_prep.py \
  --mode transcripts \
  --transcript-dir data/bob_loukas/transcripts \
  --output data/bob_loukas/transcripts

# Transcripts only (with summaries from mindprint-agent)
python scripts/run_data_prep.py \
  --mode transcripts \
  --transcript-dir data/bob_loukas/transcripts \
  --summaries-dir ../mindprint-agent/data/bob-loukas/brain-areas/bob-loukas-bitcoinlive-episodes/references \
  --output data/bob_loukas/transcripts

# Combined (60% textbook, 40% transcripts)
python scripts/run_data_prep.py \
  --mode combined \
  --textbook ../omnia/projects/bob_loukas/textbook \
  --transcript-dir data/bob_loukas/transcripts \
  --summaries-dir ../mindprint-agent/data/bob-loukas/brain-areas/bob-loukas-bitcoinlive-episodes/references \
  --output data/bob_loukas/combined \
  --textbook-ratio 0.6
```

**Note on `--summaries-dir`**: This should point to the directory containing `summaries.json`, not individual date files. The file structure is:
```
mindprint-agent/data/bob-loukas/brain-areas/bob-loukas-bitcoinlive-episodes/references/
└── summaries.json  # Contains { "YYYY-MM-DD": { "thesis": "...", ... }, ... }
```

### Test Transcript Processing
```bash
# Test processing without summaries
python scripts/test_transcript_processing.py --count 5

# Test processing with summaries
python scripts/test_transcript_processing.py \
  --count 5 \
  --summaries-dir ../mindprint-agent/data/bob-loukas/brain-areas/bob-loukas-bitcoinlive-episodes/references
```

## Data Flow

### Download
```
Box.com (folder 319911287224)
    ↓
[BoxClient with JWT auth]
    ↓
data/bob_loukas/transcripts/raw/{date}.txt
    ↓
transcripts_metadata.json
```

### Processing
```
Raw Transcript
    ↓
Episode Summary (optional, from mindprint-agent)
    ↓
[TranscriptProcessor.generate_qa_pairs()]
    ├── Summary-based questions
    ├── Topic-based questions
    └── (Optional) Chunk-based questions
    ↓
List[Question] → SFT examples
    ↓
[PreferencePairGenerator] → Preference pairs
    ↓
transcripts/sft_data.jsonl
transcripts/preference_data.jsonl
```

### Combining
```
textbook/sft_data.jsonl
    +
transcripts/sft_data.jsonl
    ↓
[Merge + Balance]
    ↓
combined/sft_data.jsonl
```

## Next Steps

1. **Download transcripts**: Run `download_transcripts.py` to fetch episodes
2. **Prepare data**: Run `run_data_prep.py` with appropriate mode
3. **Train models**: Use training scripts with new configs
4. **Evaluate**: Run `compare_models.py` to compare performance

## Notes

- **Episode summaries**: Optional but recommended for better question quality. Summaries are stored in a single `summaries.json` file in mindprint-agent's data directory. The `--summaries-dir` parameter should point to the directory containing this file.
- **Filename formats**: The system supports multiple transcript filename formats including ISO timestamps (`YYYY-MM-DDTHH-MM-SSZ.txt`). Dates are extracted and normalized to `YYYY-MM-DD` format.
- **Missing summaries**: The system gracefully handles missing summaries by using chunk-based extraction from transcripts.
- **Rejection patterns**: Transcript-specific rejection patterns help preserve Bob's casual, conversational tone in preference pairs.
- **Combining ratio**: The combining ratio can be adjusted based on performance (default: 60/40 textbook/transcripts).
- **Date parsing**: Both download script and processor handle timestamp formats, extracting the date portion before the 'T' separator.

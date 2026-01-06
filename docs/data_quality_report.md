# Training Data Quality Analysis

Generated from: /Users/benoit/Documents/Memetica/Code/mindprint-model

## SFT Data by Topic

| Topic | Examples | Avg Instruction | Avg Output | Voice Markers (%) |
|-------|----------|-----------------|------------|-------------------|
| unit-02 | 20 | 72 | 188 | 23.6 |
| unit-01 | 16 | 67 | 100 | 28.4 |
| unit-03 | 16 | 64 | 116 | 23.2 |
| unit-04 | 16 | 73 | 136 | 19.7 |
| unit-01/chapter-01 | 5 | 246 | 1280 | 11.6 |
| unit-01/chapter-02 | 5 | 210 | 1580 | 16.1 |
| unit-01/chapter-03 | 4 | 271 | 2105 | 15.0 |
| unit-02/chapter-04-primer | 4 | 251 | 1902 | 11.9 |
| unit-02/chapter-04 | 4 | 230 | 1264 | 11.6 |
| unit-02/chapter-05 | 4 | 178 | 1451 | 14.2 |
| unit-02/chapter-06 | 4 | 205 | 2338 | 12.4 |
| unit-02/chapter-07 | 4 | 270 | 2260 | 16.5 |
| unit-01/chapter-01/topic-01 | 3 | 233 | 1637 | 6.7 |
| unit-03/chapter-08/topic-01 | 3 | 139 | 1728 | 10.0 |
| unit-03/chapter-09/topic-02 | 3 | 202 | 1654 | 5.5 |
| unit-03/chapter-10/topic-02 | 3 | 175 | 2124 | 7.0 |
| unit-03/chapter-10/topic-03 | 3 | 159 | 2211 | 7.7 |
| unit-01/chapter-01/topic-02 | 2 | 166 | 1732 | 3.8 |
| unit-02/chapter-05/topic-01 | 2 | 171 | 1900 | 4.9 |
| unit-03/chapter-08/topic-04 | 2 | 174 | 1639 | 5.2 |
| unit-03/chapter-09/topic-01 | 2 | 148 | 1842 | 8.2 |
| unit-03/chapter-10/topic-01 | 2 | 166 | 2330 | 6.3 |
| unit-03/chapter-08 | 2 | 221 | 1205 | 13.4 |
| unit-03/chapter-09 | 2 | 284 | 1046 | 14.4 |
| unit-03/chapter-10 | 2 | 290 | 2276 | 15.6 |
| unit-04/chapter-11 | 2 | 275 | 1104 | 18.1 |
| unit-04/chapter-12 | 2 | 231 | 1347 | 14.3 |
| unit-04/chapter-13 | 2 | 322 | 1402 | 18.6 |

## Preference Data Quality by Topic

| Topic | Pairs | Avg Quality | High Quality | Quality Ratio |
|-------|-------|-------------|--------------|---------------|
| unknown | 149 | 6.20 | 143 (96.0%) | 2.57x |

## Overall Preference Data Quality

- **Total pairs**: 149
- **Topics**: 1
- **Average quality score**: 6.20
- **High quality pairs (>1.5)**: 143 (96.0%)
- **Low quality pairs (<0.5)**: 0 (0.0%)

## Key Insights

### Top Topics (by example count and voice markers):

- **unit-02**: 20 examples, 188 avg chars, 23.6% voice markers
- **unit-01**: 16 examples, 100 avg chars, 28.4% voice markers
- **unit-03**: 16 examples, 116 avg chars, 23.2% voice markers

### Recommendations for Training

Based on the analysis, optimal topics should have:

1. **15-25 preference pairs** (concentrated signal)
2. **600-1200 character outputs** (substantive but focused)
3. **>20% voice marker density** (strong distinctive voice)
4. **>80% clear quality differentiation** (quality ratio >1.5)

Topics outside these ranges may benefit from:
- More training epochs (if few examples)
- Lower pass thresholds (if weak voice markers)
- Data quality improvements (if low quality ratio)

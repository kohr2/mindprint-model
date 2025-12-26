# Phase 1: Data Preparation

## Objective

Convert the Bob Loukas textbook from markdown format into training-ready data structures:
1. SFT training pairs (question/answer)
2. Preference pairs (chosen/rejected) - used by both DPO and reward model training
3. Evaluation datasets (quizzes at all levels)

## Input Sources

### Textbook Location
```
../omnia/projects/bob_loukas/textbook/
├── curriculum.yaml              # Metadata and structure
├── bob-style-guide.md           # Voice characteristics
├── units/
│   ├── unit-01-foundation/      # 3 chapters, ~12 topics
│   ├── unit-02-cycles/          # 4 chapters, ~16 topics
│   ├── unit-03-rules/           # 3 chapters, ~12 topics
│   └── unit-04-advanced/        # 3 chapters, ~12 topics
└── tests/
    └── [matching test structure with quizzes]
```

### Content Structure (per topic)
```
topic-XX-name.md
├── Learning Objectives
├── Content (prose explanation)
├── Key Concepts
└── Summary

topic-XX-test.md
├── Quiz Questions (10 per topic)
├── Reference Answers (Bob's voice)
├── Evaluation Focus
└── Key Concepts Tested
```

## Output Formats

### 1. SFT Training Data (`sft_data.jsonl`)

```json
{
  "instruction": "What is the fundamental difference between a gambler and a trader?",
  "input": "",
  "output": "Look, the difference isn't about how much analysis you do..."
}
```

### 2. Preference Data (`preference_data.jsonl`)

Used by both DPO (directly) and PPO (for reward model training):

```json
{
  "prompt": "Explain how Bitcoin's fixed supply cap impacts market cycles.",
  "chosen": "Bitcoin's 21 million cap creates structural scarcity, which is important, but it's not what drives the 4-year cycle rhythm...",
  "rejected": "Bitcoin has a fixed supply of 21 million coins. This creates scarcity which drives demand and price increases over time."
}
```

### 3. Quiz Evaluation Data (`quiz_data.json`)

```json
{
  "level": "topic",
  "unit": "unit-01",
  "chapter": "chapter-01",
  "topic": "topic-01",
  "questions": [
    {
      "question": "Explain the fundamental difference between a gambler and a trader.",
      "reference_answer": "...",
      "type": "open",
      "key_concepts": ["gambler_definition", "trader_definition"],
      "evaluation_criteria": [
        "Factual accuracy",
        "Voice fidelity"
      ]
    }
  ]
}
```

## Implementation

### Data Preparation Script

```python
# src/data_prep.py

from pathlib import Path
from typing import List, Dict
import json
import re
import yaml


class TextbookParser:
    """Parse Bob Loukas textbook markdown into structured data."""
    
    def __init__(self, textbook_path: str):
        self.root = Path(textbook_path)
        self.curriculum = self._load_curriculum()
        
    def _load_curriculum(self) -> Dict:
        with open(self.root / "curriculum.yaml") as f:
            return yaml.safe_load(f)
    
    def parse_topic_test(self, test_path: Path) -> List[Dict]:
        """Extract quiz questions from test markdown file."""
        content = test_path.read_text()
        questions = []
        
        # Parse each question block
        question_blocks = re.split(r'### Question \d+', content)[1:]
        
        for block in question_blocks:
            q = self._parse_question_block(block)
            if q:
                questions.append(q)
                
        return questions
    
    def _parse_question_block(self, block: str) -> Dict:
        """Parse a single question block."""
        lines = block.strip().split('\n')
        
        question = ""
        reference_answer = ""
        in_answer = False
        
        for line in lines:
            if line.startswith("**Reference Answer**"):
                in_answer = True
                continue
            elif line.startswith("**Evaluation Focus**"):
                in_answer = False
                continue
            elif in_answer:
                reference_answer += line + "\n"
            elif not in_answer and not question:
                question = line.strip()
        
        return {
            "question": question,
            "reference_answer": reference_answer.strip(),
            "type": "open"
        }


class PreferencePairGenerator:
    """Generate preference pairs from quiz data."""
    
    def generate_rejected(self, question: str, reference: str) -> str:
        """Generate a rejected (generic) response."""
        return self._create_generic_response(question, reference)
    
    def _create_generic_response(self, question: str, reference: str) -> str:
        """Create a generic/inferior response by removing Bob's voice."""
        generic = reference
        
        # Remove confidence markers
        markers = [
            "I've tracked", "I've seen", "I've observed",
            "In my experience", "Here's what I've",
            "Look,", "Okay, so"
        ]
        for marker in markers:
            generic = generic.replace(marker, "")
        
        # Remove bold formatting
        generic = re.sub(r'\*\*([^*]+)\*\*', r'\1', generic)
        
        # Truncate to make it clearly inferior
        sentences = generic.split('.')
        if len(sentences) > 3:
            generic = '. '.join(sentences[:3]) + '.'
        
        return generic.strip()


class DataPipeline:
    """Complete data preparation pipeline."""
    
    def __init__(self, textbook_path: str, output_path: str):
        self.parser = TextbookParser(textbook_path)
        self.preference_gen = PreferencePairGenerator()
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Execute full data preparation."""
        sft_data = []
        preference_data = []
        quiz_data = []
        
        # Process each unit
        for unit_dir in sorted((self.parser.root / "units").iterdir()):
            if not unit_dir.is_dir():
                continue
            
            self._process_unit(unit_dir, sft_data, preference_data, quiz_data)
        
        # Save outputs
        self._save_jsonl(sft_data, "sft_data.jsonl")
        self._save_jsonl(preference_data, "preference_data.jsonl")
        self._save_json(quiz_data, "quiz_data.json")
        
        print(f"Generated {len(sft_data)} SFT examples")
        print(f"Generated {len(preference_data)} preference pairs")
        print(f"Generated {len(quiz_data)} quiz sets")
    
    def _process_unit(self, unit_dir, sft_data, preference_data, quiz_data):
        """Process a single unit."""
        unit_name = unit_dir.name
        
        for chapter_dir in sorted(unit_dir.iterdir()):
            if not chapter_dir.is_dir() or not chapter_dir.name.startswith("chapter"):
                continue
            
            chapter_name = chapter_dir.name
            
            for topic_file in sorted(chapter_dir.glob("topic-*.md")):
                topic_name = topic_file.stem
                test_file = self._find_test_file(unit_name, chapter_name, topic_name)
                
                if test_file and test_file.exists():
                    questions = self.parser.parse_topic_test(test_file)
                    
                    for q in questions:
                        # SFT data
                        sft_data.append({
                            "instruction": q["question"],
                            "input": "",
                            "output": q["reference_answer"]
                        })
                        
                        # Preference data
                        rejected = self.preference_gen.generate_rejected(
                            q["question"], 
                            q["reference_answer"]
                        )
                        preference_data.append({
                            "prompt": q["question"],
                            "chosen": q["reference_answer"],
                            "rejected": rejected
                        })
                    
                    # Quiz data
                    quiz_data.append({
                        "level": "topic",
                        "unit": unit_name,
                        "chapter": chapter_name,
                        "topic": topic_name,
                        "questions": questions
                    })
    
    def _find_test_file(self, unit: str, chapter: str, topic: str) -> Path:
        """Find corresponding test file for a topic."""
        unit_num = unit.split('-')[1] if '-' in unit else unit
        chapter_num = chapter.split('-')[1] if '-' in chapter else chapter
        topic_num = topic.split('-')[1] if '-' in topic else topic
        
        test_path = (self.parser.root / "tests" / 
                    f"unit-{unit_num}" / f"chapter-{chapter_num}" / 
                    f"topic-{topic_num}-test.md")
        return test_path
    
    def _save_jsonl(self, data: List[Dict], filename: str):
        with open(self.output_path / filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    def _save_json(self, data: List[Dict], filename: str):
        with open(self.output_path / filename, 'w') as f:
            json.dump(data, f, indent=2)


if __name__ == "__main__":
    pipeline = DataPipeline(
        textbook_path="../omnia/projects/bob_loukas/textbook",
        output_path="./data/bob_loukas"
    )
    pipeline.run()
```

## Critical Distinctions Dataset

Create special preference pairs for Bob's most important distinctions:

```json
{
  "prompt": "What causes Bitcoin's 4-year market cycle?",
  "chosen": "The 4-year cycle is NOT caused by the halving—that's a common misconception. I've been watching market cycles long before Bitcoin existed. The 4-year rhythm shows up in gold, in stocks, across different markets. It's about capital flows and how long it takes for market psychology to shift from fear to greed and back again.",
  "rejected": "Bitcoin's 4-year cycle is caused by the halving event, which reduces the supply of new Bitcoin by 50% every four years."
}
```

## Validation Checklist

- [ ] All 52+ topics have corresponding test files
- [ ] Each topic has 10 quiz questions
- [ ] Reference answers capture Bob's voice
- [ ] Rejected responses are clearly inferior
- [ ] Critical distinctions (halving vs cycle) are included
- [ ] JSON/JSONL formats are valid

## Output Statistics (Expected)

| Dataset | Records | Size |
|---------|---------|------|
| SFT Training | ~520 | ~2MB |
| Preference Pairs | ~520 | ~3MB |
| Quiz Data | ~52 sets | ~1MB |

## Next Steps

- **DPO branch**: Use preference pairs directly for DPO training
- **PPO branch**: Use preference pairs to train reward model

---

*Phase 1 - Bob Loukas Mindprint RLHF LoRA*
*Branch: shared*

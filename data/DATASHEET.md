# Datasheet: Slipstream Think-Quantize-Transmit Dataset

Following the [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) framework.

---

## Motivation

### For what purpose was the dataset created?

To train language models on the **Slipstream protocol** - a semantic quantization system that reduces multi-agent AI communication costs by 82%. The dataset teaches models the Think-Quantize-Transmit (TQT) cognitive pattern.

### Who created the dataset and on behalf of which entity?

Anthony Maio, Independent Researcher. Created as part of the slipcore open-source project.

### Who funded the creation of the dataset?

Self-funded research project.

---

## Composition

### What do the instances represent?

Each instance is a training example for agent-to-agent communication, containing:
- A natural language instruction (user input)
- A structured response with THOUGHT, QUANTIZE, and SLIP components

### How many instances are there?

2,283 training examples.

### What data does each instance consist of?

```json
{
  "conversations": [
    {"from": "system", "value": "[System prompt about Slipstream]"},
    {"from": "human", "value": "[Natural language instruction]"},
    {"from": "gpt", "value": "THOUGHT: [reasoning]\nQUANTIZE: [dimensions] -> [anchor]\nSLIP: SLIP v1 [src] [dst] [anchor] [payload]"}
  ]
}
```

### Is there a label or target associated with each instance?

Yes, the "gpt" response contains:
- THOUGHT: Natural language reasoning
- QUANTIZE: Semantic dimensions and anchor selection
- SLIP: Wire format output

### Is any information missing from individual instances?

No. All instances are complete.

### Are relationships between individual instances made explicit?

No explicit relationships. Examples are independent.

### Are there recommended data splits?

The dataset is provided as a single training split. For evaluation, we recommend:
- 90% train / 10% validation
- Or use the test cases in `scripts/test_slipstream.py`

### Are there any errors, sources of noise, or redundancies?

- Minor: Some examples have simplified THOUGHT (just SLIP output)
- 78% have full QUANTIZE annotations, 22% have THOUGHT+SLIP only
- 1 malformed example was removed during cleaning

### Is the dataset self-contained?

Yes. No external data dependencies.

---

## Collection Process

### How was the data collected?

Two methods:
1. **Template generation** (`finetune.py`): Programmatic generation from templates
2. **LLM generation** (`finetune_llm.py`): Generated using Claude and Gemini APIs

### What mechanisms were used to collect the data?

- Python scripts with randomized template filling
- LLM API calls with structured output parsing
- Manual review and deduplication

### Who was involved in the data collection process?

Single researcher (Anthony Maio) with LLM assistance.

### Over what timeframe was the data collected?

December 2025.

### Were any ethical review processes conducted?

Not formally. The dataset contains only synthetic agent communication examples with no personal data.

---

## Preprocessing/Cleaning

### What preprocessing was done?

1. Anchor validation (ensuring valid UCR anchors)
2. Format standardization (ShareGPT structure)
3. Deduplication
4. Removal of malformed examples (1 removed)
5. Shuffling

### Was the raw data saved?

Original generated files are preserved in `src/slipcore/train_*.jsonl`.

---

## Uses

### What tasks is the dataset intended for?

- Finetuning LLMs to use the Slipstream protocol
- Teaching Think-Quantize-Transmit cognitive pattern
- Multi-agent communication research

### What tasks should it not be used for?

- Not for general-purpose chatbot training
- Not for tasks requiring factual knowledge
- Not for safety-critical applications without additional validation

### Has the dataset been used for any tasks already?

Yes, to finetune GLM-Z1-9B-0414 (see model card).

---

## Distribution

### How is the dataset distributed?

- **Hugging Face Hub**: `anthony-maio/slipstream-tqt`
- **Kaggle**: `anthonymaio/slipstream-tqt`
- **Zenodo**: With DOI for academic citation
- **GitHub**: In slipcore repository

### When was the dataset released?

January 2025.

### What license is it under?

Apache 2.0

---

## Maintenance

### Who maintains the dataset?

Anthony Maio (anthony@making-minds.ai)

### How can users contribute?

- Open issues on GitHub for errors
- Submit PRs with new examples
- Propose new anchors for domain-specific extensions

### Will the dataset be updated?

Yes, as the UCR evolves and new anchor types are added.

### Are older versions available?

Version history maintained in GitHub.

---

## Additional Information

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total examples | 2,283 |
| With THOUGHT | 100% |
| With QUANTIZE | 78.1% |
| Fallback examples | 182 (8%) |
| Unique anchors | 21 |
| Avg tokens/example | ~150 |

### Anchor Coverage

All 21 core UCR anchors are represented with balanced distribution (3-10% each).

### Contact

- Email: anthony@making-minds.ai
- GitHub: github.com/anthony-maio/slipcore
- Twitter: @[handle]

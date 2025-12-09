from typing import Iterable, Optional, Dict, Any

def get_prompt_template(
    cfg: Dict[str, Any],
    ds: Optional[Iterable] = None,
    test_keys: Optional[Iterable[str]] = None,
    label_keys: Optional[Iterable[str]] = None
) -> str:
    """
    Args
    cfg: config dict. Expected keys:
        - prompt_technique: 'zero-shot' | 'few-shot' | 'chain-of-thought' | 'self-consistency'
        - task_name: one of 'cola','sst-2','mrpc','sts-b','qqp','mnli','qnli','rte','wnli'
        - num_shots: int, examples per label for few-shot. Default 1.
        - label_field: label field name in ds items. Default 'label'
    ds: iterable of dataset items. Each item must expose fields listed in test_keys and the label_field.
    test_keys: names of input text fields. Example for MRPC: ['sentence1','sentence2']
    label_keys: optional explicit label names or values. If omitted they are inferred from ds or from built-in GLUE mapping.

    Returns
    A prompt template string. Contains placeholders for each key in test_keys, e.g. {sentence1}, {sentence2}.
    For classification tasks the prompt includes a "Label meanings" section that tells the model what numeric output to produce.
    For regression (sts-b) the prompt instructs to output a numeric score.
    """

    # helpers
    def read_field(item, key):
        if isinstance(item, dict):
            return item.get(key, "")
        return getattr(item, key, "")

    prompt_technique = cfg.get("prompt_technique", "zero-shot").lower()
    task = cfg.get("task_name", "").lower()
    num_shots = int(cfg.get("num_shots", 1))
    label_field = cfg.get("label_field", "label")

    if not test_keys:
        raise ValueError("test_keys required. Example for MRPC: ['sentence1','sentence2']")

    # canonical GLUE label mappings
    glue_label_defs: Dict[str, Dict[str, Any]] = {
        "cola": {
            "type": "classification",
            "labels": ["unacceptable", "acceptable"],
            "mapping": {"unacceptable": 0, "acceptable": 1},
            "note": "0 means unacceptable, 1 means acceptable"
        },
        "sst-2": {
            "type": "classification",
            "labels": ["negative", "positive"],
            "mapping": {"negative": 0, "positive": 1},
            "note": "0 negative 1 positive"
        },
        "mrpc": {
            "type": "classification",
            "labels": ["not_paraphrase", "paraphrase"],
            "mapping": {"not_paraphrase": 0, "paraphrase": 1},
            "note": "0 not paraphrase 1 paraphrase"
        },
        "qqp": {
            "type": "classification",
            "labels": ["not_duplicate", "duplicate"],
            "mapping": {"not_duplicate": 0, "duplicate": 1},
            "note": "0 not duplicate 1 duplicate"
        },
        "sts-b": {
            "type": "regression",
            "range": (0.0, 5.0),
            "note": "Output a numeric similarity score between 0 and 5"
        },
        "mnli": {
            "type": "classification",
            "labels": ["contradiction", "neutral", "entailment"],
            "mapping": {"contradiction": 0, "neutral": 1, "entailment": 2},
            "note": "0 contradiction 1 neutral 2 entailment"
        },
        "qnli": {
            "type": "classification",
            "labels": ["not_entailment", "entailment"],
            "mapping": {"not_entailment": 0, "entailment": 1},
            "note": "0 not entailment 1 entailment"
        },
        "rte": {
            "type": "classification",
            "labels": ["not_entailment", "entailment"],
            "mapping": {"not_entailment": 0, "entailment": 1},
            "note": "0 not entailment 1 entailment"
        },
        "wnli": {
            "type": "classification",
            "labels": ["false", "true"],
            "mapping": {"false": 0, "true": 1},
            "note": "0 false 1 true"
        }
    }

    # pick label definition for task
    task_def = glue_label_defs.get(task)
    if task_def is None:
        # fallback generic binary classification if user provided label_keys
        if label_keys:
            labels = list(label_keys)
            mapping = {lab: i for i, lab in enumerate(labels)}
            task_def = {"type": "classification", "labels": labels, "mapping": mapping,
                        "note": "Labels mapped to integers in order provided"}
        else:
            raise ValueError("Unknown task_name and no label_keys provided")

    # construct label meaning lines
    def build_label_meanings():
        lines: List[str] = []
        if task_def["type"] == "regression":
            low, high = task_def["range"]
            lines.append("Label meanings:")
            lines.append(task_def.get("note", f"Output a number between {low} and {high}"))
            lines.append("")  # blank line
            return lines

        # classification
        mapping = task_def["mapping"]
        lines.append("Label meanings:")
        # ensure stable order: sort by mapped value
        inv = {v: k for k, v in mapping.items()}
        for i in sorted(inv.keys()):
            lines.append(f"{i}: {inv[i]}")
        lines.append("")  # blank line
        return lines

    # few-shot example collection
    def collect_examples_by_label():
        examples_by_label = {}
        if task_def["type"] == "regression":
            return {}
        labels = task_def["labels"]
        for lbl in labels:
            examples_by_label[lbl] = []
        if ds is None:
            # return placeholders
            for lbl in labels:
                while len(examples_by_label[lbl]) < num_shots:
                    examples_by_label[lbl].append([f"Example text for {k}" for k in test_keys])
            return examples_by_label

        for item in ds:
            lbl = read_field(item, label_field)
            if lbl in examples_by_label and len(examples_by_label[lbl]) < num_shots:
                texts = [read_field(item, k) for k in test_keys]
                examples_by_label[lbl].append(texts)
            if all(len(v) >= num_shots for v in examples_by_label.values()):
                break

        for lbl in labels:
            while len(examples_by_label[lbl]) < num_shots:
                examples_by_label[lbl].append([f"Example text for {k}" for k in test_keys])

        return examples_by_label

    # prompt assembly helpers
    def add_input_fields(lines: List[str]):
        for k in test_keys:
            lines.append(f"{k}: {{{k}}}")
        return lines

    # build the prompt based on technique
    lines: List[str] = []

    # always include label meanings for classification or regression guidance
    lines.extend(build_label_meanings())

    if prompt_technique == "zero-shot":
        lines.append("Classify the following input.")
        lines = add_input_fields(lines)
        if task_def["type"] == "regression":
            lines.append("Answer with a number only.")
        else:
            lines.append("Answer with the numeric label only.")
        return "\n".join(lines)

    if prompt_technique == "few-shot":
        if task_def["type"] == "regression":
            # few-shot regression: include example numeric scores if available in ds
            lines.append("Given these examples:")
            if ds is not None:
                ex_count = 0
                for item in ds:
                    if ex_count >= num_shots:
                        break
                    for k in test_keys:
                        lines.append(f"{k}: {read_field(item, k)}")
                    score = read_field(item, label_field)
                    lines.append(f"Score: {score}")
                    lines.append("")
                    ex_count += 1
                if ex_count < num_shots:
                    for _ in range(num_shots - ex_count):
                        for k in test_keys:
                            lines.append(f"{k}: Example text for {k}")
                        lines.append("Score: 3.0")
                        lines.append("")
            else:
                for _ in range(num_shots):
                    for k in test_keys:
                        lines.append(f"{k}: Example text for {k}")
                    lines.append("Score: 3.0")
                    lines.append("")
            lines.append("Now rate the following input. Answer with a number only between 0 and 5.")
            lines = add_input_fields(lines)
            return "\n".join(lines)

        # classification few-shot
        examples_by_label = collect_examples_by_label()
        lines.append("Given these examples:")
        ex_idx = 1
        for lbl in task_def["labels"]:
            for texts in examples_by_label[lbl]:
                lines.append(f"Example {ex_idx}:")
                for key, txt in zip(test_keys, texts):
                    lines.append(f"{key}: {txt}")
                # show the numeric label to the model so it learns the mapping
                numeric = task_def["mapping"][lbl]
                lines.append(f"Label: {numeric}")
                lines.append("")  # blank line
                ex_idx += 1

        lines.append("Now classify the following input.")
        lines = add_input_fields(lines)
        lines.append("Answer with the numeric label only.")
        return "\n".join(lines)

    if prompt_technique in ("chain-of-thought", "chain_of_thought", "cot"):
        if task_def["type"] == "regression":
            lines.append("Explain your reasoning step by step. Then give the numeric score on one line.")
            lines = add_input_fields(lines)
            lines.append("Answer:")
            return "\n".join(lines)
        lines.append("Explain your reasoning step by step. After reasoning give the final numeric label on a single line.")
        lines = add_input_fields(lines)
        lines.append("Answer:")
        return "\n".join(lines)

    if prompt_technique in ("self-consistency", "self_consistency"):
        if task_def["type"] == "regression":
            lines.append("Think step by step and show your chain of thought. Then give the numeric score on one line.")
            lines = add_input_fields(lines)
            lines.append("Answer:")
            return "\n".join(lines)
        lines.append("Think step by step and show your chain of thought. Then give the final numeric label on a single line.")
        lines = add_input_fields(lines)
        lines.append("Answer:")
        return "\n".join(lines)

    raise ValueError(f"Unknown prompt_technique: {prompt_technique}")

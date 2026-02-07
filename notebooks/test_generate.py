import sys
from pathlib import Path

root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

import json
import gzip
import os
from pathlib import Path
from tqdm import tqdm
from syncode import Syncode

MODEL_NAME = "microsoft/phi-2"
LANG_MAP = {
    "js": "javascript",
    "py": "python",
    "java": "java",
    "go": "go",
    "c": "c"
}

def run_syncode_benchmark(dataset_path, lang_code, output_base="syncode_results"):
    """
    dataset_path: MultiPL-E prompts, eg. humaneval-js.jsonl
    lang_code: abbrevation like js, py, rs, c, java, go
    output_base: base directory to save the results
    """
    grammar = LANG_MAP.get(lang_code)
    if not grammar:
        raise ValueError(f"Not supported language: {lang_code}")

    syn_llm = Syncode(
        model=MODEL_NAME, 
        mode='grammar_mask', 
        grammar=grammar, 
        parse_output_only=True
    )

    # MultiPL-E output directory structure: {dataset_name}-{model_name}-syncode/{task_id}.json.gz
    dataset_name = Path(dataset_path).stem
    sub_dir = f"{dataset_name}-{MODEL_NAME.replace('/', '_')}-syncode"
    save_path = Path(output_base) / sub_dir
    save_path.mkdir(parents=True, exist_ok=True)

    with open(dataset_path, 'r', encoding='utf-8') as f:
        tasks = [json.loads(line) for line in f]

    print(f"Start processing dataset: {dataset_name}, total {len(tasks)} tasks...")

    for task in tqdm(tasks):
        task_id = task['name']
        prompt_text = task['prompt']
        
        completion = syn_llm.infer(prompt_text)[0]
        
        # Output format compatible with MultiPL-E evaluation script
        output_data = {
            "name": task_id,
            "language": lang_code,
            "prompt": prompt_text,
            "completions": [completion]
        }
        
        # Saved to .json.gz
        file_name = f"{task_id.replace('/', '_')}.json.gz"
        with gzip.open(save_path / file_name, 'wt', encoding='utf-8') as zf:
            json.dump(output_data, zf)

    print(f"\nFinished! Results saved to: {save_path}")

if __name__ == "__main__":
    run_syncode_benchmark("D:\\code\\constrained-llm-js-codegen\\benchmark\\MultiPL-E\\datasets\\js_prompts.jsonl", "js")
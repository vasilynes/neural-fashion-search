from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List
import pandas as pd
import os
import time
import json
from tqdm.notebook import tqdm
from pathlib import Path
import re

SAVE_DIR = os.environ.get('CACHE_SAVE_DIR')
if SAVE_DIR:
    BASE_DIR = Path(SAVE_DIR)
else:
    BASE_DIR = Path(__file__).parent

def _checkpoint_paths(filename: str):
    return BASE_DIR / f'{filename}.json', BASE_DIR / f'{filename}.tmp'
 
def _load_cache(filename: str) -> List:
    path, _ = _checkpoint_paths(filename)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return []
 
def _save_cache(data: List, filename: str):
    path, tmp = _checkpoint_paths(filename)
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)
 
def _clean_cache(filename: str):
    path, _ = _checkpoint_paths(filename)
    if path.exists():
        os.remove(path)
 

def load_query_model(model_name: str):
    """Load the model for query processing"""
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left' # for processing in batches

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype=torch.float16, 
        device_map='auto'
    )

    return model, tokenizer

def create_prompt(description: str, tokenizer) -> str:
    """
    Create a formatted prompt to extract query
    chain from a single description
    """
    messages = [
        {'role': 'system', 'content': """You are an expert e-commerce search query generator.
        Create a chain of 6 search queries that progressively add attributes to form a logical sequence.
        Each query must be a valid search phrase that includes ALL previous attributes plus one new one.
        Never drop attributes once added."""},
        {'role': 'user', 'content': f"""Based on this product description, create a chain of 6 search queries.

        Product description: "{description}"

        Create a chain where:
        1. Start with the most basic attribute (material + product type)
        2. Each subsequent query adds ONE new important attribute
        3. NEVER remove attributes from previous queries
        4. End with a comprehensive query capturing all key features

        Format: Return exactly 6 lines, each a complete search query.
        No numbers, no explanations.

        Example format for a dress:
        cotton dress
        cotton dress with sleeves
        cotton dress with long sleeves
        cotton dress with long sleeves and V-neck
        cotton dress with long sleeves and V-neck in floral
        cotton dress with long sleeves and V-neck in floral print

        Now generate for this product:"""}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

def extract_chains_batch(descriptions: List[str], batch_size: int, model, tokenizer) -> pd.DataFrame:
    """Extract chains from descriptions in batches"""
    cache_name = 'chains'
    chains = _load_cache(cache_name)
    
    if chains:
        processed = {record['detail_desc'] for record in chains}
        descriptions = [d for d in descriptions if d not in processed]
        print(f"There are {len(processed)} descriptions.")
        print(f"Resuming to process {len(descriptions)} more...")
    
    time_stamp = time.time()
    for i in tqdm(range(0, len(descriptions), batch_size), desc='Processing batches'):
        batch_descs = descriptions[i:i+batch_size]
        batch_prompts = [create_prompt(desc, tokenizer) for desc in batch_descs]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2000
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        for j, output in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            response = tokenizer.decode(output[input_length:], skip_special_tokens=True)

            for l, q in enumerate(response.strip().split('\n')):
                q = q.strip()
                if not q or len(q.split()) < 2 or len(q) < 5:
                    q = None
                chains.append({
                        'detail_desc': descriptions[i + j],
                        'length': l,
                        'query': q,
                })

        if time.time() - time_stamp >= 20 * 60:
            time_stamp = time.time()
            _save_cache(chains, cache_name)
    
    _clean_cache(cache_name)
    return pd.DataFrame(chains)

def create_prompt_eval(query: str, description: str, tokenizer) -> str:
    """
    Create a formatted prompt to evaluate description 
    with respect to query
    """
    messages = [
        {"role": "system", "content": "You are an expert evaluator for product search relevance. Answer only YES or NO."},
        {"role": "user", "content": f"""Determine if this caption is relevant to the query.

        Query: {query}
        Caption: {description}

        Is this caption relevant? Answer YES or NO."""}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

def evaluate_retrieval(query_description_pairs, batch_size: int, model, tokenizer):
    """Evaluate retrieval relevance in batches"""
    cache_name = 'retrieval'
    results = _load_cache(cache_name)
 
    if results:
        processed = {(r['id'], r['description']) for r in results}
        query_description_pairs = [
            (idx, q, d) for idx, q, d in query_description_pairs
            if (idx, d) not in processed
        ]
        print(f"Resuming from {len(processed)} already evaluated pairs...")
 
    time_stamp = time.time()
    for i in tqdm(range(0, len(query_description_pairs), batch_size), desc='Evaluating retrieval'):
        batch = query_description_pairs[i:i+batch_size]
 
        prompts = [create_prompt_eval(q, d, tokenizer) for _, q, d in batch]
 
        inputs = tokenizer(
            prompts, 
            return_tensors='pt', 
            padding=True,
            truncation=True,
        ).to(model.device)
 
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            
        for j, (idx, query, description) in enumerate(batch):
            input_length = inputs['input_ids'][j].shape[0]
            first_token_id = outputs[j][input_length].item()
            first_token = tokenizer.decode([first_token_id], skip_special_tokens=True).strip().upper()
 
            if first_token == 'YES':
                r = 1
            elif first_token == 'NO':
                r = 0
            else:
                r = None
                
            results.append({
                'id': idx,
                'query': query,
                'description': description,
                'relevance': r
            })
 
        if time.time() - time_stamp >= 20 * 60:
            time_stamp = time.time()
            _save_cache(results, cache_name)
 
    _clean_cache(cache_name)
    return results

def create_prompt_queries(description: str, tokenizer) -> str:
    messages = [
        {'role': 'system', 'content': """You are an expert e-commerce search analyst. Your task is to generate synthetic search queries based on a given product description.

CRITICAL RULES - YOU MUST FOLLOW EXACTLY:

1. **TOTAL QUERIES**: Generate EXACTLY 8 queries total - NO MORE, NO LESS
2. **CATEGORY COUNTS**: EXACTLY 2 queries per category (typos, synonyms, attribute_swaps, real_user)
3. **OUTPUT FORMAT**: Return ONLY raw JSON - no markdown blocks, no explanations, no extra text

CATEGORY-SPECIFIC RULES:

## typos (2 queries)
- Take core words from description
- Add 1-2 realistic keyboard slips (adjacent keys: u→i, m→n, s→a)
- Use phonetic misspellings (lightweight→liteweight, fabric→fabrik)
- Keep same vocabulary, same word order
- DO NOT change meaning or remove words entirely

## synonyms (2 queries)
- Replace nouns and adjectives with EXACT synonyms
- KEEP the same level of detail (don't simplify)
- KEEP the same garment type (don't change "shorts" to "pants" or "dress" to "top")
- KEEP all attributes (color, fit, material, features)
- Example: "slim fit" → "tailored fit" (good), "slim fit" → "fit" (bad - lost detail)

## attribute_swaps (2 queries)
- Change EXACTLY ONE attribute per query (ONLY ONE!)
- Possible attributes: sleeve length, fit, color, material, neckline, length, closure type, pocket type
- Leave EVERYTHING ELSE identical to original description
- Valid: "short sleeves" → "long sleeves" (only sleeve length changed)
- Invalid: "short sleeves" → "long sleeves with pockets" (changed 2 things)
- Invalid: "blue cotton dress" → "red silk dress" (changed color AND material)

## real_user (2 queries)
- Write like a lazy, rushed mobile shopper
- NO punctuation (no periods, commas, question marks)
- Drop filler words (a, an, the, and, of, with)
- Use abbreviations (btn for button, w/ for with, drss for dress)
- Focus on only 2-3 most important features
- Examples: "slim fit drss" "blue coton shirt long sleev"

CRITICAL NEGATIVE EXAMPLES - DO NOT DO THESE:

 WRONG attribute_swap: "short loose dress with wide straps" (changed fit AND strap width)
 CORRECT attribute_swap: "long fitted dress with narrow straps" (only changed length)

 WRONG synonym: "pants with elastic" (original: "shorts with elastic drawstring" - changed garment type)
 CORRECT synonym: "shorts with elasticated drawstring waist" (preserves all details)

 WRONG real_user: "Slim fit dress with pockets." (has punctuation, too clean)
 CORRECT real_user: "slim fit drss pckts" (no punctuation, lazy spelling)

 WRONG count: 3 typos or 1 synonym (must be exactly 2 per category)

Your output must be valid JSON. No exceptions."""},
        {'role': 'user', 'content': f"""EXAMPLE OF CORRECT OUTPUT:

Input description:
"Long-sleeved shirt in crisp cotton poplin with a turn-down collar, classic front and a yoke at the back. Slim fit with narrow shoulders and a tailored waist."

Correct output:
{{
    "typos": [
        "long sleved shirt crisp coton poplin",
        "slim fit norrow sholders"
    ],
    "synonyms": [
        "long-sleeve button-up in stiff cotton poplin with a turn-down collar and classic front",
        "tailored fit shirt with narrow shoulders and a fitted waist"
    ],
    "attribute_swaps": [
        "short-sleeved shirt in crisp cotton poplin with a turn-down collar",
        "long-sleeved shirt in crisp cotton poplin with a button-down collar"
    ],
    "real_user": [
        "slim fit dress shirt long sleev",
        "cotton poplin btn up"
    ]
}}

Notice:
- EXACTLY 2 items per category (total 8)
- Synonyms preserve all original details
- Attribute swaps change ONLY ONE thing each
- Real user has no punctuation and lazy spelling

NOW GENERATE QUERIES FOR THIS PRODUCT DESCRIPTION:

{description}

Remember: EXACTLY 8 total queries, 2 per category. Return ONLY valid JSON."""}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

def extract_queries_batch(descriptions: List[str], batch_size: int, model, tokenizer) -> pd.DataFrame:
    cache_name = 'queries'
    queries = _load_cache(cache_name)
    
    if queries:
        processed = {record['detail_desc'] for record in queries}
        descriptions = [d for d in descriptions if d not in processed]
        print(f"There are {len(processed)} descriptions.")
        print(f"Resuming to process {len(descriptions)} more...")
    
    time_stamp = time.time()
    for i in tqdm(range(0, len(descriptions), batch_size), desc='Processing batches'):
        batch_descs = descriptions[i:i+batch_size]
        batch_prompts = [create_prompt_queries(desc, tokenizer) for desc in batch_descs]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2000
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.6,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )

        for j, output in enumerate(outputs):
            input_length = inputs['input_ids'][j].shape[0]
            response = tokenizer.decode(output[input_length:], skip_special_tokens=True)
            queries.append({
                'detail_desc': descriptions[i + j],
                'raw_json': response.strip()
            })

        if time.time() - time_stamp >= 20 * 60:
            time_stamp = time.time()
            _save_cache(queries, cache_name)
    
    _clean_cache(cache_name)
    return pd.DataFrame(queries)

def clean_and_parse(json_str):
    try:
        cleaned = json_str.replace('\n', '').replace('\t', '')
        cleaned = re.sub(r' +', ' ', cleaned)
        cleaned = re.sub(r',\s*\]', ']', cleaned)
        cleaned = re.sub(r',\s*\}', '}', cleaned)
        return json.loads(cleaned)
    except:
        return None

#!/usr/bin/env python3
"""Fetch DeepSeek-V4-Pro model specs from HuggingFace API."""
import requests
import json

MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro"
API_URL = f"https://huggingface.co/api/models/{MODEL_ID}"

def main():
    resp = requests.get(API_URL)
    if resp.status_code != 200:
        print(f"HTTP {resp.status_code}: {resp.text[:500]}")
        return

    data = resp.json()

    print(f"Model: {data.get('modelId', 'N/A')}")
    print(f"Pipeline: {data.get('pipeline_tag', 'N/A')}")
    print(f"Downloads: {data.get('downloads', 0):,}")
    print(f"Likes: {data.get('likes', 0)}")
    print()

    # Config
    config = data.get('config', {})
    if config:
        from pprint import pprint
        print("=== CONFIG ===")
        for k in sorted(config):
            v = config[k]
            if isinstance(v, (dict, list)):
                print(f"  {k}: {json.dumps(v)[:200]}")
            else:
                print(f"  {k}: {v}")
        print()
    else:
        print("No config in API response.")
        print(f"Keys: {list(data.keys())[:20]}")
        print()

    # Siblings (files)
    siblings = data.get('siblings', [])
    if siblings:
        total_size = 0
        safetensor_files = []
        for s in siblings:
            fname = s.get('rfilename', '')
            fsize = s.get('size', 0)
            total_size += fsize
            if fname.endswith('.safetensors'):
                safetensor_files.append((fname, fsize))
            if fname in ('config.json', 'tokenizer.json', 'tokenizer_config.json'):
                print(f"  [{fname}] -> {fsize:,} bytes")

        print(f"\n  Total files: {len(siblings)}")
        print(f"  Total size: {total_size / 1e9:.2f} GB")
        print(f"  Safetensors files: {len(safetensor_files)}")
        if safetensor_files:
            total_sf = sum(sz for _, sz in safetensor_files)
            print(f"  Safetensors total: {total_sf / 1e9:.2f} GB")
        print()

    # Card metadata (model card)
    card = data.get('cardData', {})
    if card:
        print("=== CARD DATA ===")
        for k in sorted(card):
            if k not in ('language', 'license', 'datasets'):
                print(f"  {k}: {str(card[k])[:300]}")
    else:
        print("No cardData.")

if __name__ == "__main__":
    main()
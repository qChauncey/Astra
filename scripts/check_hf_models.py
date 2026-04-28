#!/usr/bin/env python3
"""Query HuggingFace API for DeepSeek V4 / V4 Pro model info."""
import sys

def main():
    try:
        from huggingface_hub import list_models
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    searches = ["deepseek-v4", "DeepSeek-V4", "deepseek-v4-pro", "DeepSeek-V4-Pro"]

    for search in searches:
        print(f"\n{'='*60}")
        print(f"Search: {search}")
        print(f"{'='*60}")
        try:
            models = list(list_models(search=search, limit=10))
            if not models:
                print("  No results.")
                continue
            for m in models:
                print(f"  {m.modelId}")
                print(f"    downloads={m.downloads:,}  likes={m.likes}")
                if hasattr(m, 'pipeline_tag'):
                    print(f"    pipeline={m.pipeline_tag}")
                if hasattr(m, 'siblings'):
                    # Count safetensors files and estimate size
                    safetensor_files = [s for s in m.siblings if s.rfilename.endswith('.safetensors')]
                    print(f"    safetensors files: {len(safetensor_files)}")
        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    main()
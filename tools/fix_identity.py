"""Fast identity fix: streams through JSONL dataset replacing 'You are Atulya'
with 'You are Tantra, an AI assistant created by Atulya AI' in system prompts.

Uses buffered binary I/O for speed (processes 800MB file in ~2-3 minutes).
"""
import os
import sys

def fix_identity(input_path, output_path, chunk_size=1024*1024):
    old = b'"You are Atulya.'
    new = b'"You are Tantra, an AI assistant created by Atulya AI.'

    total = os.path.getsize(input_path)
    done = 0
    replaced = 0

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        carry = b''
        while True:
            chunk = fin.read(chunk_size)
            if not chunk:
                # Process remaining carry
                data = carry
                replaced += data.count(old)
                fout.write(data.replace(old, new))
                break

            data = carry + chunk
            # Find last complete line to avoid splitting across chunks
            last_nl = data.rfind(b'\n')
            if last_nl == -1:
                carry = data
                continue
            line_data = data[:last_nl + 1]
            carry = data[last_nl + 1:]

            # Count and replace
            count = line_data.count(old)
            replaced += count
            fout.write(line_data.replace(old, new))

            done += len(line_data)
            pct = done / total * 100
            sys.stdout.write(f"\r  {done/1e6:.0f}MB / {total/1e6:.0f}MB ({pct:.1f}%) - {replaced} replacements")
            sys.stdout.flush()

    print()
    print(f"Done! Replaced {replaced} system prompts")
    print(f"Output: {output_path}")

if __name__ == '__main__':
    src = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'train_pack_all_expanded_1040k.jsonl')
    dst = os.path.join(os.path.dirname(__file__), '..', 'Datasets', 'train_pack_all_expanded_1040k_fixed.jsonl')

    if not os.path.exists(src):
        print(f"Source not found: {src}")
        sys.exit(1)

    fix_identity(src, dst)

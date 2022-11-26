import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_fn', type=str, default='../raw_data/figer2coarse_typemap.json')
parser.add_argument('-o', '--output_fn', type=str, default='../raw_data/coarse_typeset.txt')

args = parser.parse_args()

with open(args.input_fn, 'r', encoding='utf8') as f:
    typemap = json.load(f)
    typeset = set(typemap.values())
    print(f"Number of types: {len(typeset)}")
    with open(args.output_fn, 'w', encoding='utf8') as ofp:
        # json.dump(sorted(list(typeset)), ofp, ensure_ascii=False, indent=4)
        for t in sorted(list(typeset)):
            assert t[0] == '/'
            ofp.write(t[1:] + '\n')

print("Done.")
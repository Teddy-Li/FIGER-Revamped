import json

import entity_pb2 as protc
import argparse
from google.protobuf.internal.encoder import _VarintBytes
from google.protobuf.internal.decoder import _DecodeVarint32
import time


def main(input_fn, output_fn):
    ofp = open(output_fn, 'w', encoding='utf8')

    with open(input_fn, 'rb') as f:
        buf = f.read()
        n = 0
        itm_cnter = 0
        while n < len(buf):
            if itm_cnter % 10000 == 0:
                print(itm_cnter)
            msg_len, new_pos = _DecodeVarint32(buf, n)
            # print(f"msg_len: {msg_len}; new_pos: {new_pos}")
            n = new_pos
            msg_buf = buf[n:n + msg_len]
            n += msg_len
            curr_mention = protc.Mention()
            curr_mention.ParseFromString(msg_buf)
            # print(curr_mention)
            json_item = {
                'id': str(itm_cnter),
                'start': curr_mention.start,
                'end': curr_mention.end,
                'tokens': list(curr_mention.tokens),
                'pos_tags': list(curr_mention.pos_tags),
                'deps': [{'type': x.type, 'gov': x.gov, 'dep': x.dep} for x in curr_mention.deps],
                'entity_name': curr_mention.entity_name,
                'labels': list(curr_mention.labels),
                'sentid': curr_mention.sentid,
                'fileid': curr_mention.fileid
            }
            out_line = json.dumps(json_item, ensure_ascii=False)
            itm_cnter += 1
            ofp.write(out_line+'\n')

    print(f"Total number of entries: {itm_cnter}.")
    ofp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fn', type=str, default='../raw_data/train.data')
    parser.add_argument('--output_fn', type=str, default='../json_data/all.json')
    args = parser.parse_args()
    main(args.input_fn, args.output_fn)

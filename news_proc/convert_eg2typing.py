import json
import random
import os
import argparse
from typing import List, Dict, Tuple, Set, Optional


def find_match_from_lst(whole_words: List[str], lst: List[str], predicate_anchor_idx, head_tok: str = None):
    if head_tok is not None:
        head_aid = lst.index(head_tok)
    else:
        head_aid = None

    arg_whw_beam = {}
    for wwid, whw in enumerate(whole_words):
        arg_whw_beam_last_keys = list(arg_whw_beam.keys())
        for aidx, a in enumerate(lst):
            if a in whw:
                if f'{aidx}' not in arg_whw_beam:
                    arg_whw_beam[f'{aidx}'] = {}
                arg_whw_beam[f'{aidx}'][f'{wwid}'] = abs(wwid - predicate_anchor_idx)
                # for all the keys that are already there before adding in the concatenations
                for k in arg_whw_beam_last_keys:
                    key_lst = k.split('#')
                    if int(key_lst[-1]) < aidx:
                        new_k = '#'.join(key_lst + [f'{aidx}'])
                        if new_k not in arg_whw_beam:
                            arg_whw_beam[new_k] = {}
                        if len(arg_whw_beam) > 262144:
                            print(f"find_match_from_lst: len combs > 262144: {len(arg_whw_beam)}; lst: {lst}; whole_words: {whole_words};")

                        k_last_combs = list(arg_whw_beam[k].keys())

                        for comb in k_last_combs:
                            comb_last_idx = int(comb.split('#')[-1])
                            comb_first_idx = int(comb.split('#')[0])
                            assert comb_last_idx <= wwid
                            if comb_last_idx < wwid:  # namely, if this instance is not newly added with this whw token
                                assert comb + f'#{wwid}' not in arg_whw_beam[new_k]
                                if wwid - comb_first_idx > 40:
                                    continue

                                arg_whw_beam[new_k][comb + f'#{wwid}'] = arg_whw_beam[k][comb] + abs(
                                    wwid - predicate_anchor_idx)

                                # if the current token is consecutive to the previous one both in terms of the sentence
                                # and in terms of the predicate/argument, we are certain that they are a combination, and
                                # can safely delete the shorter version.
                                if comb_last_idx == wwid-1 and int(key_lst[-1]) == aidx-1:
                                    del arg_whw_beam[k][comb]
                            if len(arg_whw_beam[new_k]) > 2000:
                                print(f"find_match_from_lst: len instances > 2000: {len(arg_whw_beam[new_k])}; new_k: {new_k}; lst: {lst}; whole_words: {whole_words};")

        arg_whw_beam_cur_keys = list(arg_whw_beam.keys())
        for aidxes in arg_whw_beam_cur_keys:
            if len(arg_whw_beam[aidxes]) == 0:
                del arg_whw_beam[aidxes]

        if len(arg_whw_beam) > 131072:
            arg_whw_beam = {k: v for (k, v) in sorted(arg_whw_beam.items(), key=lambda x: len(x[0].split('#')), reverse=True)[:8192]}
            print(
                f"find_match_from_lst: len combs > 131072: {len(arg_whw_beam)}; lst: {lst}; whole_words: {whole_words};")

        # cut beam to size 100 whenever its size exceeds 1000, so as to limit the memory usage.
        for aidxes in arg_whw_beam:
            if len(arg_whw_beam[aidxes]) > 1000:
                arg_whw_beam[aidxes] = {k: v for (k, v) in
                                        sorted(arg_whw_beam[aidxes].items(), key=lambda x: x[1])[:100]}
                print(f"find_match_from_lst: len instances > 1000: {len(arg_whw_beam[aidxes])};")

    arg_whw_beam = {k: v for (k, v) in sorted(arg_whw_beam.items(), key=lambda x: len(x[0].split('#')), reverse=True)}
    arg_whw_idxes = None
    head_whw_idx = None
    if len(arg_whw_beam) == 0:
        pass
    else:
        # check if any chain contains the head (if the head is specified); if the head is present in some chains, then
        # we discard all the other chains with the head missing; otherwise, never mind the head.
        if head_aid is None:
            head_missing_from_all_chains = False
        else:
            head_missing_from_all_chains = True
            for aidxes_k in arg_whw_beam:
                aidxes_lst = [int(x) for x in aidxes_k.split('#')]
                if head_aid in aidxes_lst:
                    head_missing_from_all_chains = False
                    break

        for aidxes_k in arg_whw_beam:
            aidxes_lst = [int(x) for x in aidxes_k.split('#')]
            # If the head_tok is specified, then that head_tok must be present in the returned whole word id list.
            if head_aid is not None:
                if not head_missing_from_all_chains and head_aid not in aidxes_lst:
                    continue
                elif head_missing_from_all_chains:
                    head_id_in_subsequence = None
                else:
                    head_id_in_subsequence = aidxes_lst.index(head_aid)
            else:
                head_id_in_subsequence = None

            # select the one instance of this chain with minimum accumulated distance
            instances = {k: v for (k, v) in sorted(arg_whw_beam[aidxes_k].items(), key=lambda x: x[1])}

            for wwid_k in instances:
                arg_whw_idxes = [int(x) for x in wwid_k.split('#')]
                assert arg_whw_idxes[-1] >= arg_whw_idxes[0]
                assert len(arg_whw_idxes) == len(aidxes_lst)
                if head_id_in_subsequence is not None:
                    head_whw_idx = arg_whw_idxes[head_id_in_subsequence]
                break
            break

    if arg_whw_idxes is None:
        if len(lst) > 1:
            abbrv = ''.join([x[0] for x in lst])
            for wwid, whw in enumerate(whole_words):
                if abbrv in whw:
                    arg_whw_idxes = [wwid]
                    # print(f"find_match_from_lst Backoff abbreviation: {lst} to {abbrv};")
                    break
        elif len(lst) == 1:
            for wwid, whw in enumerate(whole_words):
                for ww_inst in whw:
                    if lst[0] in ww_inst and len(lst[0]) > 3 and (len(ww_inst) - len(lst[0])) <= 2:
                        # print(f"find_match_from_lst Backoff partial match: {lst[0]} to {ww_inst};")
                        arg_whw_idxes = [wwid]
                        break
                if arg_whw_idxes is not None:
                    break
        else:
            print(f"find_match_from_lst Warning: the pattern to match is empty!")

        assert head_whw_idx is None
        # if after the back-off, still there is no match

    assert arg_whw_idxes is not None and len(arg_whw_idxes) > 0
    if head_whw_idx is None:
        head_whw_idx = arg_whw_idxes[0]

    return arg_whw_idxes, head_whw_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_fn', type=str, default=None)
    parser.add_argument('--out_fn', type=str, default=None)

    args = parser.parse_args()

    with open(args.in_fn, 'r', encoding='utf8') as f:
        for lidx, line in enumerate(f):
            if lidx % 10000 == 0:
                print(f"lidx: {lidx}")
            item = json.loads(line)
            item['id'] = lidx
            out_line = json.dumps(item, ensure_ascii=False)
            args.out_fn.write(out_line+'\n')
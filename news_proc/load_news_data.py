import copy
import json
import os
import random
import argparse
import numpy as np
import transformers
import subprocess
from nltk.stem import WordNetLemmatizer
import nltk
from transformers import BertTokenizer


"""
Below: borrowed from ``/Users/teddy/PycharmProjects/NE_pipeline/CLPEG_reformat_utils.py''
"""
fmfl_span_not_found_cnt = 0
sentid_outofrange_cnt = 0
arg_complete_absence_cnt = 0
args_discontinuous_cnt = 0
args_abbrv_cnt = 0
arg_toolong_cnt = 0
args_total_cnt = 0

lemmatizer = WordNetLemmatizer()


def parse_urel(rel, mode: str):
    assert mode in ['ns', 'nc']
    rel = rel["r"]
    assert rel[0] == '(' and rel[-1] == ')'
    rel = rel[1:-1]
    rel_list = rel.split('::')
    if mode == 'ns':
        assert len(rel_list) == 6
        upred = rel_list[0]
        subj = rel_list[1]
        obj = rel_list[2]
        inpara_subsentid = int(rel_list[4])
        pred_head_anchor_idx = int(rel_list[5])
        return upred, subj, obj, inpara_subsentid, pred_head_anchor_idx
    elif mode == 'nc':
        assert len(rel_list) == 7
        upred = rel_list[0]
        subj = rel_list[1]
        obj = rel_list[2]
        inpara_subsentid = 0
        pred_idxes = rel_list[6].split('_')
        try:
            pred_head_anchor_idx = sum([int(x) for x in pred_idxes]) / len(pred_idxes)  # this thing can be float!
        except ValueError as e:
            print(f"Error in parsing {rel_list[6]} in {rel}")
            pred_head_anchor_idx = 0
        return upred, subj, obj, inpara_subsentid, pred_head_anchor_idx
    else:
        raise NotImplementedError


def build_wholewords_from_bert_tokens(bert_tokens):
    whole_words = []
    whole_words_inbert_idxes = []
    pref = None
    accumulated_inbert_idxes = []
    for btid, tok in enumerate(bert_tokens):
        if tok.startswith('##'):
            assert len(tok) > 2
            pref = pref + tok.lstrip('##')
        # otherwise, it is either a prefix or a standalone word, but we don't know that yet! Therefore, we put that in
        # to the prefix bucket, and clear the previous prefix bucket out! (Remember that we have one last word in the
        # prefix cache at the end of each ``for'' loop.)
        else:
            if pref is not None:
                whole_words.append(pref)
                assert len(accumulated_inbert_idxes) > 0
                whole_words_inbert_idxes.append(accumulated_inbert_idxes)
            pref = tok
            accumulated_inbert_idxes = []

        accumulated_inbert_idxes.append(btid)
    if pref is not None:
        whole_words.append(pref)
        assert len(accumulated_inbert_idxes) > 0
        whole_words_inbert_idxes.append(accumulated_inbert_idxes)

    return whole_words, whole_words_inbert_idxes


def lemmatize_whole_words(whole_words):
    for wwid in range(len(whole_words)):
        raw_word = whole_words[wwid]
        lemm_set = [raw_word]
        for pt in ['a', 's', 'r', 'n', 'v']:
            lemm = lemmatizer.lemmatize(raw_word.lower(), pt)
            if lemm not in lemm_set:
                lemm_set.append(lemm)
        whole_words[wwid] = lemm_set

    return whole_words


def find_match_from_lst(whole_words_orig, lst, predicate_anchor_idx, head_tok=None):
    whole_words = []
    for ww in whole_words_orig:
        whole_words.append([w.lower() for w in ww])

    global fmfl_span_not_found_cnt, args_abbrv_cnt

    if head_tok is not None:
        head_aid = lst.index(head_tok)
    else:
        head_aid = None

    arg_whw_beam = {}
    for wwid, whw in enumerate(whole_words):
        arg_whw_beam_last_keys = list(arg_whw_beam.keys())
        for aidx, a in enumerate(lst):  # for all the tokens in the desired span
            if a.lower() in whw:
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
                        # if len(arg_whw_beam) > 262144:
                        #     print(f"find_match_from_lst: len combs > 262144: {len(arg_whw_beam)}; lst: {lst}; whole_words: {whole_words};")

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
                # print(f"find_match_from_lst: len instances > 1000: {len(arg_whw_beam[aidxes])};")

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

    not_found_flag = False
    if arg_whw_idxes is None:
        if len(lst) > 1:
            abbrv = ''.join([x[0] for x in lst]).lower()
            for wwid, whw in enumerate(whole_words):
                if abbrv in whw and len(whw) == 1:  # if the whole word is an abbreviation, we dictate that there can be only one lemma to exclude false positives
                    arg_whw_idxes = [wwid]
                    args_abbrv_cnt += 1
                    # print(f"find_match_from_lst Backoff abbreviation: {lst} to {abbrv};")
                    break
        elif len(lst) == 1:
            for wwid, whw in enumerate(whole_words):
                for ww_inst in whw:
                    if lst[0].lower() in ww_inst and len(lst[0]) > 3 and (len(ww_inst) - len(lst[0])) <= 2:
                        # print(f"find_match_from_lst Backoff partial match: {lst[0]} to {ww_inst};")
                        arg_whw_idxes = [wwid]
                        break
                if arg_whw_idxes is not None:
                    break
        else:
            print(f"find_match_from_lst Warning: the pattern to match is empty!")

        assert head_whw_idx is None
        # if after the back-off, still there is no match
        if arg_whw_idxes is None:
            fmfl_span_not_found_cnt += 1
            not_found_flag = True
            arg_whw_idxes = []

    assert arg_whw_idxes is not None
    if head_whw_idx is None:
        head_whw_idx = arg_whw_idxes[0] if len(arg_whw_idxes) > 0 else None

    return arg_whw_idxes, head_whw_idx, not_found_flag


"""
Above: borrowed from ``/Users/teddy/PycharmProjects/NE_pipeline/CLPEG_reformat_utils.py''
"""


def build_outentry(argument, whole_words_per_sentence, subsent_id, pred_head_anchor_idx, tokenizer,
                   line_item, in_line_argid: str, backoff_neigh_sents: bool):
    global sentid_outofrange_cnt, arg_complete_absence_cnt, args_total_cnt, args_discontinuous_cnt, arg_toolong_cnt

    argument = argument.replace('_', ' ')
    arg_lst = tokenizer.tokenize(argument)
    arg_lst, _ = build_wholewords_from_bert_tokens(arg_lst)
    arg_lst = [x for x in arg_lst if x not in ['.']]

    total_paragraph = []
    for sent in whole_words_per_sentence:
        total_paragraph += sent

    effective_subsent_id = None
    arg_toolong_flag = False

    lst_isascii = [x.isascii() for x in arg_lst]
    lst_islong = [len(x) > 3 for x in arg_lst]
    if len(arg_lst) >= 5 and sum(lst_isascii) / len(lst_isascii) < 0.4 and sum(lst_islong) / len(lst_islong) < 0.4:
        print(f"build_outentry: Warning: the argument is too long and non-sensical!: {arg_lst}; skipping and using the [CLS] token instead.")
        arg_whw_final_idxes = []
        arg_toolong_flag = True
        not_found_flag = False  # we don't want to trickle down to back-off matching for too long arguments
        effective_subsent_id = 0
        arg_toolong_cnt += 1
    elif 0 <= subsent_id < len(whole_words_per_sentence):
        arg_whw_final_idxes, _, not_found_flag = find_match_from_lst(whole_words_per_sentence[subsent_id], arg_lst,
                                                                    pred_head_anchor_idx, head_tok=None)
    else:
        arg_whw_final_idxes = None
        not_found_flag = True
        sentid_outofrange_cnt += 1

    # we are only separating sentences by '. ' (i.e., a space after the period), so we are in general separating
    # sentences in a coarse way; chances are, when the argument is not found in the designated subsentence index,
    # it is because the argument is actually in the previous sentence, so we try to find it in the previous sentence,
    # and if it is still not found, then we try to find it in the next sentence.
    if not_found_flag and backoff_neigh_sents and 0 <= subsent_id-1 < len(whole_words_per_sentence):
        arg_whw_final_idxes, _, not_found_flag = find_match_from_lst(whole_words_per_sentence[subsent_id-1], arg_lst,
                                                                     pred_head_anchor_idx, head_tok=None)
    elif not not_found_flag and effective_subsent_id is None:  # namely, the argument is found in (subsent_id)
        effective_subsent_id = subsent_id
    else:
        pass

    if not_found_flag and backoff_neigh_sents and 0 <= subsent_id+1 < len(whole_words_per_sentence):
        arg_whw_final_idxes, _, not_found_flag = find_match_from_lst(whole_words_per_sentence[subsent_id+1], arg_lst,
                                                                     pred_head_anchor_idx, head_tok=None)
    elif not not_found_flag and effective_subsent_id is None:  # namely, the argument is found in (subsent_id - 1)
        effective_subsent_id = subsent_id - 1
    else:
        pass

    if not_found_flag and backoff_neigh_sents:
        arg_whw_final_idxes, _, not_found_flag = find_match_from_lst(total_paragraph, arg_lst, pred_head_anchor_idx, head_tok=None)
    elif not not_found_flag and effective_subsent_id is None:  # namely, the argument is found in (subsent_id + 1)
        effective_subsent_id = subsent_id + 1

    if not_found_flag:
        arg_complete_absence_cnt += 1
        arg_whw_final_idxes = []
        assert effective_subsent_id is None
        # namely, we don't know where the argument is, so we just put it in the current sentence
        # if subsent_id >= len(whole_words_per_sentence):
        #     print(f"build_outentry Warning: the subsentence id is out of range: {subsent_id} vs. {len(whole_words_per_sentence)}")
        effective_subsent_id = min(subsent_id, len(whole_words_per_sentence) - 1)
        effective_tokens = whole_words_per_sentence[effective_subsent_id]
    elif arg_toolong_flag:
        arg_complete_absence_cnt += 1
        assert len(arg_whw_final_idxes) == 0 and effective_subsent_id == 0
        effective_tokens = whole_words_per_sentence[effective_subsent_id]
    elif not not_found_flag and effective_subsent_id is None:  # namely, the argument is found somewhere else in the paragraph
        eff_tokens_start = max(min(arg_whw_final_idxes) - 60, 0)
        eff_tokens_end = min(125+eff_tokens_start, len(total_paragraph))
        arg_whw_final_idxes = [x - eff_tokens_start for x in arg_whw_final_idxes]
        effective_tokens = total_paragraph[eff_tokens_start:eff_tokens_end]
        effective_subsent_id = -1
    else:
        assert effective_subsent_id is not None
        effective_tokens = whole_words_per_sentence[effective_subsent_id]

    arg_span = (min(arg_whw_final_idxes), max(arg_whw_final_idxes)) if len(arg_whw_final_idxes) > 0 else None

    effective_tokens = [x[0] for x in effective_tokens]  # use the original tokens, not the lemmatized ones

    if arg_span is not None and arg_span[1] >= len(effective_tokens):
        raise AssertionError(f"build_outentry: the argument span is out of range: {arg_span} vs. {len(effective_tokens)}")

    # The start and end tokens are both inclusive
    out_entry = {
        "id": args_total_cnt,
        "start": arg_span[0] if arg_span is not None else None,
        "end": arg_span[1] if arg_span is not None else None,
        "tokens": effective_tokens,
        "pos_tags": None,
        "deps": None,
        "entity_name": argument,
        "labels": None,
        "subsent_id": effective_subsent_id,
        "lineid": line_item['lineId'],
        "fileid": line_item['articleId'],
        'inline_argid': in_line_argid,
        "figer_ling_labels": None,
        "figer_extended_labels": None,
        "entgraph_labels": None,
    }

    args_total_cnt += 1

    return out_entry


def data_reformat_ns(in_path, out_path, max_seq_len):
    """
    :param in_path:
    :param out_path:
    :param max_seq_len:
    :return:

    Input data format:
        item = {
                "s": xx,
                "date": xx,
                "articleId": xx,
                "lineId": xx,
                "rels": [
                    {
                        "r": "(pred::arg1::arg2::EE::0::XX)",
                    }
                ]
        }
    """

    out_fp = open(out_path, "w", encoding="utf8")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, max_seq_len=max_seq_len)
    total_lines_cnt = 0

    with open(in_path, 'r', encoding='utf8') as f:
        for lidx, line in enumerate(f):
            if lidx % 10000 == 0:
                """
                fmfl_span_not_found_cnt = 0
                sentid_outofrange_cnt = 0
                arg_complete_absence_cnt = 0
                args_discontinuous_cnt = 0
                args_total_cnt = 0
                """
                print(f"lidx: {lidx}; fmfl_span_not_found_cnt: {fmfl_span_not_found_cnt}; "
                      f"sentid_outofrange_cnt: {sentid_outofrange_cnt}; arg_complete_absence_cnt: {arg_complete_absence_cnt}; "
                      f"args_discontinuous_cnt: {args_discontinuous_cnt}; args_abbrv_cnt: {args_abbrv_cnt}; args_total_cnt: {args_total_cnt};")

            item = json.loads(line)

            sents = nltk.tokenize.sent_tokenize(item['s'])  # separate the lines into sentences with nltk
            tokens_per_sentence = [tokenizer.tokenize(x) for x in sents]
            whole_words_per_sent, whole_words_inbert_idxes_per_sent = [], []
            for sentence_tokens in tokens_per_sentence:
                curr_whole_words, curr_whole_words_inbert_idxes = build_wholewords_from_bert_tokens(sentence_tokens)
                curr_whole_words = lemmatize_whole_words(curr_whole_words)
                whole_words_per_sent.append(curr_whole_words)
                whole_words_inbert_idxes_per_sent.append(curr_whole_words_inbert_idxes)

            for ridx, rel in enumerate(item["rels"]):
                upred, subj, obj, subsent_id, pred_head_anchor_idx = parse_urel(rel, mode="ns")
                assert ' ' not in upred

                subj_entry = build_outentry(subj, whole_words_per_sent, subsent_id, pred_head_anchor_idx, tokenizer,
                                            line_item=item, in_line_argid=f"{ridx}_subj", backoff_neigh_sents=True)
                obj_entry = build_outentry(obj, whole_words_per_sent, subsent_id, pred_head_anchor_idx, tokenizer,
                                           line_item=item, in_line_argid=f"{ridx}_obj", backoff_neigh_sents=True)

                subj_line = json.dumps(subj_entry, ensure_ascii=False)
                obj_line = json.dumps(obj_entry, ensure_ascii=False)

                out_fp.write(subj_line+'\n')
                out_fp.write(obj_line+'\n')
            total_lines_cnt += 1

    print(f"Finished! Total statistics as follows: ")
    print(f"Number of lines in corpus: {total_lines_cnt}; fmfl_span_not_found_cnt: {fmfl_span_not_found_cnt}; "
          f"sentid_outofrange_cnt: {sentid_outofrange_cnt}; arg_complete_absence_cnt: {arg_complete_absence_cnt}; "
          f"args_discontinuous_cnt: {args_discontinuous_cnt}; args_total_cnt: {args_total_cnt};")

    out_fp.close()


def data_reformat_nc(in_path, out_path, max_seq_len, reload=False):
    """
    :param in_path:
    :param out_path:
    :param max_seq_len:
    :return:

    Input data format:
        item = {
                "s": xx,
                "date": xx,
                "articleId": xx,
                "lineId": xx,
                "rels": [
                    {
                        "r": "(pred::arg1::arg2::EE::0::XX)",
                    }
                ]
        }
    """

    # if not os.path.exists(out_path) or reload:
    #     out_fp = open(out_path, "w", encoding="utf8")
    # else:
    #     last_line = subprocess.check_output(['tail', '-1', out_path]).decode('utf-8')
    #     last_item = json.loads(last_line)
    #     last_idx = last_item['id']
    #     out_fp = open(out_path, "a", encoding="utf8")

    out_fp = open(out_path, "w", encoding="utf8")
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False, max_seq_len=max_seq_len)
    total_lines_cnt = 0

    with open(in_path, 'r', encoding='utf8') as f:
        for lidx, line in enumerate(f):
            if lidx % 10000 == 0:
                """
                fmfl_span_not_found_cnt = 0
                sentid_outofrange_cnt = 0
                arg_complete_absence_cnt = 0
                args_discontinuous_cnt = 0
                args_total_cnt = 0
                """
                print(f"lidx: {lidx}; fmfl_span_not_found_cnt: {fmfl_span_not_found_cnt}; "
                      f"sentid_outofrange_cnt: {sentid_outofrange_cnt}; arg_complete_absence_cnt: {arg_complete_absence_cnt}; "
                      f"args_discontinuous_cnt: {args_discontinuous_cnt}; args_abbrv_cnt: {args_abbrv_cnt}; args_total_cnt: {args_total_cnt};")
            try:
                item = json.loads(line)
            except json.decoder.JSONDecodeError as e:
                print(f"Error in line {lidx}: {line}")
                continue

            tokens = item['tokens'].split(' ')
            whole_words = lemmatize_whole_words(tokens)

            for ridx, rel in enumerate(item["rels"]):
                upred, subj, obj, _, pred_anchor_idx = parse_urel(rel, mode="nc")
                assert ' ' not in upred

                subj_entry = build_outentry(subj, [whole_words], 0, pred_anchor_idx, tokenizer, line_item=item,
                                            in_line_argid=f"{ridx}_subj", backoff_neigh_sents=False)
                obj_entry = build_outentry(obj, [whole_words], 0, pred_anchor_idx, tokenizer, line_item=item,
                                           in_line_argid=f"{ridx}_obj", backoff_neigh_sents=False)

                subj_line = json.dumps(subj_entry, ensure_ascii=False)
                obj_line = json.dumps(obj_entry, ensure_ascii=False)

                out_fp.write(subj_line+'\n')
                out_fp.write(obj_line+'\n')
            total_lines_cnt += 1

    print(f"Finished! Total statistics as follows: ")
    print(f"Number of lines in corpus: {total_lines_cnt}; fmfl_span_not_found_cnt: {fmfl_span_not_found_cnt}; "
          f"sentid_outofrange_cnt: {sentid_outofrange_cnt}; arg_complete_absence_cnt: {arg_complete_absence_cnt}; "
          f"args_discontinuous_cnt: {args_discontinuous_cnt}; args_total_cnt: {args_total_cnt};")

    out_fp.close()


def split_typing_input(out_path, num_slices, expected_num_lines):
    """
    :param out_path:
    :param num_slices:
    :param expected_num_lines:
    :return:
    """

    assert out_path[-5:] == '.json'

    seperated_outfps = [open(f"{out_path[:-5]}_{i}.json", "w", encoding="utf8") for i in range(num_slices)]
    slice_size = expected_num_lines // num_slices
    print(f"Separating {out_path} into {num_slices} slices of size {slice_size}...")

    empty_line_cnt = 0
    total_line_cnt = 0

    with open(out_path, 'r', encoding='utf8') as jfp:
        for lidx, line in enumerate(jfp):
            if lidx % 100000 == 0:
                print(f"lidx: {lidx}")
            if len(line) == 0:
                empty_line_cnt += 1

            slice_idx = min(lidx // slice_size, num_slices - 1)
            seperated_outfps[slice_idx].write(line)
            total_line_cnt += 1

    print(f"Actual total number of lines: {total_line_cnt}; empty_line_cnt: {empty_line_cnt}")
    for sep_outfp in seperated_outfps:
        sep_outfp.close()

    os.remove(out_path)
    print(f"The {out_path} has been separated into {num_slices} slices of size {slice_size}; original file is deleted. Quitting...")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', type=str, default='en')
    parser.add_argument('--in_path', type=str, default='../../NE_pipeline/news_gen8_p.json')
    parser.add_argument('--out_dir', type=str, default='../news_data/')
    parser.add_argument('--data_name', type=str, default='newsspike')
    parser.add_argument('--out_fn', type=str, default='%s_gparser_typing_input_full.json')
    parser.add_argument('--num_slices', type=int, default=8)
    parser.add_argument('--expected_num_lines', type=int, default=63876006)  # 63876006 for newsSpike; 10000000 for newsCrawl
    parser.add_argument('--mode', type=str, default='load')
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()

    args.out_fn = args.out_fn % args.data_name

    if args.mode == 'load':
        args.out_path = os.path.join(args.out_dir, args.out_fn)
        if args.lang == 'en':
            if args.data_name == 'newsspike':
                data_reformat_ns(args.in_path, args.out_path, max_seq_len=512)
            elif args.data_name == 'newscrawl':
                data_reformat_nc(args.in_path, args.out_path, max_seq_len=512, reload=args.reload)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
    elif args.mode == 'split':
        args.out_path = os.path.join(args.out_dir, args.out_fn)
        if args.lang == 'en':
            split_typing_input(args.out_path, args.num_slices, args.expected_num_lines)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError


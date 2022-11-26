import os
import json
import argparse


def build_entry(subj, pred, obj, lidx, atc, proposition_name):
    subj_lst = subj.split(' ')
    pred_lst = pred.split(' ')
    obj_lst = obj.split(' ')

    tokens = subj_lst + pred_lst + obj_lst + ['.']
    if len(subj_lst) == 0:
        print('subj_lst is empty')
    if len(obj_lst) == 0:
        print('obj_lst is empty')

    subj_entry = {
        "id": atc,
        "start": 0,
        "end": max(len(subj_lst) - 1, 0),
        "tokens": tokens,
        "pos_tags": None,
        "deps": None,
        "entity_name": subj,
        "labels": None,
        "sentid": f"{lidx}_{proposition_name}_subj",
        "fileid": None,
        "figer_ling_labels": None,
        "figer_extended_labels": None,
        "entgraph_labels": None,
    }

    obj_entry = {
        "id": atc+1,
        "start": len(subj_lst) + len(pred_lst),
        "end": len(subj_lst) + len(pred_lst) + max(len(obj_lst) - 1, 0),
        "tokens": tokens,
        "pos_tags": None,
        "deps": None,
        "entity_name": obj,
        "labels": None,
        "sentid": f"{lidx}_{proposition_name}_obj",
        "fileid": None,
        "figer_ling_labels": None,
        "figer_extended_labels": None,
        "entgraph_labels": None,
    }

    return subj_entry, obj_entry, atc+2


def load_levy_data(in_path, out_path):
    args_total_cnt = 0

    ofp = open(out_path, 'w', encoding='utf-8')

    with open(in_path, 'r', encoding='utf8') as ifp:
        for lidx, line in enumerate(ifp):
            line = line.strip('\n')
            if line:
                hypo, prem, label = line.split('\t')
                hypo = hypo.split(',')
                prem = prem.split(',')
                h_subj_entry, h_obj_entry, args_total_cnt = build_entry(hypo[0], hypo[1], hypo[2], lidx, args_total_cnt, 'hypo')
                p_subj_entry, p_obj_entry, args_total_cnt = build_entry(prem[0], prem[1], prem[2], lidx, args_total_cnt, 'prem')

                h_subj_line = json.dumps(h_subj_entry, ensure_ascii=False)
                h_obj_line = json.dumps(h_obj_entry, ensure_ascii=False)
                p_subj_line = json.dumps(p_subj_entry, ensure_ascii=False)
                p_obj_line = json.dumps(p_obj_entry, ensure_ascii=False)

                ofp.write(h_subj_line+'\n')
                ofp.write(h_obj_line+'\n')
                ofp.write(p_subj_line+'\n')
                ofp.write(p_obj_line+'\n')

    ofp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default=None, help='dev or test')
    parser.add_argument('--in_path', type=str, default='/Users/teddy/Files/relational-implication-dataset_levy_holts/%s/%s.tsv', help='input directory')
    parser.add_argument('--out_path', type=str, default='../levy_data/%s_input.json', help='output path')

    args = parser.parse_args()
    if args.subset is None:
        for subset in ['dev', 'test']:
            load_levy_data(args.in_path % (subset, subset), args.out_path % subset)
    else:
        load_levy_data(args.in_path % (args.subset, args.subset), args.out_path % args.subset)
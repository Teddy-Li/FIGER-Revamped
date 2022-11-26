import os
import json
import argparse


def fuzzy_match(a, b):
    return (a in b and len(b) - len(a) < 3) or (b in a and len(a) - len(b) < 3)


def main(results_path, rels_path, out_path):
    results = {}
    ofp = open(out_path, 'w', encoding='utf-8')

    with open(results_path, 'r', encoding='utf8') as rfp:
        for line in rfp:
            item = json.loads(line)
            results[item['sentid']] = item

    with open(rels_path, 'r', encoding='utf8') as rfp:
        for lidx, line in enumerate(rfp):
            hypo, prem, label = line.strip('\n').split('\t')
            try:
                hypo_pred, hypo_subj, hypo_obj = hypo.split(' ')
            except ValueError:
                print(f"Hypothesis empty!")
                hypo_pred = None
                hypo_subj = None
                hypo_obj = None
            try:
                prem_pred, prem_subj, prem_obj = prem.split(' ')
            except ValueError:
                print(f"Premise empty!")
                prem_pred = None
                prem_subj = None
                prem_obj = None

            hypo_subj_str, hypo_subj_jtype = hypo_subj.split('::') if hypo_subj else (None, None)  # these are Javad types
            hypo_obj_str, hypo_obj_jtype = hypo_obj.split('::') if hypo_obj else (None, None)
            prem_subj_str, prem_subj_jtype = prem_subj.split('::') if prem_subj else (None, None)
            prem_obj_str, prem_obj_jtype = prem_obj.split('::') if prem_obj else (None, None)

            hypo_tsubj_q_key = f"{lidx}_hypo_subj"
            hypo_tobj_q_key = f"{lidx}_hypo_obj"
            prem_tsubj_q_key = f"{lidx}_prem_subj"
            prem_tobj_q_key = f"{lidx}_prem_obj"

            hypo_tsubjs = ['#'.join([x[0].lstrip('/'), "%.5f" % x[1]]) for x in results[hypo_tsubj_q_key]['type_preds']] if hypo_subj_str else []
            hypo_tobjs = ['#'.join([x[0].lstrip('/'), "%.5f" % x[1]]) for x in results[hypo_tobj_q_key]['type_preds']] if hypo_obj_str else []
            prem_tsubjs = ['#'.join([x[0].lstrip('/'), "%.5f" % x[1]]) for x in results[prem_tsubj_q_key]['type_preds']] if prem_subj_str else []
            prem_tobjs = ['#'.join([x[0].lstrip('/'), "%.5f" % x[1]]) for x in results[prem_tobj_q_key]['type_preds']] if prem_obj_str else []

            hypo_subj_out = "::".join([hypo_subj_str] + hypo_tsubjs) if hypo_subj_str else None
            hypo_obj_out = "::".join([hypo_obj_str] + hypo_tobjs) if hypo_obj_str else None
            prem_subj_out = "::".join([prem_subj_str] + prem_tsubjs) if prem_subj_str else None
            prem_obj_out = "::".join([prem_obj_str] + prem_tobjs) if prem_obj_str else None

            hypo_out = f"{hypo_pred} {hypo_subj_out} {hypo_obj_out}" if hypo_pred else ''
            prem_out = f"{prem_pred} {prem_subj_out} {prem_obj_out}" if prem_pred else ''
            assert hypo_pred or prem_pred
            ofp.write(f"{hypo_out}\t{prem_out}\t{label}\n")

    ofp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, default=None)
    # parser.add_argument('--input_path', type=str, default='../levy_data/%s_input.json')
    parser.add_argument('--results_path', type=str, default='../levy_data/%s_input_preds.json')
    parser.add_argument('--rels_path', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/ent/%s_rels.txt')
    parser.add_argument('--out_path', type=str, default='/Users/teddy/eclipse-workspace/entgraph_eval/gfiles/ent/%s_rels_ntv1.txt')
    args = parser.parse_args()

    if args.subset is None:
        for subset in ['dev', 'test']:
            main(args.results_path % subset, args.rels_path % subset, args.out_path % subset)
    else:
        main(args.results_path % args.subset, args.rels_path % args.subset, args.out_path % args.subset)

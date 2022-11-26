import json
import os
import copy
import argparse


def integrate_model_outputs_to_typing_files(args):
    type_stats = {}

    output_fn = os.path.join(args.data_dir, args.output_fn)
    output_fp = open(output_fn, 'w', encoding='utf-8')
    print(f"Reading results and dumping into a single output file {output_fn}...")

    total_missing_results_cnt = 0

    for slice_id in range(args.num_slices):
        input_fn = os.path.join(args.data_dir, args.input_fn % slice_id)
        results_fn = os.path.join(args.data_dir, args.results_fn % slice_id)

        print(f"Processing slice {slice_id}: reading from {input_fn} and {results_fn}...")

        # First read in all the results, since they are smaller!
        results = {}
        results_fp = open(results_fn, 'r', encoding='utf-8')
        last_id = -1
        missing_entry_cnt = 0
        for lidx, line in enumerate(results_fp):
            if len(line) < 2:
                continue
            if lidx % 100000 == 0:
                print(f"Reading RESULT line {lidx}...")
            item = json.loads(line)
            new_item = copy.copy(item)
            del new_item['id']
            if int(item['id']) != last_id + 1:
                missing_entry_cnt += 1
                assert int(item['id']) > last_id + 1
                last_id = int(item['id'])
            results[int(item['id'])] = new_item
        results_fp.close()

        print(f"Missing entry count: {missing_entry_cnt}")

        # Now read in the input file, add results to input entries and dump to output
        input_fp = open(input_fn, 'r', encoding='utf-8')
        for lidx, line in enumerate(input_fp):
            if len(line) < 2:
                continue
            if lidx % 100000 == 0:
                print(f"Reading INPUT line {lidx}...")
            item = json.loads(line)
            if item['id'] not in results:
                print(f"Missing result for {item['id']}!")
                total_missing_results_cnt += 1
                item['out_types'] = ['/thing']
            else:
                if item['entity_name'] != results[item['id']]['entity_name']:
                    print_flag = len(item['entity_name']) != len(results[item['id']]['entity_name'])
                    if not print_flag:
                        diffcnt = 0
                        for i in range(len(item['entity_name'])):
                            if item['entity_name'][i] != results[item['id']]['entity_name'][i]:
                                diffcnt += 1
                        print_flag = diffcnt > 5 or (diffcnt > len(item['entity_name']) / 2 and diffcnt > 2)
                    if print_flag:
                        print(f"ERROR: entity name mismatch: {item['entity_name']} vs {results[item['id']]['entity_name']}")
                item['out_types'] = results[item['id']]['type_preds']
            for tp in item['out_types']:
                if tp not in type_stats:
                    type_stats[tp] = {'weight': 0, 'count': 0}
                else:
                    pass
                type_stats[tp]['weight'] += 1/len(item['out_types'])
                type_stats[tp]['count'] += 1

            out_line = json.dumps(item, ensure_ascii=False)
            output_fp.write(out_line + '\n')
        input_fp.close()

    output_fp.close()
    type_stats = {k: v for k, v in sorted(type_stats.items(), key=lambda x: x[1]['weight'], reverse=True)}

    for tp in type_stats:
        print(f"Type {tp} stats: {type_stats[tp]};")

    print(f"Total missing results count: {total_missing_results_cnt}")

    print(f"Done!")


def get_typed_parsed(args):
    typing_output_fn = os.path.join(args.data_dir, args.output_fn)

    print(f"Reading typing results from {typing_output_fn}...")

    typing_results = {}

    with open(typing_output_fn, 'r', encoding='utf-8') as typing_out_fp:
        for lidx, line in enumerate(typing_out_fp):
            if len(line) < 2:
                continue
            if lidx % 100000 == 0:
                print(f"Reading results line {lidx}...")
            entry = json.loads(line)
            entry['lineid'] = int(entry['lineid'])

            if entry['lineid'] not in typing_results:
                typing_results[entry['lineid']] = {}
            entry_inline_relid, entry_inline_position = entry['inline_argid'].split('_')
            entry_inline_relid = int(entry_inline_relid)
            if entry_inline_relid not in typing_results[entry['lineid']]:
                typing_results[entry['lineid']][entry_inline_relid] = {}
            if entry_inline_position == 'subj':
                typing_results[entry['lineid']][entry_inline_relid]['s'] = entry['out_types']
            elif entry_inline_position == 'obj':
                typing_results[entry['lineid']][entry_inline_relid]['o'] = entry['out_types']
            else:
                raise ValueError(f"Unknown inline position {entry_inline_position}!")

    print_cnt = 0
    for lineid in typing_results:
        if print_cnt > 10:
            break
        print(f"Line {lineid} typing results:")
        print(typing_results[lineid])
        print_cnt += 1

    typed_corpus_fp = open(args.parsed_typed_fn, 'w', encoding='utf-8')
    type_result_missing_cnt = 0

    with open(args.parsed_fn, 'r', encoding='utf8') as ifp:
        for lidx, line in enumerate(ifp):
            if len(line) < 2:
                continue
            if lidx % 100000 == 0:
                print(f"Reading line {lidx}; type result missing count: {type_result_missing_cnt}...")
            item = json.loads(line)
            lineid = int(item['lineId'])

            new_rels = []

            for ridx, rel in enumerate(item['rels']):
                rstr = rel["r"]
                assert rstr[0] == '(' and rstr[-1] == ')'
                rstr = rstr[1:-1]
                rlst = rstr.split('::')
                assert (len(rlst) == 6 and args.data_name == 'newsspike') or (len(rlst) == 7 and args.data_name == 'newscrawl')
                try:
                    subj_types = typing_results[lineid][ridx]['s']
                    obj_types = typing_results[lineid][ridx]['o']
                except KeyError:
                    print(f"Missing typing for line {lineid} rel {ridx}!")
                    type_result_missing_cnt += 1
                    subj_types = ['/thing']
                    obj_types = ['/thing']

                if args.data_name == 'newsspike':
                    rlst = rlst
                elif args.data_name == 'newscrawl':
                    rlst = rlst[:6]
                else:
                    raise ValueError(f"Unknown data name {args.data_name}!")
                for st in subj_types:
                    for ot in obj_types:
                        new_rlst = rlst + [st, ot]
                        new_rstr = '::'.join(new_rlst)
                        new_rels.append({'r': f'({new_rstr})', 't': len(subj_types) * len(obj_types)})

            item['rels'] = new_rels
            out_line = json.dumps(item, ensure_ascii=False)
            typed_corpus_fp.write(out_line + '\n')

    print(f"Done! Type result missing count in total: {type_result_missing_cnt}.")
    typed_corpus_fp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../news_data/')
    parser.add_argument('--data_name', type=str, default='newsspike')
    parser.add_argument('--input_fn', type=str, default='%s_gparser_typing_input_%s.json')
    parser.add_argument('--results_fn', type=str, default='%s_gparser_typing_input_%s_preds.json')
    parser.add_argument('--output_fn', type=str, default='%s_gparser_typing_output.json')
    parser.add_argument('--num_slices', type=int, default=None)

    parser.add_argument('--parsed_fn', type=str, default='../../NE_pipeline/news_gen8_p.json')
    parser.add_argument('--parsed_typed_suff', type=str, default='_neural_typed')

    parser.add_argument('--job_name', type=str, required=True, help='[model / corpus]')

    args = parser.parse_args()
    args.input_fn = args.input_fn % (args.data_name, '%d')
    args.results_fn = args.results_fn % (args.data_name, '%d')
    args.output_fn = args.output_fn % args.data_name

    assert args.parsed_fn.endswith('.json')
    args.parsed_typed_fn = args.parsed_fn[:-5] + args.parsed_typed_suff + '.json'

    if args.job_name == 'model':
        integrate_model_outputs_to_typing_files(args)
    elif args.job_name == 'corpus':
        get_typed_parsed(args)
    else:
        raise ValueError(f"Unknown job name {args.job_name}")

    print("Done!")

import json
import random
import pdb

# {"sentence": ["嗯", "。"], "label_seq": ["NN", "PU"], "wrong_idx": 0, "wrong_word": "嗯", "annotated_label": "NN", "predicted_label": "IJ", "sort_key_value": 0.9996890425682068}

def read_data(file_name):
    all_data_lst = []
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            this_dict = json.loads(line.strip())
            all_data_lst.append(this_dict)
    return all_data_lst

def extract_data(all_data, ex_num=150):
    extracted = []
    sum_num = len(all_data)
    step_length = sum_num//ex_num
    for i in range(0, sum_num, step_length):
        idx = round(i + random.random()*step_length)
        if idx >= sum_num:
            idx = sum_num-1
        extracted.append(all_data[idx])
    random.shuffle(extracted)
    print(f'Have Extracted {len(extracted)} Instances')
    return extracted

def write_data(extracted_data, complite_file, partical_file):
    cf = open(complite_file, 'w', encoding='utf-8')
    pf = open(partical_file, 'w', encoding='utf-8')

    for this_dict in extracted_data:
        cf.write(json.dumps(this_dict, ensure_ascii=False)+'\n')
        partical_dict = {}
        partical_dict['sentence'] = this_dict['sentence']
        partical_dict['sentence'][this_dict['wrong_idx']] += '/[TODO]'
        partical_dict['sentence'] = ' '.join(this_dict['sentence'])
        # partical_dict['label_seq'] = this_dict['label_seq']
        # partical_dict['label_seq'][this_dict['wrong_idx']] = '[TODO]'

        # partical_dict['label_seq'] = ' '.join(partical_dict['label_seq'])

        pf.write(json.dumps(partical_dict, ensure_ascii=False)+'\n')

    cf.close()
    pf.close()


def process(input_file, complite_file, partical_file):
    all_data = read_data(input_file)
    ext_data = extract_data(all_data, ex_num=150)
    write_data(ext_data, complite_file, partical_file)

if __name__ == "__main__":
    # pick top n 
    input_file = 'data-error/outpt-full.txt'
    cf = 'data-error/picked_full_data.txt'
    pf = 'data-error/picked_partical.txt'
    process(input_file, cf, pf)
import argparse

def read_data(file_name):
    with open(file_name, 'r') as f:
        lines = [line.strip() for line in f]
    sentences = []
    start, i = 0, 0
    for line in lines:
        if not line:
            sentences.append(lines[start:i])
            start = i + 1
        i += 1

    # for s in sentences:
    #     if(len(s) < 4):
    #         print(s)
    return sentences

def process(src, tgt):
    sentences = read_data(src)
    print('num of sents:', len(sentences))
    new_sentences_lst = []
    for sent in sentences:
        new_s_lst = []
        for i, line in enumerate(sent, 1):
            word, label = line.split()
            new_line_lst = [str(i), word] + ['_'] * 6 + [label, '_']
            new_line = '\t'.join(new_line_lst)
            new_s_lst.append(new_line)
        new_sentences_lst.append(new_s_lst)
    
    with open(tgt, 'w') as f:
        for new_s_lst in new_sentences_lst:
            for new_line in new_s_lst:
                f.write(new_line+'\n')
            f.write('\n')

if __name__ == '__main__':
    # train = 'data.txt.char.train.ner'
    # dev = 'data.txt.char.eval.ner'
    # test = 'data.txt.char.test.ner'

    # tgt_train = 'train.conllu'
    # tgt_dev = 'dev.conllu'
    # tgt_test = 'test.conllu'

    parser = argparse.ArgumentParser(
        description='to transform the source to tgt data.')
    parser.add_argument('--src', help='file name.')
    parser.add_argument('--tgt', help='file name.')
    args = parser.parse_args()

    process(args.src, args.tgt)
    # process(dev, tgt_dev)
    # process(test, tgt_test)


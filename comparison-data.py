# -*-coding:utf-8-*-
import json
import pdb
class pos_sentence():
    # one sentence
    def __init__(self, obj):
        self.sentence = obj["sentence"]
        self.label_seq = obj["label_seq"]
        self.wrong_idx = obj["wrong_idx"]
        self.wrong_word = obj["wrong_word"]
        self.annotated_label = obj["annotated_label"]
        self.predicted_label = obj["predicted_label"]
        self.sort_key_value = obj["sort_key_value"]

    def get_s(self):
        # for word in self.sentence:
        #     s.append(word)
        print(" ".join(self.sentence))
        exit()
    def all_attri(self):
        print(" ".join(s))
        print(self.label_seq, self.wrong_idx)


def get_pos_sentences(filepath, use_se_marker=False):
    sentences = []
    with open(filepath) as f:
        for line in f.readlines():
            sen = json.loads(line)
            pos_sen = pos_sentence(sen)
            sentences.append(pos_sen)
        print("{} total sentences number {}".format(filepath, len(sentences)))
    return sentences


def contrast(file1, file2, file3, file4):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        s1, s2 = [], []
        for i in f1:
            label = i.strip().split("}")[1]
            sentence = i.strip().split("{")[1].split("}")[0]
            s1.append([sentence, label])
        for i in f2:
            
            label = i.strip().split("}")[1]
            sentence = i.strip().split("{")[1].split("}")[0]
            s2.append([sentence, label])
    with open(file3, "w", encoding="utf-8") as f3, open(file4, "w", encoding="utf-8") as f4:
        all_c, correct, notcorrect = 0, 0, 0
        for i, j in zip(s1, s2):
            all_c += 1
            if i[1] != j[1]:
                notcorrect += 1
                temp = "".join(["{",i[0],"}"])
                f4.write(("\t").join([i[1], j[1], temp]))
                f4.write("\n")
                # print(i[1],j[1], i,j)
            else:
                correct += 1
                f3.write(("").join(["{",i[0],"}",i[1]]))
                f3.write("\n")
        print(notcorrect/all_c, correct/all_c)

        
def Errordata(f1,f2,f3):
    with open(f2, 'r', encoding='utf-8') as f2, open(f3, 'w', encoding='utf-8') as f3:
        s1, s2 = [], []
        s_dict = {}
        sentences = get_pos_sentences(f1)
        for s in sentences:
            wrong_idx = s.wrong_idx
            s.sentence[wrong_idx] = s.sentence[wrong_idx]+"/[TODO]"
            sentence = " ".join(s.sentence)
            wrong_word = s.wrong_word
            annotated_label = s.annotated_label
            predicted_label = s.predicted_label
            sort_key_value = s.sort_key_value
            s1.append([sentence, wrong_word, wrong_idx, annotated_label, predicted_label, sort_key_value])
        for s in s1:
            if s[0] not in s_dict.keys():
                s_dict[s[0]] = s
        # pdb.set_trace()
        all_count, diff, same = 0, 0, 0
        avg_same, avg_diff = 0, 0 
        for line in f2:
            # print(line)
            s_ = line.strip().split('}')[0] +"}"
            label_ = line.strip().split('}')[1]
            s_ = eval(s_)
            if s_["sentence"] in s_dict.keys():
                all_count += 1
                if s_dict[s_["sentence"]][3] == label_:
                    same += 1
                    avg_same += s_dict[s_["sentence"]][5]
                else:
                    # print(s_dict[s_["sentence"]][0]+"\n")
                    # print(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]))
                    diff += 1
                    avg_diff += s_dict[s_["sentence"]][5]
                f3.write(s_dict[s_["sentence"]][0]+"\n")
                f3.write(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]))
                f3.write("\n")
            else:
                print('joiafjhoiaps')
        print(diff, all_count, diff/all_count)
        print(same, avg_same/same, diff, avg_diff/diff)

        # for s in s_dict.keys():
        #     print(s)



def Confusiondata(f1,f2):
    with open(f2, 'r', encoding='utf-8') as f2:
        s1, s2 = [], []
        s_dict = {}
        sentences = get_pos_sentences(f1)
        for s in sentences:
            wrong_idx = s.wrong_idx
            s.sentence[wrong_idx] = s.sentence[wrong_idx]+"/[TODO]"
            sentence = " ".join(s.sentence)
            wrong_word = s.wrong_word
            annotated_label = s.annotated_label
            predicted_label = s.predicted_label
            sort_key_value = s.sort_key_value
            s1.append([sentence, wrong_word, wrong_idx, annotated_label, predicted_label, sort_key_value])
        for s in s1:
            if s[0] not in s_dict.keys():
                s_dict[s[0]] = s
        avg_diff_our, number = 0, 0
        diff, same = 0, 0
        avg_diff, avg_same = 0, 0 
        for line in f2:
            xl_label, yh_label = line.strip().split("\t")[0], line.strip().split("\t")[1]
            s_  = line.strip().split("\t")[2]
            s_ = eval(s_)
            determined = line.strip().split("\t")[3]
            if s_["sentence"] in s_dict.keys():
                number += 1
                avg_diff_our += s_dict[s_["sentence"]][5]
                if determined == s_dict[s_["sentence"]][3]:
                    same += 1
                    avg_same += s_dict[s_["sentence"]][5]
                else:
                    diff += 1
                    avg_diff += s_dict[s_["sentence"]][5]
        print(same, diff, number, same/number, diff/number)
        print(avg_same/same, avg_diff/diff)
                    
        print(number, avg_diff_our/number)
        #     s2.append("\t".join([xl_label, yh_label]))
        # print(len(s1), len(s2))
        # for i, j in zip(s1, s2):
        #     print(i, j)
        #     count+=1
        # print(count)


f1 = "picked_partical_cxl.txt"
f2 = "picked_partical.txt"
f3 = "data-error/our-same-label.txt"
f4 = "picked_full_data.txt"
f5 = "data-error/our-different-label.txt"
f6 = "data-error/our-same-label-but-gold.txt"

# contrast(f1, f2, f3, f5)
# Errordata(f4, f3, f6)
Confusiondata(f4, f5)
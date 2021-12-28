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
                    # f3.write(s_dict[s_["sentence"]][0]+"\n")
                    # f3.write(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]))
                    # f3.write("\n"+"\n")
                else:
                    print(s_dict[s_["sentence"]][0])
                    print(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]))
                    diff += 1
                    avg_diff += s_dict[s_["sentence"]][5]
                    f3.write(s_dict[s_["sentence"]][0]+"\n")
                    f3.write(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]))
                    f3.write("\n"+"\n")
            else:
                print('joiafjhoiaps')
        print("人工标注一致的词性个数：",same, "人工一致个数的比率：",same/all_count)
        print("人工标注不一致的词性个数：",diff, "人工不一致个数的比率：", diff/all_count)
        print("人工标注的词性总个数：", all_count, "人工标注一致的模型平均置信度:", avg_same/same)
        print("人工标注的词性总个数：", all_count, "人工标注不一致的模型平均置信度:", avg_diff/diff)
        # print(diff, all_count, )
        return same, diff, all_count

        # for s in s_dict.keys():
        #     print(s)



def Confusiondata(f1, f2, f3):
    with open(f2, 'r', encoding='utf-8') as f2, open(f3, "a", encoding="utf-8") as f_w:
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
        xl_num, yh_num = 0, 0
        for line in f2:
            print(line.strip().split("\t"))
            if line.strip().split("\t")[0] == line.strip().split("\t")[3]:
                xl_num += 1
            if line.strip().split("\t")[1] == line.strip().split("\t")[3]:
                yh_num += 1
            xl_label, yh_label = line.strip().split("\t")[0], line.strip().split("\t")[1]
            s_  = line.strip().split("\t")[2]
            s_ = eval(s_)
            label_ = line.strip().split("\t")[3]
            determined = line.strip().split("\t")[3]
            if s_["sentence"] in s_dict.keys():
                number += 1
                avg_diff_our += s_dict[s_["sentence"]][5]
                if determined == s_dict[s_["sentence"]][3]:
                    same += 1
                    avg_same += s_dict[s_["sentence"]][5]
                    # f_w.write(s_dict[s_["sentence"]][0] + "\n")
                    # f_w.write(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]) + "\n")
                    # f_w.write("\n")
                else:
                    # print(s_dict[s_["sentence"]][0])
                    # print(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]))
                    f_w.write(s_dict[s_["sentence"]][0] + "\n")
                    f_w.write(" ".join(["wrong_word:", s_dict[s_["sentence"]][1], "wrong index:", str(s_dict[s_["sentence"]][2]), "gold:", s_dict[s_["sentence"]][3], "predict:", s_dict[s_["sentence"]][4], "our label:", label_, "confidence:", str(s_dict[s_["sentence"]][5])]) + "\n")
                    f_w.write("\n")
                    diff += 1
                    avg_diff += s_dict[s_["sentence"]][5]
        print("人工标注不一致但已确定的词性和gold相比，相同的词性个数：",same, "占人工不一致个数的比率：",same/number)
        print("人工标注不一致但已确定的词性和gold相比，不相同的词性个数：",diff, "占人工不一致个数的比率：", diff/number)
        print("人工标注不一致但已确定的词性和gold相比，相同词性个数的模型平均置信度：", avg_same/same)
        print("人工标注不一致但已确定的词性和gold相比，不相同词性个数的模型平均置信度：", avg_diff/diff)
                    
        print("人工标注不一致总个数：", number,"人工标注不一致的模型平均置信度:", avg_diff_our/number)
        print(xl_num, yh_num)
        return same, diff, number
        #     s2.append("\t".join([xl_label, yh_label]))
        # print(len(s1), len(s2))
        # for i, j in zip(s1, s2):
        #     print(i, j)
        #     count+=1
        # print(count)
        
def Analysis(f1, f2, f_dict):
    sentences = get_pos_sentences(f1)
    dic = {}
    for s in sentences:
        if s not in dic.keys():
            dic[s] = s.sort_key_value
    list_s = sorted(dic.items(),key=lambda x:x[1],reverse=True)
    # split
    a1 = list_s[:50]
    a2 = list_s[50:100]
    a3 = list_s[100:]
    dic1, dic2, dic3 = {}, {}, {}
    all_num = 76
    for i in a1:
        i[0].sentence[i[0].wrong_idx] = i[0].sentence[i[0].wrong_idx]+"/[TODO]"
        sen_temp = " ".join(i[0].sentence)
        dic1[sen_temp] = i

    for i in a2:
        i[0].sentence[i[0].wrong_idx] = i[0].sentence[i[0].wrong_idx]+"/[TODO]"
        sen_temp = " ".join(i[0].sentence)
        dic2[sen_temp] = i  

    for i in a3:
        i[0].sentence[i[0].wrong_idx] = i[0].sentence[i[0].wrong_idx]+"/[TODO]"
        sen_temp = " ".join(i[0].sentence)
        dic3[sen_temp] = i 
        # print(i) 
    # Read data
    with open(f2, "r", encoding="utf-8") as f_2:
        list_diff = []
        s_l = []
        for i in f_2:
            if i != "\n":
                s_l.append(i.strip())
                # print(i.strip())
            else:
                list_diff.append(s_l)
                s_l = []
        count1, count2, count3 = 0, 0, 0
        for i in list_diff:
            if i[0] in dic1.keys():
                count1 += 1
                # print(i)
            elif i[0] in dic2.keys():
                count2 += 1
            elif i[0] in dic3.keys():
                count3 += 1
            else:
                print("error")
    print(count1/50, count1/all_num,  count2/50, count2/all_num, count3/51, count3/all_num)



# f1 = "picked_partical_cxl.txt"
# f2 = "picked_partical.txt"
# f3 = "data-error/our-same-label.txt"
# f4 = "picked_full_data.txt"
# f5 = "data-error/our-different-label.txt"
# f6 = "data-error/our-same-label-but-gold.txt"
# f6 = "data-error/yh-xl-gold-same.txt"
# f7 = "data-error/our-different-label-UPDATE.txt"
# f_dic = "data-error/data-order-confidence.txt"
f1 = "data-error/picked_partical-layerdrop_cxl.txt"
f2 = "data-error/picked_partical-layerdrop-yh.txt"
f3 = "data-error/layerdrop-our-same-label-.txt"
f4 = "data-error/picked_full_data.txt"
f5 = "data-error/layderdrop-our-different-label.txt"
f6 = "data-error/layerdrop-our-same-label-but-gold.txt"
f7 = "data-error/layderdrop-our-different-label-UPDATE.txt"
f_dic = "data-error/data-layer-order-confidence.txt"

# contrast(f1, f2, f3, f5)
Errordata(f4, f3, f6)
Confusiondata(f4, f7, f6)
Analysis(f4, f6, f_dic)
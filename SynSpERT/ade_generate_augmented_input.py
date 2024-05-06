import json
import scispacy
import spacy
from spacy.tokens import Doc
from more_itertools import locate


#!pip install scispacy
#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.3.0/en_core_sci_sm-0.3.0.tar.gz
#####################################
#### Customized tokenizer        ####
#####################################
#nlp = spacy.load('en_core_web_sm')
#nlp = spacy.load("en_core_sci_sm")
nlp = spacy.load('en_core_sci_sm')


def custom_tokenizer(text):
    tokens = text.split(" ")
    return Doc(nlp.vocab, tokens)
    #global tokens_dict
    #if text in tokens_dict:
    #   return Doc(nlp.vocab, tokens_dict[text])
    #else:
    #   VaueError("No tokenization for input text: ", text)

nlp.tokenizer = custom_tokenizer
#####################################


class JsonInputAugmenter():
    def __init__(self):
        basepath = './data/datasets/ade/'
        self.input_dataset_paths  = [basepath + 'ade_split_0_train.json', 
                                     basepath + 'ade_split_0_test.json', 
                                     basepath + 'ade_split_1_train.json',
                                     basepath + 'ade_split_1_test.json',
                                     basepath + 'ade_split_2_train.json',
                                     basepath + 'ade_split_2_test.json',
                                     basepath + 'ade_split_3_train.json',
                                     basepath + 'ade_split_3_test.json',
                                     basepath + 'ade_split_4_train.json',
                                     basepath + 'ade_split_4_test.json',
                                     basepath + 'ade_split_5_train.json',
                                     basepath + 'ade_split_5_test.json',
                                     basepath + 'ade_split_6_train.json',
                                     basepath + 'ade_split_6_test.json',
                                     basepath + 'ade_split_7_train.json',
                                     basepath + 'ade_split_7_test.json',
                                     basepath + 'ade_split_8_train.json',
                                     basepath + 'ade_split_8_test.json',
                                     basepath + 'ade_split_9_train.json',
                                     basepath + 'ade_split_9_test.json'                                 
                                     ]
        self.output_dataset_paths = [basepath + 'ade_split_0_train_aug.json', 
                                     basepath + 'ade_split_0_test_aug.json', 
                                     basepath + 'ade_split_1_train_aug.json',
                                     basepath + 'ade_split_1_test_aug.json',
                                     basepath + 'ade_split_2_train_aug.json',
                                     basepath + 'ade_split_2_test_aug.json',
                                     basepath + 'ade_split_3_train_aug.json',
                                     basepath + 'ade_split_3_test_aug.json',
                                     basepath + 'ade_split_4_train_aug.json',
                                     basepath + 'ade_split_4_test_aug.json',
                                     basepath + 'ade_split_5_train_aug.json',
                                     basepath + 'ade_split_5_test_aug.json',
                                     basepath + 'ade_split_6_train_aug.json',
                                     basepath + 'ade_split_6_test_aug.json',
                                     basepath + 'ade_split_7_train_aug.json',
                                     basepath + 'ade_split_7_test_aug.json',
                                     basepath + 'ade_split_8_train_aug.json',
                                     basepath + 'ade_split_8_test_aug.json',
                                     basepath + 'ade_split_9_train_aug.json',
                                     basepath + 'ade_split_9_test_aug.json'                                 
                                     ]
            
            
        #self.output_dataset_paths = [basepath + 'scierc_train_aug2.json', basepath + 'scierc_dev_aug2.json', basepath + 'scierc_train_dev_aug2.json', basepath + 'scierc_test_aug2.json']
        #self.taglist =[]

    def augment_docs_in_datasets(self):
        for ipath, opath  in zip(self.input_dataset_paths, self.output_dataset_paths):
            self._augment_docs(ipath, opath)
            #self._datasets[dataset_label] = dataset

    def _augment_docs(self, ipath, opath):
        global tokens_dict
        documents = json.load(open(ipath))
        augmented_documents = []
        nmultiroot=0
        for document in documents:
            jtokens = document['tokens']
            jrelations = document['relations']
            jentities = document['entities']
            jorig_id = document['orig_id']

            lower_jtokens = jtokens #[t.lower() for t in jtokens]
            text = ' '.join(lower_jtokens)
            #text = str.lower(text)
    
            #tokens_dict = {text: jtokens} #put the text in token_dict
            tokens = nlp(text)            #get annotated tokens
            jtags = [token.tag_ for token in tokens]
            #self.taglist =self.taglist + jtags
            jdeps = [token.dep_ for token in tokens]
            #"verb_indicator", "dep_head"
            #root = jdeps.index("ROOT") + 1 #as tokens are numbered from 1 by CoreNLP convention 
            vpos = list(locate(jdeps, lambda x: x == 'ROOT'))
            
            if (len(vpos) != 1):
                flag = 1
                nmultiroot += 1
                print("*** Full sentence:", text)
                for i in vpos:
                    print("ROOT [", i, "]: ", jtokens[i], ", pos tag: ", jtags[i], ", dep: ", jdeps[i])
            else:
                flag = 0

            verb_indicator = [0] * len(jdeps)
            for i in vpos:
                verb_indicator[i] = 1  

            jdep_heads = []
            for i, token in enumerate(tokens):
              if token.head == token:
                 token_idx = 0
              else:
                 token_idx = token.head.i - tokens[0].i + 1
              jdep_heads.append(token_idx)
            if (flag==1):
              print("dep_head: ", jdep_heads)
            d = {"tokens": jtokens, "pos_tags": jtags, "dep_label": jdeps, "verb_indicator": verb_indicator, "dep_head": jdep_heads, "entities": jentities, "relations": jrelations, "orig_id": jorig_id}
            augmented_documents.append(d)
        print("===============  #docs with multiroot = ", nmultiroot)
        with open(opath, "w") as ofile:
            json.dump(augmented_documents, ofile) 

if __name__ == "__main__":
    augmenter = JsonInputAugmenter()
    augmenter.augment_docs_in_datasets()
    #print(list(set(augmenter.taglist)))


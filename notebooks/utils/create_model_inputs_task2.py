import os
from eval_script import RecordTrack1
import json
from nltk.tokenize.punkt import PunktSentenceTokenizer

def process_n2c2_files_task2(base_dir, data_split, from_ner=False, test_ann_dir=None):
    '''
    base_dir: (str) base directory (abs or rel to current)
    data_split: (str) train, test, or dev
    from_ner: (bool) if true, load from predicted .ann files from task 1. If false, load from gold standard.
    test_ann_dir: (str) directory of annotation files for test split
    '''
    text_dir = os.path.join(base_dir, data_split)
    # If we are testing based on annotations from Task 1 NER
    if from_ner:
        ann_dir = os.path.join(base_dir, 'output', test_ann_dir)
    else:
        ann_dir = text_dir
        
    root_names = [file.replace(".ann", "") for file in 
                  os.listdir(ann_dir) if ".ann" in file]
    counts = {}
    json_output = []
    sent_tokenizer = PunktSentenceTokenizer()
    all_med_mentions = []
    for root_name in root_names:
        ann_file = os.path.join(ann_dir, root_name + ".ann")
        txt_file = os.path.join(text_dir, root_name + ".txt")
        txt = open(txt_file, "r")
        doc_text = txt.read()
        sentences = sent_tokenizer.tokenize(doc_text)
        spans = sent_tokenizer.span_tokenize(doc_text)
        # Process annotation file for same file
        # Only keep medical mentions for now; sort by start pos
        annotation = RecordTrack1(ann_file)
        all_tags = annotation.annotations['tags'].values()
        
        meds = sorted([tag for tag in all_tags if tag.ttype != "Drug"],
                        key = lambda item: item.start)
                
        all_med_mentions.extend(meds)
        for sent_start, sentence in enumerate(sentences):
            start_idx = spans[sent_start][0]
            end_idx = spans[sent_start][1]
            for med in meds:
                if med.start >= start_idx and med.start <= end_idx:
                    if (sent_start > 0) and (sent_start < (len(sentences) - 1)):
                        # all 3 sentences
                        text = sentences[sent_start-1] + sentence + sentences[sent_start+1]
                    elif sent_start > 0:
                        # prev_sentence + sentence
                        text = sentences[sent_start-1] + sentence
                    elif sent_start < (len(sentences) - 1):
                        # sentence + next_sentence
                        text = sentence + sentences[sent_start+1]

                    label = med.ttype
                    if label not in counts:
                        counts[label] = 0
                    counts[label] += 1
                    json_output.append({"text": text, 
                                        "label": label, 
                                        "note_id": root_name, 
                                        "tid": med.rid, 
                                        "start": med.start, 
                                        "end": med.end, 
                                        "name": med.text})
        assert len(all_med_mentions) == len(json_output)
    print(counts)
    print(len(all_med_mentions))
    if from_ner:
        # test file starting from NER output is labeled differently
        filepath = f"{base_dir}/input/event_input/event_input_{data_split}_from_ner.json"
    else:
        filepath = f"{base_dir}/input/event_input/event_input_{data_split}.json"

    # with open(filepath, 'w') as fp:
    #     fp.write('\n'.join(json.dumps(i) for i in json_output) +'\n')
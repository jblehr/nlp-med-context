# import required python libraries
import importlib
import os
import datetime
import json
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader


os.chdir('/content/drive/Shareddrives/NLP Project/notebooks/subtask 3 Context')
import sequence_classifier
from sequence_classifier import BERT_PL
os.chdir('/content/drive/Shareddrives/NLP Project')

def get_best_model(model_name):
    '''
    Looks in a model directory and returns the most recent model run
    
    Input:
        model_name: (str) name of model
    Return:
        latest_version: (str) latest version of model
    '''
    latest_date = datetime.datetime.strptime("1-1-2000", "%m-%d-%Y")
    latest_version = None
    for version in os.listdir(os.path.join('data/models', model_name)):
        if 'version' in version:
            version_time = datetime.datetime.strptime(version.replace("version_", ""), "%d-%m-%Y--%H-%M-%S")
            if version_time >= latest_date:
                latest_date = version_time
                latest_version = version
    return latest_version

def load_best_model(ckpt_path):
    # Load best model and trainer
    best_model_path = os.path.join(ckpt_path, "best_model.ckpt")
    model = BERT_PL.load_from_checkpoint(best_model_path).cuda()
    trainer = pl.Trainer(gpus=1)
    return model
    
    
def make_predictions(datapath, model_name, model):
    # PREDICT ON TEST SET

    with open(datapath) as json_file:
        data = [json.loads(line) for line in json_file]
    parsed_data =[]
    for line in data:
        parsed_data.append([sequence_classifier.LABEL_TO_INT[model_name][line["label"]], line["text"]])

    dataset = sequence_classifier.BERTSST2Dataset(model.tokenizer, parsed_data)
    test_data = DataLoader(dataset, batch_size=8, collate_fn=dataset.collate_fn)
    prediction_list = []
    for i, batch in enumerate(test_data):
        torch.cuda.empty_cache()
        b1, b2, b3 = (batch[0].cuda(), batch[2].cuda(), 
                      batch[1].cuda())
        ughhh = {"input_ids": b1, "labels": b2, "attention_mask": b3}
        output = model(**ughhh)
        del b1
        del b2
        del b3
        del ughhh
        prediction_list.append(output[1].tolist())
        del output
        torch.cuda.empty_cache()

    return prediction_list
    
    
def pre_process_results(prediction_list, input_path, model_name):
    '''
    predict_path: (str) filepath of predicted labels (in txt file)
    input_path: (str) filepath of input data (in json form)

    return:
        pred_lines: (list) of strings of len = no. sentences
        json_lines: (list) of json-formatted objects, len = no. sentences
    '''
    # predictions
    INT_TO_NAME = {}
    for key, val in sequence_classifier.LABEL_TO_INT.items():
        INT_TO_NAME[key] = {idx: name for name, idx in val.items()} 

    pred_lines = []
    for batch_list in prediction_list:
      for pred in batch_list:
        pred_lines.append(INT_TO_NAME[model_name][pred])

    # input data
    json_file = open(input_path,'r')
    json_lines = [line for line in json_file]
    assert len(pred_lines)  == len(json_lines)
    return pred_lines, json_lines
    

def process_mentions(pred_lines, json_lines):
    '''
    pred_lines: (list) of strings of len = no. mentions
    json_lines: (list) of json-formatted objects, len = no. mentions

    return:
        doc_dic: (dict) of key = doc id (eg "137-04") and values = list of 
                  strings already formatted to be written to ann files
    '''
    doc_dic = {}
    
    att_counter = {}

    for med_idx, label in enumerate(pred_lines):
        med = json.loads(json_lines[med_idx])
        doc_id = med['note_id']
        
        mention_str = f'''{med['tid'].replace("E","T")}\t{label} {med['start']} {med['end']}\t{med['name']}\n'''
        event_str = f'''{med['tid']}\t{label}:{med['tid'].replace("E","T")}\n'''
        
        
        med_list = doc_dic.get(doc_id, [])
        med_list.extend([mention_str, event_str])
        
        # Add in blank attributes to make data reading for Task 3 more straightforward
        blank_attributes = [("Certainty", "Unknown"),
                            ("Actor", "Unknown"),
                            ("Action", "Unknown"),
                            ("Temporality","Unknown"),
                            ("Negation", "NotNegated")]
        
        if label == "Disposition":
            for att, att_label in blank_attributes:
                att_counter[doc_id] = att_counter.get(doc_id, 0) + 1
                att_str = f'''A{att_counter[doc_id]}\t{att} {med['tid']} {att_label}\n'''
                med_list.append(att_str)

        doc_dic[doc_id] = med_list
    
    return doc_dic  
    
    
def write_to_ann(doc_dic, ann_path):
    # Write results to ann files
    for ann_file, mention_list in doc_dic.items():
        file_name = f'{ann_path}/{ann_file}.ann'
        if os.path.exists(file_name):
            os.remove(file_name)
        with open(file_name, 'w') as ann:
            for mention_str in mention_list:
                ann.writelines(mention_str)
                
# main program execution
def go(input_path, model_name, ann_path):
    # get most recent run
    latest_version = get_best_model(model_name)
    ckpt_path = os.path.join('data/models', model_name, latest_version, 'checkpoints')
    # load model
    model = load_best_model(ckpt_path)
    # Make predictions on test set
    prediction_list = make_predictions(input_path, model_name, model)

    # Process data
    pred_lines, json_lines = pre_process_results(prediction_list, input_path, model_name)
    doc_dic = process_mentions(pred_lines, json_lines)

    # write to new ann files
    write_to_ann(doc_dic, ann_path)

    

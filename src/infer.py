import configparser
import os
import pickle
import pprint
import torch

import numpy as np

from src.models import LSTMClassifier
from src.utils import FeatureExtractor


def inference(model_name: str = 'LSTMClassifier', device: str = torch.device('cpu')) -> dict:
    """Script to do inference using trained model config, feature_config: model configuration
    and feature configuration files
    :param model_name: define the model used for inference
    :param device: device where the inference take place
    :return: A list of id -> labels with a probability score
    """

    # Load the feature configuration
    feature_config = configparser.ConfigParser()
    feature_config.read('config/feature_config')

    # Load the inference configuration
    inference_config = configparser.ConfigParser()
    inference_config.read('config/inference_config')

    # Feature extractor
    FE = FeatureExtractor(feature_config['default'])

    # Loop over all files
    scores = {}
    file_list = open('data/wav_test.scp').readlines()
    file_list = [line.strip().split() for line in file_list]
    for fileId, path in file_list:
        # Prepare features
        try:
            F = FE.extract(path)
        except IOError:
            print('failed for ' + fileId)
            continue
        if inference_config['training_dataset'].get('apply_mean_norm', False):
            F = F - torch.mean(F, dim=0)
        if inference_config['training_dataset'].get('apply_var_norm', False):
            F = F / torch.std(F, dim=0)
        feat = F.to(device)

        # Inference Phase
        if model_name == 'LSTMClassifier':
            model_args = {'architecture': 'LSTMClassifier', 'apply_mean_norm': False, 'apply_var_norm': False,
                          'input_dimension': 192, 'lstm_encoder_units': 128, 'lstm_num_layers': 2,
                          'lstm_bidirectional': True,
                          'lstm_dropout': 0.1, 'lstm_pooling': 'average', 'classifier_units': 64,
                          'classifier_activation': 'Tanh', 'classifier_dropout': 0.1}

            # Load model, use CPU for inference
            model = LSTMClassifier(model_args)
            model.load_state_dict(torch.load('model/torch_based/final.pt', map_location='cpu'))
            model = model.to(device)
            model.eval()

            seg_mode = inference_config['training_dataset'].get('mode', 'file')
            if seg_mode == 'file':
                feat = [feat]
            elif seg_mode == 'segment':
                segment_length = int(inference_config['training_dataset'].get('segment_length', 300))
                segment_hop = int(inference_config['training_dataset'].get('segment_hop', 10))
                feat = [feat[i:i + segment_length, :] for i in
                        range(0, max(1, F.shape[0] - segment_length), segment_hop)]
            else:
                raise ValueError('Unknown eval model')

            with torch.no_grad():
                output = model.predict_proba(feat)

            # Average the scores of all segments from the input file
            scores[fileId] = sum(output)[0].item() / len(output)
        else:
            # load model
            sk_model = pickle.load(open('model/sklearn_based/model.pkl', 'rb'))
            score = sk_model.validate([F])
            # Average the scores of all segments from the input file
            score = np.mean(score[0], axis=0)[1]
            scores[fileId] = score

    # Write output scores
    output_path = 'data/scores.txt'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        print('Scores written to ' + output_path)
        for item in scores:
            f.write(item + " " + str(scores[item]) + "\n")
    pprint.pprint(scores)
    return scores

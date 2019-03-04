import argparse
from ujson import load as json_load

import typing

import util


def get_setup_args():
  """Get arguments needed in setup.py."""
  parser = argparse.ArgumentParser(
      'Evaluate BERT results from Huggingface repo.')

  parser.add_argument(
      '--eval_file_path', type=str, default='./data/test_eval.json')
  parser.add_argument(
      '--result_file_path',
      type=str,
      default='../BERT/save/bert-base-uncased/predictions.json')
  args = parser.parse_args()
  return args


def invert_golden(gold_dict):
  """Golden normally maps from idx -> object. This function changes the mapping so that
  it is now uuid -> object + {idx : idx }
  """
  inverted = {}
  for key, val in gold_dict.items():
    uuid = val["uuid"]
    assert uuid not in inverted
    val["idx"] = key
    inverted[uuid] = val
  return inverted


def evaluate_bert(args):
  eval_file_path: str = args.eval_file_path
  with open(eval_file_path, 'r') as fh:
    gold_dict = invert_golden(json_load(fh))

  result_file_path: str = args.result_file_path
  with open(result_file_path, 'r') as fh:
    pred_dict = json_load(fh)

  use_squad_v2: bool = True
  results = util.eval_dicts(gold_dict, pred_dict, use_squad_v2)
  return results


if __name__ == '__main__':
  args = get_setup_args()
  results = evaluate_bert(args)
  print(results)

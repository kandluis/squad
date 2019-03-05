import argparse
import csv
import os
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
  parser.add_argument(
      '--write_predictions_csv',
      dest='write_predictions_csv',
      action='store_true')
  parser.add_argument(
      '--no-write_predictions_csv',
      dest='write_predictions_csv',
      action='store_false')
  parser.set_defaults(write_predictions_csv=True)
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

  if args.write_predictions_csv:
    path, filename = os.path.split(eval_file_path)
    filename = ".".join(filename.split(".")[:-1])
    sub_path = os.path.join(path, filename) + ".csv"
    print("Saving results to: %s" % sub_path)
    with open(sub_path, 'w', newline='', encoding='utf-8') as csv_fh:
      csv_writer = csv.writer(csv_fh, delimiter=',')
      csv_writer.writerow(['Id', 'Predicted'])
      for uuid in sorted(pred_dict):
        csv_writer.writerow([uuid, pred_dict[uuid]])

  return results


if __name__ == '__main__':
  args = get_setup_args()
  results = evaluate_bert(args)
  print(results)

import numpy as np 

import args

def pre_process(args):
  pass

if __name__ == '__main__':
  # Get command-line args
  args_ = args.get_setup_bert_args()

  pre_process(args_)

    np.savez(
      out_file,
      context_idxs=np.array(context_idxs),
      context_char_idxs=np.array(context_char_idxs),
      ques_idxs=np.array(ques_idxs),
      ques_char_idxs=np.array(ques_char_idxs),
      y1s=np.array(y1s),
      y2s=np.array(y2s),
      ids=np.array(ids))
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import os.path as op

import tensorflow as tf

import configuration
import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
import time
#
# FLAGS = tf.flags.FLAGS
#
# tf.flags.DEFINE_string("checkpoint_path", "/im2txt/model/",
#                        "Model checkpoint file or directory containing a "
#                        "model checkpoint file.")
# tf.flags.DEFINE_string("vocab_file", "/im2txt/word_counts.txt", "Text file containing the vocabulary.")
# tf.flags.DEFINE_string("input_files", "/im2txt/test.jpg",
#                        "File pattern or comma-separated list of file patterns "
#                        "of image files.")


class ImgCaption():

  def __init__(self):
    c_dir = os.path.dirname(__file__)
    self.checkpoint_path = op.join(c_dir, "im2txt/model/model.ckpt-2000000")
    self.vocab_file = op.join(c_dir, "im2txt/word_counts.txt")

  def image_caption(self,img_path, ret = dict()):
    # Build the inference graph.
    start = time.time()
    try:
      g = tf.Graph()
      with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                                   self.checkpoint_path)
      g.finalize()

      # Create the vocabulary.
      vocab = vocabulary.Vocabulary(self.vocab_file)

      with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)
        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)
        # for filename in filenames:
        with tf.gfile.GFile(img_path, "r") as f:
          image = f.read()
        captions = generator.beam_search(sess, image)
        # print("Captions for image %s:" % os.path.basename(img_path))
        meta =[]
        for i, caption in enumerate(captions):
          # Ignore begin and end words.
          sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
          sentence = " ".join(sentence)
          prob = math.exp(caption.logprob)
          meta.append((sentence,"%.6f"%prob))
          # print("  %d) %s (p=%f)" % (i, sentence,)
        end = time.time()
        ret['result'] = (True,meta,"%.3f"%(end-start))
        return meta
    except Exception as err:
      ret['result'] = (False,"something went wrong with image caption function","maybe another one?")
      return False
if __name__ == "__main__":
  # tf.app.run()
  IP = ImgCaption()
  ret = IP.image_caption("elepant.jpg")
  if ret != False:
    for i in ret:
      print("%s (p=%f)"%(i[0],i[1]))
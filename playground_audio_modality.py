from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
import soundfile as sf
import torch
import numpy as np
# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


# define function to read in sound file
def map_to_array(filepath):
    speech, _ = sf.read(filepath)
    return speech



# retrieve logits
# define function to read in sound file
 # load dummy dataset and read soundfiles

 # tokenize
sample_file = map_to_array('sample1.flac')

input_values = tokenizer([sample_file, sample_file], return_tensors="pt",
                         padding="longest").input_values  # Batch size 1

logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)
# 1. Take care of audio model
# 2. Take care of video model
# 3. Train simple CLIP model
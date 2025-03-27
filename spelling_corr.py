import happytransformer
import nltk
nltk.download('punkt')

model_loaded = happytransformer.HappyTextToText('T5','gec_t5_model')

beam_params =  happytransformer.TTSettings(num_beams=5, min_length=1, max_length=20)

sentence = "I likeis to eatt. applees"
input = "grammar: "+ sentence
correction_1 = model_loaded.generate_text(input, args=beam_params)
print('Incorrect sentence:',sentence)
print('Corrected sentence:',correction_1.text)
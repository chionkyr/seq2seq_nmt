import tensorflow as tf

from thesis_lib.data.text import load_parallel_texts
from thesis_lib.ds_utils.ds_map_functions import to_seq2seq_format, to_tokenize_input_target, to_one_hot_target_only
from thesis_lib.ds_utils.helpers import split_ds
from thesis_lib.metrics import corpus_gleu, corpus_bleu
from thesis_lib.metrics import sentence_gleu, sentence_bleu, sentence_wer, sentence_cer
from thesis_lib.models import EmbeddingSeq2Seq
from thesis_lib.preprocessing.constants import CHARACTER_PAD_TOKEN, WORD_PAD_TOKEN
from thesis_lib.preprocessing.parallel_text import preprocess_text_pairs_for_seq2seq
from thesis_lib.preprocessing.utils import reverse_tokenization, get_string_lookup

data_path = "parallel_corpus.txt"
raw_texts = load_parallel_texts(data_path, max_length=80)

# Preprocessing
character_level = False

to_lower = False
to_ascii = True
remove_numbers = False

texts, vocabs = preprocess_text_pairs_for_seq2seq(raw_texts, character_level=character_level, to_lower=to_lower, to_ascii=to_ascii, remove_numbers=remove_numbers)

input_vocab = vocabs[0]
target_vocab = vocabs[1]

input_vocab_size = len(input_vocab)
target_vocab_size = len(target_vocab)
max_input_seq_length = len(max(texts[0], key=len))
max_target_seq_length = len(max(texts[1], key=len)) - 2

print("Number of sentences:", len(texts[0]))
print("Number of unique input tokens:", input_vocab_size)
print("Number of unique output tokens:", target_vocab_size)
print("Max sentences length for inputs:", max_input_seq_length)
print("Max sentences length for outputs:", max_target_seq_length)

input_vocab_to_num, input_num_to_vocab = get_string_lookup(input_vocab)
target_vocab_to_num, target_num_to_vocab = get_string_lookup(target_vocab)

# Dataset
dataset = tf.data.Dataset.from_tensor_slices(texts)

val_percentage = 0.2
test_percentage = 0.2

train_ds, val_ds, test_ds = split_ds(dataset, val_percentage=val_percentage, test_percentage=test_percentage)
train_ds = train_ds.shuffle(64 * 64)
val_ds = val_ds.shuffle(64 * 64)
test_ds = test_ds.shuffle(64 * 64)

if character_level:
	i_pad_value = input_vocab_to_num(CHARACTER_PAD_TOKEN).numpy().astype("int32")
	t_pad_value = target_vocab_to_num(CHARACTER_PAD_TOKEN).numpy().astype("int32")
else:
	i_pad_value = input_vocab_to_num(WORD_PAD_TOKEN).numpy().astype("int32")
	t_pad_value = target_vocab_to_num(WORD_PAD_TOKEN).numpy().astype("int32")

batch_size = 128 * 2

train_ds_eb = train_ds.map(lambda x, y: to_tokenize_input_target(x, y, input_vocab_to_num, target_vocab_to_num, character_level))
train_ds_eb = train_ds_eb.map(lambda x, y: to_seq2seq_format(x, y, t_pad_value))
train_ds_eb = train_ds_eb.cache()
train_ds_eb = train_ds_eb.padded_batch(batch_size, drop_remainder=True, padding_values=((i_pad_value, t_pad_value), t_pad_value))
train_ds_eb = train_ds_eb.map(lambda x, y: to_one_hot_target_only(x, y, target_vocab_size))
train_ds_eb = train_ds_eb.prefetch(tf.data.experimental.AUTOTUNE)

val_ds_eb = val_ds.map(lambda x, y: to_tokenize_input_target(x, y, input_vocab_to_num, target_vocab_to_num, character_level))
val_ds_eb = val_ds_eb.map(lambda x, y: to_seq2seq_format(x, y, t_pad_value))
val_ds_eb = val_ds_eb.cache()
val_ds_eb = val_ds_eb.padded_batch(batch_size, drop_remainder=True, padding_values=((i_pad_value, t_pad_value), t_pad_value))
val_ds_eb = val_ds_eb.map(lambda x, y: to_one_hot_target_only(x, y, target_vocab_size))
val_ds_eb = val_ds_eb.prefetch(tf.data.experimental.AUTOTUNE)

test_ds_eb = test_ds.map(lambda x, y: to_tokenize_input_target(x, y, input_vocab_to_num, target_vocab_to_num, character_level))
test_ds_eb = test_ds_eb.map(lambda x, y: to_seq2seq_format(x, y, t_pad_value))
test_ds_eb = test_ds_eb.cache()
test_ds_eb = test_ds_eb.padded_batch(batch_size, drop_remainder=True, padding_values=((i_pad_value, t_pad_value), t_pad_value))
test_ds_eb = test_ds_eb.map(lambda x, y: to_one_hot_target_only(x, y, target_vocab_size))
test_ds_eb = test_ds_eb.prefetch(tf.data.experimental.AUTOTUNE)

print("batch_size: {0}".format(batch_size))
print("train_ds_eb length: {0} batches".format(len(train_ds_eb)))
print("val_ds_eb length: {0} batches".format(len(val_ds_eb)))
print("val_ds_eb length: {0} batches".format(len(test_ds_eb)))

# Hyperparameters and Callbacks

rnn_type = "bi_lstm"
rnn_size = 128

embedding_size = 300
epochs = 30

callbacks = [
	tf.keras.callbacks.ReduceLROnPlateau(patience=5, min_delta=1e-3, min_lr=1e-4, verbose=1),
	tf.keras.callbacks.EarlyStopping(patience=12, min_delta=1e-4, verbose=1),
]

# Embeddings Model

embeddings_model = EmbeddingSeq2Seq(
	rnn_type=rnn_type,
	rnn_size=rnn_size,
	embedding_size=embedding_size,
	num_encoder_tokens=input_vocab_size,
	num_decoder_tokens=target_vocab_size,
	target_token_to_index=target_vocab_to_num,
	target_index_to_token=target_num_to_vocab,
	max_decoder_seq_length=max_target_seq_length,
	character_level=character_level
)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
embeddings_model.create_model()

embeddings_model.fit(train_ds_eb, epochs, val_ds_eb, callbacks)

embeddings_model.plot_history()
embeddings_model.evaluate(test_ds_eb)

# Testing
ds_len = len(test_ds_eb)
prog_bar = tf.keras.utils.Progbar(ds_len)

test_target = []
test_predicted = []
for i, batch in enumerate(test_ds_eb.as_numpy_iterator()):
	input_batch = batch[0][0]
	target_batch = batch[0][1]
	target = reverse_tokenization(target_batch, target_num_to_vocab, character_level)
	predicted = embeddings_model.decode_sequences(input_batch)
	test_target.extend(target)
	test_predicted.extend(predicted)
	prog_bar.update(i)
prog_bar.update(ds_len, finalize=True)

print(corpus_gleu(test_target, test_predicted))
print(corpus_bleu(test_target, test_predicted))

for batch in val_ds_eb.take(1).as_numpy_iterator():
	input_batch = batch[0][0]
	target_batch = batch[0][1]
	original = reverse_tokenization(input_batch, input_num_to_vocab, character_level)
	target = reverse_tokenization(target_batch, target_num_to_vocab, character_level)
	predicted = embeddings_model.decode_sequences(input_batch)
	for original_seq, target_seq, predicted_seq in zip(original, target, predicted):
		original_seq = original_seq.strip()
		target_seq = target_seq.strip()
		predicted_seq = predicted_seq.strip()
		print("Original: {0}".format(original_seq))
		print("Target: {0}".format(target_seq))
		print("Translated: {0}".format(predicted_seq))
		s_gleu = sentence_gleu(target_seq, predicted_seq)
		s_bleu = sentence_bleu(target_seq, predicted_seq)
		s_wer = sentence_wer(target_seq, predicted_seq)
		s_cer = sentence_cer(target_seq, predicted_seq)
		print("Metrics: sentence_glue: {0}, sentence_bleu: {1}, sentence_wer: {2}, sentence_cer: {3}".format(s_gleu, s_bleu, s_wer, s_cer))
		print("#")
	print("###")

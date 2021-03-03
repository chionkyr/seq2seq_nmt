import tensorflow as tf


def _str_to_sequence(label, vocab_to_num, character_level):
	if character_level:
		label_seq = vocab_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
	else:
		label_seq = vocab_to_num(tf.strings.split(label))
	return tf.cast(label_seq, dtype=tf.int32), tf.shape(label_seq)[0]


def to_one_hot_decoder_only(input_data, label_sequence, vocab_size):
	decoder_input_data = tf.one_hot(input_data[1], depth=vocab_size)
	decoder_target_data = tf.one_hot(label_sequence, depth=vocab_size)
	return (input_data[0], decoder_input_data), decoder_target_data


def to_one_hot_all(input_data, target_data, input_vocab_size, output_vocab_size):
	encoder_input_data = tf.one_hot(input_data[0], depth=input_vocab_size)
	decoder_input_data = tf.one_hot(input_data[1], depth=output_vocab_size)
	decoder_target_data = tf.one_hot(target_data, depth=output_vocab_size)
	return (encoder_input_data, decoder_input_data), decoder_target_data


def to_one_hot_target_only(input_data, target_data, output_vocab_size):
	encoder_input_data = input_data[0]
	decoder_input_data = input_data[1]
	decoder_target_data = tf.one_hot(target_data, depth=output_vocab_size)
	return (encoder_input_data, decoder_input_data), decoder_target_data


def to_tokenize_input_target(input_data, target_data, input_vocab_to_num, target_vocab_to_num, character_level):
	input_data_seq, _ = _str_to_sequence(input_data, input_vocab_to_num, character_level)
	target_data_seq, _ = _str_to_sequence(target_data, target_vocab_to_num, character_level)
	return input_data_seq, target_data_seq


def to_seq2seq_format(input_data, target_data, pad_value):
	return (input_data, target_data), tf.concat([target_data[1:], tf.constant([pad_value])], axis=0)

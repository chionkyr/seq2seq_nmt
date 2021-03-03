import tensorflow as tf


def split_ds(ds: tf.data.Dataset, val_percentage=None, test_percentage=None, buffer_size=None):
	val_percentage = val_percentage or 0
	test_percentage = test_percentage or 0
	buffer_size = buffer_size or 128 * 128
	if val_percentage < 0 or val_percentage >= 1.0:
		raise ValueError("val_percentage must be between (0,1)")
	if test_percentage < 0 or test_percentage >= 1.0:
		raise ValueError("test_percentage must be between (0,1)")
	if (val_percentage + test_percentage) >= 1.0:
		raise ValueError("val_percentage+test_percentage must be between (0,1)")

	full_ds_size = len(ds)
	print("Full size: {0}".format(full_ds_size))
	if val_percentage == 0 and test_percentage == 0:
		print("No split returning ds shuffled")
		return ds.shuffle(buffer_size, reshuffle_each_iteration=False)
	elif val_percentage != 0 and test_percentage == 0:
		val_ds_size = int(full_ds_size * val_percentage)
		train_ds_size = full_ds_size - val_ds_size

		ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

		train_ds = ds.take(train_ds_size)
		val_ds = ds.skip(train_ds_size)

		print("Train size: {0}".format(len(train_ds)))
		print("Val size: {0}".format(len(val_ds)))
		return train_ds, val_ds

	elif val_percentage == 0 and test_percentage != 0:
		test_ds_size = int(full_ds_size * test_percentage)
		train_ds_size = full_ds_size - test_ds_size

		ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

		train_ds = ds.take(train_ds_size)
		test_ds = ds.skip(train_ds_size)

		print("Train size: {0}".format(len(train_ds)))
		print("Test size: {0}".format(len(test_ds)))
		return train_ds, test_ds
	else:
		val_ds_size = int(full_ds_size * val_percentage)
		test_ds_size = int(full_ds_size * test_percentage)
		train_ds_size = full_ds_size - test_ds_size - val_ds_size

		ds = ds.shuffle(buffer_size, reshuffle_each_iteration=False)

		train_ds = ds.take(train_ds_size)
		remaining = ds.skip(train_ds_size)

		test_ds = remaining.take(test_ds_size)
		val_ds = remaining.skip(test_ds_size)

		print("Train size: {0}".format(len(train_ds)))
		print("Val size: {0}".format(len(val_ds)))
		print("Test size: {0}".format(len(test_ds)))
		return train_ds, val_ds, test_ds,

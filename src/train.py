import tensorflow as tf
import numpy as np
from datetime import datetime
from imageio import imwrite
import glob
import os


class Model(object):
    def __init__(self):
        self.sess = tf.Session()

        self.train_dataset, self.val_dataset = self.create_datasets()

        self.train_init, self.val_init, self.input, self.target, self.output, self.loss, self.is_training = \
            self.create_network()

        self.train_summaries, self.val_summaries, self.summary_writer = self.create_summaries()

        self.filename_pl, self.load_image_op = self.load_image()

        self.global_step = tf.Variable(0, trainable=False, name='global_step', dtype=tf.int64)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=5e-4)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step)
        self.init = tf.global_variables_initializer()

    def load_image(self):
        filename = tf.placeholder(tf.string, shape=())
        img = tf.read_file(filename)
        img = tf.image.decode_image(img, channels=1)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        return filename, img

    def create_summaries(self):
        loss_summary = tf.summary.scalar('train/loss', self.loss)
        validation_loss_summary = tf.summary.scalar('validation/validation_loss', self.loss)
        train_summaries = [loss_summary]

        train_summaries = tf.summary.merge(train_summaries)
        val_summaries = tf.summary.merge([validation_loss_summary])

        datestring = datetime.strftime(datetime.now(), '%m-%d_%H%M%S')
        run_name = datestring

        log_dir = "../logs/" + run_name + "/"
        summary_writer = tf.summary.FileWriter(log_dir, self.sess.graph)

        return train_summaries, val_summaries, summary_writer

    @staticmethod
    def map_data(filename, target_filename):
        image = tf.read_file(filename)
        image = tf.image.decode_image(image, channels=1)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        target_image = tf.read_file(target_filename)
        target_image = tf.image.decode_image(target_image, channels=1)
        target_image = tf.image.convert_image_dtype(target_image, dtype=tf.float32)

        return image, target_image

    def create_datasets(self):
        blurred_files = glob.glob('../data/blurred/*.png')
        target_files = []

        n_samples = len(blurred_files)
        for i, f in enumerate(blurred_files):
            target_f = f.replace('blurred', 'target')[:-6] + '.png'
            target_files.append(target_f)

        n_train_samples = int(0.8 * n_samples)

        train_dataset = tf.data.Dataset.from_tensor_slices((blurred_files[:n_train_samples],
                                                            target_files[:n_train_samples]))
        train_dataset = train_dataset.repeat().map(Model.map_data, num_parallel_calls=8)

        val_dataset = tf.data.Dataset.from_tensor_slices((blurred_files[n_train_samples:],
                                                          target_files[n_train_samples:]))
        val_dataset = val_dataset.map(Model.map_data, num_parallel_calls=8)

        batch_size = 16
        train_dataset = train_dataset.batch(batch_size=batch_size).prefetch(2)
        val_dataset = val_dataset.batch(batch_size=batch_size)

        return train_dataset, val_dataset

    def create_network(self):
        iter = tf.data.Iterator.from_structure(self.train_dataset.output_types,
                                               self.train_dataset.output_shapes)
        train_init = iter.make_initializer(self.train_dataset)
        val_init = iter.make_initializer(self.val_dataset)

        input, target = iter.get_next()
        input = tf.reshape(input, shape=[-1, 28, 28, 1])
        target = tf.reshape(target, shape=[-1, 28, 28, 1])

        kernel_sizes = [7, 5, 3]
        channel_numbers = [64, 64, 1]
        is_training = tf.placeholder_with_default(True, shape=())

        output = tf.layers.conv2d(input, channel_numbers[0], [kernel_sizes[0], kernel_sizes[0]], padding="same",
                                  activation=tf.nn.relu)

        output = tf.layers.dropout(output, rate=0.1, training=is_training)

        for i in range(1, len(kernel_sizes[:-1])):
            output = tf.layers.conv2d(output, channel_numbers[i], [kernel_sizes[i], kernel_sizes[i]], padding="same",
                                      activation=tf.nn.relu)

            output = tf.layers.dropout(output, rate=0.1, training=is_training)
        output = tf.layers.conv2d(output, channel_numbers[-1], [kernel_sizes[-1], kernel_sizes[-1]], padding="same",
                                  activation=None)

        loss = tf.losses.mean_squared_error(labels=target, predictions=output)

        return train_init, val_init, input, target, output, loss, is_training

    def run_training(self):
        self.sess.run(self.init)

        i = 0
        while True:
            self.train()
            self.validate(i)
            if i % 25 == 0:
                self.test(i)
            i += 1

    def train(self, num_steps=250):
        self.sess.run(self.train_init)
        loss = 0.

        for i in range(num_steps):
            try:
                _, l = self.sess.run([self.train_op, self.loss])
                loss += l
            except tf.errors.OutOfRangeError:
                break
        summ, step = self.sess.run([self.train_summaries, self.global_step], feed_dict={self.loss: loss / num_steps})
        self.summary_writer.add_summary(summ, step)

    def validate(self, i):
        self.sess.run(self.val_init)
        val_loss = 0
        n = 0
        while True:
            try:
                inp, targets, outp, l = self.sess.run([self.input, self.target, self.output, self.loss],
                                                      feed_dict={self.is_training: False})
                val_loss += l
                n += 1

                if n > i or i % 25 != 0:
                    continue

                path = '../data/val/{:05d}'.format(i)
                os.makedirs(path, exist_ok=True)
                for i in range(min(50, inp.shape[0])):
                    img = np.clip(inp[i], 0, 1) * 255.
                    pred = np.clip(outp[i], 0, 1) * 255.
                    target = np.clip(targets[i], 0, 1) * 255.
                    stacked_imgs = np.vstack([img, pred, target]).astype(np.uint8)
                    imwrite(os.path.join(path, "img_{:05d}.png".format(i)), stacked_imgs)
            except tf.errors.OutOfRangeError:
                break
        print(i, "Val Loss: ", val_loss / n)

        summ, step = self.sess.run([self.val_summaries, self.global_step], feed_dict={self.loss: val_loss / n})
        self.summary_writer.add_summary(summ, step)
        self.summary_writer.flush()

    def test(self, i):
        path = '../data/test/{:05d}'.format(i)
        os.makedirs(path, exist_ok=True)

        test_files = glob.glob('../data/test_images/*.png')

        for i, f in enumerate(test_files):
            img = self.sess.run(self.load_image_op, feed_dict={self.filename_pl: f})

            img = np.reshape(img, newshape=[28, 28, 1])
            pred = self.sess.run(self.output, feed_dict={self.input: [img], self.is_training: False})[0]

            pred = np.clip(pred, 0, 1)
            stacked_imgs = np.vstack([img, pred]) * 255
            imwrite(os.path.join(path, "img_{:05d}.png".format(i)), stacked_imgs.astype(np.uint8))
            if i > 500:
                break


def main():
    m = Model()
    m.run_training()


if __name__ == '__main__':
    main()

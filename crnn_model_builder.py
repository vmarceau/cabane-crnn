import tensorflow as tf
from typing import Any, Dict, Tuple


def ctc_batch_cost(args: Any) -> Any:
    y_pred, labels, input_length, label_length = args
    shift = 2
    y_pred = y_pred[:, shift:, :]
    return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)


def stack_last_axes(x: Any) -> Any:
    old_shape = tf.keras.backend.int_shape(x)
    new_shape = [-1, old_shape[1], old_shape[2] * old_shape[3]]
    y = tf.reshape(x, new_shape)
    return y


class CRNNModelBuilder:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._cnn_config = config['cnn']
        self._cnn_num_features = self._cnn_config['num_features']
        self._cnn_convs_per_pool = self._cnn_config['convs_per_pool']
        self._cnn_kernel_sizes = self._cnn_config['kernel_sizes']
        self._cnn_pool_sizes = self._cnn_config['pool_sizes']

        self._rnn_config = config['rnn']
        self._rnn_type = self._rnn_config['type']
        self._rnn_input_features = self._rnn_config['input_features']
        self._rnn_hidden_units = self._rnn_config['hidden_units']
        self._rnn_dropout = self._rnn_config['dropout']
        self._rnn_recurrent_dropout = self._rnn_config['recurrent_dropout']

    def build(
        self, input_shape: Tuple[int, int, int], num_classes: int, max_sequence_length: int, training: bool = True
    ) -> tf.keras.models.Model:
        input = x = tf.keras.layers.Input(name='input_images', shape=input_shape, dtype='float32')

        x = self._build_cnn(x)
        x = self._build_cnn_to_rnn_transition(x)
        x = self._build_rnn(x, num_classes=num_classes)

        model_inputs, model_outputs = self._build_inputs_and_outputs(
            image_input=input, softmax_output=x, max_sequence_length=max_sequence_length, training=training
        )

        return tf.keras.models.Model(inputs=model_inputs, outputs=model_outputs)

    def _build_cnn(self, x: Any) -> Any:
        for i, (depth, n_convs, kernel_size, pool_size) in enumerate(
            zip(self._cnn_num_features, self._cnn_convs_per_pool, self._cnn_kernel_sizes, self._cnn_pool_sizes)
        ):
            for j in range(n_convs):
                x = tf.keras.layers.Conv2D(depth, kernel_size, padding='same', name=f'conv_{i}_{j}', activation='relu')(
                    x
                )
                x = tf.keras.layers.BatchNormalization(name=f'norm_{i}_{j}')(x)
            x = tf.keras.layers.MaxPooling2D(pool_size=pool_size, name=f'pool_{i}')(x)

        return x

    def _build_cnn_to_rnn_transition(self, x: Any) -> Any:
        x = tf.keras.layers.Lambda(stack_last_axes, name='reshape')(x)
        x = tf.keras.layers.Dense(self._rnn_input_features, activation='relu', name='rnn_input')(x)
        return x

    def _build_rnn(self, x: Any, num_classes: int) -> Any:
        for i, n_units in enumerate(self._rnn_hidden_units):
            if self._rnn_type.lower() == 'lstm':
                rnn1 = tf.keras.layers.LSTM(
                    n_units,
                    return_sequences=True,
                    dropout=self._rnn_dropout,
                    recurrent_dropout=self._rnn_recurrent_dropout,
                    name=f'lstm_{i}_1',
                )(x)
                rnn2 = tf.keras.layers.LSTM(
                    n_units,
                    return_sequences=True,
                    go_backwards=True,
                    dropout=self._rnn_dropout,
                    recurrent_dropout=self._rnn_recurrent_dropout,
                    name=f'lstm_{i}_2',
                )(x)
                x = tf.keras.layers.concatenate([rnn1, rnn2], name=f'rnn_concat_{i}')
            elif self._rnn_type.lower() == 'gru':
                rnn1 = tf.keras.layers.GRU(
                    n_units,
                    return_sequences=True,
                    dropout=self._rnn_dropout,
                    recurrent_dropout=self._rnn_recurrent_dropout,
                    name=f'gru_{i}_1',
                )(x)
                rnn2 = tf.keras.layers.GRU(
                    n_units,
                    return_sequences=True,
                    go_backwards=True,
                    dropout=self._rnn_dropout,
                    recurrent_dropout=self._rnn_recurrent_dropout,
                    name=f'gru_{i}_2',
                )(x)
                x = tf.keras.layers.concatenate([rnn1, rnn2], name=f'rnn_concat_{i}')
            else:
                raise ValueError('Unsupported RNN type... Use either LSTM or GRU.')

        x = tf.keras.layers.Dense(num_classes + 1, name='rnn_output')(x)
        x = tf.keras.layers.Activation('softmax', name='softmax')(x)
        return x

    def _build_inputs_and_outputs(
        self, image_input: Any, softmax_output: Any, max_sequence_length: int, training: bool
    ) -> Tuple[Any, Any]:
        input_labels = tf.keras.layers.Input(name='input_labels', shape=[max_sequence_length], dtype='float32')
        input_length = tf.keras.layers.Input(name='input_length', shape=[1], dtype='int64')
        label_length = tf.keras.layers.Input(name='label_length', shape=[1], dtype='int64')

        loss_output = tf.keras.layers.Lambda(ctc_batch_cost, output_shape=(1,), name='ctc')(
            [softmax_output, input_labels, input_length, label_length]
        )

        model_inputs = [image_input]
        if training:
            model_inputs += [input_labels, input_length, label_length]
            model_outputs = loss_output
        else:
            model_outputs = softmax_output

        return model_inputs, model_outputs


if __name__ == '__main__':
    model_config = {
        'cnn': {
            'num_features': [16, 32, 64, 128],
            'convs_per_pool': [2, 2, 2, 2],
            'kernel_sizes': [(3, 3), (3, 3), (3, 3), (3, 3)],
            'pool_sizes': [(2, 2), (2, 2), (1, 2), (1, 2)],
        },
        'rnn': {
            'type': 'lstm',
            'input_features': 128,
            'hidden_units': [32, 16],
            'dropout': 0.1,
            'recurrent_dropout': 0.1,
        },
    }

    builder = CRNNModelBuilder(model_config)
    model = builder.build((280, 32, 1), num_classes=40, max_sequence_length=10, training=True)
    model.summary()

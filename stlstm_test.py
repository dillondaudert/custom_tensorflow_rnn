import stlstm
import tensorflow as tf
from tensorflow import test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import init_ops

class STLSTMTest(test.TestCase):

    def test_cell_build(self, *args, **kwargs):
        """Check that state transition variables are created properly."""
        num_units=5
        num_layers=2
        inputs_shape=[2,8]
        st_kernel_initializer=init_ops.zeros_initializer()

        with self.test_session() as sess:
            cell = stlstm.STLSTMCell(num_units,
                                     transition_kernel_initializer=st_kernel_initializer,
                                     transition_num_layers=num_layers)
            cell.build(inputs_shape)
            # check cell._st_kernels/biases
            # NOTE: check length of new variable arrays
            self.assertEqual(num_layers, len(cell._st_kernels))
            self.assertEqual(len(cell._st_kernels), len(cell._st_biases))

            # NOTE: check sizes of new variable arrays
            for layer in range(num_layers):
                self.assertAllEqual(cell._st_kernels[layer].shape, [num_units, num_units])
                self.assertEqual(cell._st_biases[layer].shape[0], num_units)


if __name__ == '__main__':
    tf.test.main()

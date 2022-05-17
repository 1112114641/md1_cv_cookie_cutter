from src.abstracts import BaseModelABC
import tensorflow as tf
from tensorflow_addons.losses import focal_loss

# https://github.com/Ahmkel/Keras-Project-Template/blob/master/trainers/simple_mnist_trainer.py
# if hasattr(self.config,"comet_api_key"):
#     from comet_ml import Experiment
#     experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
#     experiment.disable_mp()
#     experiment.log_multiple_params(self.config)
#     self.callbacks.append(experiment.get_keras_callback())


# https://www.tensorflow.org/tutorials/distribute/keras
strategy = tf.distribute.MirroredStrategy()


def decay(epoch):
    if epoch < 3:
        return 1e-3
    elif epoch >= 3 and epoch < 7:
        return 1e-4
    else:
        return 1e-5


# cross entropy as pollyloss
def poly1_cross_entropy(logits, labels, epsilon=1.0):
    """ Poly-1 loss as introduced in https://arxiv.org/pdf/2204.12511.pdf."""
    # pt, CE, and Poly1 have shape [batch].
    pt = tf.reduce_sum(labels * tf.nn.softmax(logits), axis=-1)
    CE = tf.nn.softmax_cross_entropy_with_logits(labels, logits)
    Poly1 = CE + epsilon * (1 - pt)
    return Poly1


# focal loss as pollyloss
def poly1_focal_loss(logits, labels, epsilon=1.0, gamma=2.0):
    # p, pt, FL, and Poly1 have shape [batch, num of classes].
    p = tf.math.sigmoid(logits)
    pt = labels * p + (1 - labels) * (1 - p)
    FL = focal_loss(pt, gamma)
    Poly1 = FL + epsilon * tf.math.pow(1 - pt, gamma + 1)
    return Poly1


class EfficientNetV2(BaseModelABC):
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)

    def load_data(self):
        pass

    def build_model(self):
        pass

    def custom_lr_schedule(self):
        class PrintLR(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print(
                    f"\nLearning rate for epoch {epoch + 1} is {self.model.optimizer.lr.numpy()}"
                )

        def decay(epoch):
            if epoch < 3:
                return 1e-3
            elif epoch >= 3 and epoch < 7:
                return 1e-4
            else:
                return 1e-5

        self.callbacks.append(tf.keras.callbacks.LearningRateScheduler(decay))
        self.callbacks.append(PrintLR())

    def train(self):
        pass

    def evaluate(self):
        pass

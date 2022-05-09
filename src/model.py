import tensorflow as tf


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

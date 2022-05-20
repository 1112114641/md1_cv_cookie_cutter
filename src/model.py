from src.abstracts import BaseModelABC
import tensorflow as tf
from tensorflow_addons.losses import focal_loss
from src.data_loader import DataLoader

# https://github.com/Ahmkel/Keras-Project-Template/blob/master/trainers/simple_mnist_trainer.py
# if hasattr(self.config,"comet_api_key"):
#     from comet_ml import Experiment
#     experiment = Experiment(api_key=self.config.comet_api_key, project_name=self.config.exp_name)
#     experiment.disable_mp()
#     experiment.log_multiple_params(self.config)
#     self.callbacks.append(experiment.get_keras_callback())
DENSE_KERNEL_INITIALIZER = {
    "class_name": "VarianceScaling",
    "config": {"scale": 1 / 3.0, "mode": "fan_out", "distribution": "uniform"},
}


class Model(BaseModelABC):
    """
    TODO: do the thing.
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config)
        if len(tf.config.list_physical_devices("GPU")) > 1:
            # https://www.tensorflow.org/tutorials/distribute/keras
            self.strategy = tf.distribute.MirroredStrategy()
        self.dl = DataLoader(config)
        self.model_build()

    def load_data(self) -> None:
        self.train, self.valid, self.test = self.dl.data_generator()

    def model_build(self):
        """
        Do be careful with model input preprocessing!
        e.g. tf.keras.applications.efficientnet_v2.preprocess_input
        or  tf.keras.applications.resnet.preprocess_input
        expecting RGB / BGR respectively, and normalisations.
        """
        inputs = tf.keras.layers.Input(shape=(None, None, 3))

        # Generate backbone from configs
        if self.config.training["model_name"] == "ResNet50":
            model = tf.keras.applications.resnet50.ResNet50(
                include_top=False,
                weights=(
                    self.config.training["path_to_weights"]
                    if self.config.training["path_to_weights"]
                    else "imagenet"
                ),
                input_shape=(None, None, 3),
                pooling="avg",
            )(inputs)
        else:
            model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
                include_top=False,
                weights=(
                    self.config.training["path_to_weights"]
                    if self.config.training["path_to_weights"]
                    else "imagenet"
                ),
                input_shape=(None, None, 3),
                pooling="avg",
                include_preprocessing=True,
            )(inputs)

        # Add top layer with size=classes
        outputs = tf.keras.layers.Dense(
            self.config.training["number_of_classes"],
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            kernel_regularizer=tf.keras.regularizers.l2(5e-6),
            bias_regularizer=tf.keras.regularizers.l2(5e-6),
            name="logits",
            activation="linear",
        )(model)
        self.model = tf.keras.Model(
            inputs, outputs, name=self.config.training["model_name"]
        )

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

    def model_train(self):
        pass

    def model_evaluate(self):
        pass

    def model_load_weights(self):
        pass

    def model_strip_top_transferlearn(
        self, loc: str, activation: str = "sigmoid", name: str = "retrain"
    ) -> None:
        model = tf.keras.models.load_model(loc)
        layer_no = 2 if self.config.training["model_name"] == "ResNet50" else 3
        y_1 = tf.keras.layers.Dropout(0.2, name="top_dropout")(
            model.layers[-layer_no].output
        )
        y_1 = tf.keras.layers.Dense(
            self.config.training["number_of_classes"],
            kernel_initializer=DENSE_KERNEL_INITIALIZER,
            kernel_regularizer=tf.keras.regularizers.l2(5e-6),
            bias_regularizer=tf.keras.regularizers.l2(5e-6),
            name="logits",
            activation=activation,
        )(y_1)
        self.model = tf.keras.Model(
            inputs=model.layers[1].input, outputs=y_1, name=name
        )


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


# EXPS_DIR = 'experiments'
# class Evaluator(object):
#     def __init__(self, dataset, exp_dir, poly_degree=3):
#         self.dataset = dataset
#         # self.predictions = np.zeros((len(dataset.annotations), dataset.max_lanes, 4 + poly_degree))
#         self.predictions = None
#         self.runtimes = np.zeros(len(dataset))
#         self.loss = np.zeros(len(dataset))
#         self.exp_dir = exp_dir
#         self.new_preds = False

#     def add_prediction(self, idx, pred, runtime):
#         if self.predictions is None:
#             self.predictions = np.zeros((len(self.dataset.annotations), pred.shape[1], pred.shape[2]))
#         self.predictions[idx, :pred.shape[1], :] = pred
#         self.runtimes[idx] = runtime
#         self.new_preds = True

#     def eval(self, **kwargs):
#         return self.dataset.dataset.eval(self.exp_dir, self.predictions, self.runtimes, **kwargs)


# if __name__ == "__main__":
#     evaluator = Evaluator(LaneDataset(split='test'), exp_dir=sys.argv[1])
#     evaluator.tusimple_eval()

import rospy
import numpy as np
import tensorflow as tf
from styx_msgs.msg import TrafficLight

class TLClassifier(object):
    def __init__(self, file_path):
        #TODO load classifier
        self.classifier_model = tf.Graph()
        self.min_score_threshold = 0.5
        with self.classifier_model.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(file_path, 'rb') as mf:
                ser_graph = mf.read()
                graph_def.ParseFromString(ser_graph)
                tf.import_graph_def(graph_def, name='')
                rospy.loginfo("loaded graph from frozen model")
            self.image_tensor = self.classifier_model.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.classifier_model.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.classifier_model.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.classifier_model.get_tensor_by_name('detection_classes:0')
            self.num_d = self.classifier_model.get_tensor_by_name('num_detections:0')
        config = tf.ConfigProto()
        jl = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jl
        self.sess = tf.Session(config=config, graph=self.classifier_model)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        light_state = TrafficLight.UNKNOWN

        with self.classifier_model.as_default():
            img = np.expand_dims(image, axis=0)  
            (boxes, scores, classes, num_detections) = self.sess.run(
                    [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                    feed_dict={self.image_tensor: img})
            boxes = np.squeeze(boxes)
            scores = np.squeeze(scores)
            classes = np.squeeze(classes).astype(np.int32)

            for i in range(boxes.shape[0]):
                if scores is None or scores[i] > self.min_score_threshold:
                    if classes[i] == 1:
                        rospy.loginfo('Light Detected: GREEN')
                        light_state = TrafficLight.GREEN
                    elif classes[i] == 2:
                        rospy.loginfo('Light Detected: RED')
                        light_state = TrafficLight.RED
                    elif classes[i] == 3:
                        rospy.loginfo('Light Detected: YELLOW')
                        light_state = TrafficLight.YELLOW
                    else:
                        rospy.loginfo('No Light Detected')
        return light_state

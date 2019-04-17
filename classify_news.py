import sys
import prepare
import tensorflow as tf
import numpy as np
'''
raw_data is the web scraped news and the input needs to be a string.

cleandata is the output of our cleanning process. Functions in prepare.py are called.

If cleandata is qualified, we convert the words into numbers.

we use numbers as input for CNN and predict the category for news.
'''
raw_data = sys.argv[1]
cleandata = prepare.load_data_and_labels_another(raw_data)


if cleandata is not None:
    transdata = prepare.word_id_convert(cleandata)
    # convert the data into a (1,300) shape.
    a = transdata.reshape(-1,1)
    tdata = np.transpose(a)
    with tf.Session() as sess:
        # restore the CNN model
        saver = tf.train.import_meta_graph('./my_model_probability/my_model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./my_model_probability'))

        graph = tf.get_default_graph()
		# access and create placeholders and feed-dict to feed new data
        input_x = graph.get_tensor_by_name("input_x:0")
        op_to_restore = graph.get_tensor_by_name("predictions:0")
        prob = graph.get_tensor_by_name("predictions_b:0")
        dropout_keep_prob = graph.get_tensor_by_name("dropout_keep_prob:0")
        feed_dict = {input_x:tdata, dropout_keep_prob: 0.5}
        #sess.run(tf.global_variables_initializer())
        predictions = sess.run(op_to_restore, feed_dict)    
        predictions_b = sess.run(prob, feed_dict)
        # print(predictions)
        # print(predictions_b)
        # The sequence of topics strictly obey the sequence of label in clean_helper.py
        topics =["Financial","International","Legal","Social","Tech,Sci&Innov","Hemp"]
        for idx, topic in enumerate(topics):
            probability = predictions_b[0][idx]
            probability = round(probability,2)
            print([topic, probability])

    
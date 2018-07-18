import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import os


class Params():#used to store parameter values
    def __init__(self,params):
        #STORING PARAMETER VALUES 
        self.input_shape=params['input_shape']
        self.num_outputs=params['num_outputs']
        self.layer_hierarchy=params['layer_hierarchy']
        self.activation_fn=params.get('activation_fn',tf.nn.relu)
        self.loss_fn=params.get('loss_fn',tf.losses.softmax_cross_entropy)
        self.learning_rate=params['learning_rate']
        self.optimizer_fn=params['optimizer_fn']
        self.initializer_fn=params['initializer_fn']
        self.name_scope=params['name_scope']
#         self.step_no=0


class CNN_Model():
    def __init__(self,param_dict={},restore_params=False,pickle_file_path=""):
        #STORING PARAMETER VALUES
        if not restore_params:
            self.params=Params(param_dict)
        else:
            self.restore_params_fn(pickle_file_path)

    def form_placeholder(self,shape,dt=tf.float32):
        X=tf.placeholder(dt,shape=shape)
        return X
    def form_variable(self,shape,dt=tf.float32,name="",trainable=True,initializer=tf.zeros_initializer):
        if name=="":
            return tf.Variable(initial_value=initializer,trainable=trainable,dtype=dt)
        else:
#             initializer=tf.constant_initializer(initail_val)
            return tf.get_variable(name=name,shape=shape, dtype=dt,initializer=initializer(),trainable=trainable)
    
    def form_convolutional_layer(self,inputs,layer_params):
        return tf.layers.conv2d(
                    inputs=inputs,
                    filters=layer_params['num_filters'],
                    kernel_size=layer_params['kernel_size'],
                    strides=layer_params['kernel_strides'],
                    padding=layer_params['padding'],
                    kernel_initializer=self.params.initializer_fn(),
                    activation=None)
    def form_max_pooling_layer(self,inputs,layer_params):
        tf.layers.max_pooling2d(
                    inputs=inputs,
                    pool_size=layer_params['pool_size'],
                    strides=layer_params['pool_strides'])
    def form_activation_layer(self,inputs):
        return self.params.activation_fn(inputs)
    
    def form_fc_layer(self,inputs,layer_params):
        return tf.layers.dense(inputs,layer_params['num_hidden_units'],activation=None,kernel_initializer=self.params.initializer_fn())
    
    def form_batch_normalization_layer(self,inputs):
        return tf.layers.batch_normalization(inputs=inputs,axis=-1,training=self.training_mode)
    
    def form_dropout_layer(self,inputs,layer_params):
        dropout_probability=layer_params.get('dropout_probability',0.5)
        noise_shape=layer_params.get('dropout_mask_shape',None)
        return tf.layers.dropout(inputs,rate=dropout_probability,noise_shape=noise_shape,training=self.training_mode)
    
    def form_loss(self,logits,targets):
        entropies=self.params.loss_fn(onehot_labels=targets,logits=logits,reduction=tf.losses.Reduction.NONE)
        return entropies
    
    def build_model_till_logits(self):
        with tf.variable_scope(self.params.name_scope):
            self.X=self.form_placeholder(self.params.input_shape)
            self.lr_placeholder=self.form_placeholder([]) #since we can change learning arate during training
            self.training_mode=self.form_placeholder([],tf.bool)
            self.step_no=self.form_variable(shape=[],name="step_no",trainable=False)#stores number of steps for which training has occured
            self.epoch_no=self.form_variable(shape=[],name="epoch_no",trainable=False) #stores number of epochs for which training is performed

            inputs=self.X
            for layer_params in self.params.layer_hierarchy:
                if layer_params['layer_type']=='conv_layer':
                    inputs=self.form_convolutional_layer(inputs,layer_params)
                elif layer_params['layer_type']=='fc_layer':
                    inputs=self.form_fc_layer(inputs,layer_params)
                elif layer_params['layer_type']=='activation_layer':
                    inputs=self.form_activation_layer(inputs)
                elif layer_params['layer_type']=='pooling_layer':
                    inputs=self.form_max_pooling_layer(inputs,layer_params)
                elif layer_params['layer_type']=='flattening_layer':
                    inputs=tf.contrib.layers.flatten(inputs)
                elif layer_params['layer_type']=='batch_normalization_layer':
                    inputs=self.form_batch_normalization_layer(inputs)
                elif layer_params['layer_type']=='dropout_layer':
                    inputs=self.form_dropout_layer(inputs,layer_params)

    #         making logits layer (final output layer)
            self.logits=tf.layers.dense(inputs,self.params.num_outputs,activation=None,kernel_initializer=self.params.initializer_fn())
            
            self.model_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.params.name_scope)#saving only the varuiables belonging to this scope
            self.saver=tf.train.Saver(var_list=self.model_variables)
            self.increment_epoch_op=tf.assign(self.epoch_no, self.epoch_no+1)#op to update number of epoch by + 1
            self.model_trainable_variables=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.params.name_scope)

    def Build_model(self):
        self.build_model_till_logits()
        with tf.variable_scope(self.params.name_scope):
            
            self.Y=self.form_placeholder((None,self.params.num_outputs),tf.float32)
            self.loss=tf.reduce_mean(self.form_loss(self.logits,self.Y))            

            self.predictions=tf.argmax(tf.nn.softmax(self.logits),1)
            equality = tf.equal(self.predictions,tf.argmax(self.Y,1))
            self.accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

            optimizer=self.params.optimizer_fn(learning_rate=self.lr_placeholder)
            self.grads_and_vars=optimizer.compute_gradients(loss=self.loss,var_list=self.model_trainable_variables)
            self.train_op=optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,global_step=self.step_no)
#             self.train_op=optimizer.minimize(loss = self.loss,global_step=self.step_no)


            #summary ops
            loss_summary=tf.summary.scalar("loss",self.loss)
            acc_summary=tf.summary.scalar("accuracy",self.accuracy)
            #             self.summaries=tf.summary.merge_all(scope=self.params.name_scope)
            self.summaries=tf.summary.merge([loss_summary,acc_summary])
            self.initializer=tf.global_variables_initializer()
    def create_log_directory_if_doesnt_exist(self,savedir):
        savedir=os.path.join(os.getcwd(),savedir)
        savedir=os.path.join(savedir,self.params.name_scope)
        savedir=os.path.join(savedir,"logs")
        if not os.path.isdir(savedir):#creating directory if not exists
            try:  
                os.makedirs(savedir)
                return savedir,True
            except OSError:
                print ('failed to make the specified_directory.Returning...')
                return "",False
        return savedir,True
            
        
    def save_model(self,sess,savedir="/",step=0):
        savedir=os.path.join(os.getcwd(),savedir)
        savedir=os.path.join(savedir,self.params.name_scope)
        if not hasattr(self,'saved_before'):#calling save model for the first time
            if not os.path.isdir(savedir):#creating directory if not exists
                try:  
                    os.makedirs(savedir)
                except OSError:
                    print ('failed to make the specified_directory.Returning...')
                    return
            file_pi = open(os.path.join(savedir,"model_object.pkl"), 'wb+') #saving param object
            pickle.dump(self.params, file_pi)
            #saving tensorflow graph and weight values
#             path=os.path.join(savedir,(self.params.name_scope+".ckpt"))
#             print ("saving path:{}".format(str(savedir)))
            self.saver.save(sess,os.path.join(savedir,"model_weights.ckpt"), global_step=step) #saving model weights
            self.saved_before=True
        else:    #saving model weights
            self.saver.save(sess,os.path.join(savedir,"model_weights.ckpt"), global_step=step,write_meta_graph=False)#writes meta graph for the first time save_model is called
    def restore_params_fn(self,pickle_file_path):
        if os.path.exists(pickle_file_path):
            filehandler = open(pickle_file_path, 'rb')
            self.params=pickle.load(filehandler)
        else:
            print("no such file exists")
        
    def restore_model(self,sess,restore_dir):
        restore_dir=os.path.join(os.getcwd(),restore_dir)
#         restore_dir=os.path.join(restore_dir,"\\")
#         path=restore_dir
        restore_dir=os.path.join(restore_dir,self.params.name_scope)
        print ("restoring path:{}".format(str(restore_dir)))
#         print (path)
        self.saver.restore(sess, tf.train.latest_checkpoint(restore_dir))#loading latest model
         
    def train(self,sess,n_epochs,get_next_batch_fn,get_validation_set_fn,save_every_n_iter,log_train_every_n_iter,log_validation_every_n_iter,save_dir,initialize=False,set_logging=True):
        if initialize:
            sess.run([self.initializer])
        if set_logging:
            log_dir,set_logging=self.create_log_directory_if_doesnt_exist(save_dir)
        if set_logging: #creating file handlers if dir cretaed or found in above statement
            train_writer = tf.summary.FileWriter(os.path.join(log_dir,'train'), sess.graph)
            validation_writer = tf.summary.FileWriter(os.path.join(log_dir ,'validation'))
        [step_no]=sess.run([self.step_no]) 
        [epoch]=sess.run([self.epoch_no])
        ending_epoch=n_epochs+epoch
        while epoch < ending_epoch:
            print("----Epoch="+str(epoch)+"\n")
            for x,y in get_next_batch_fn():
                feed_dict={self.X:x,self.Y:y,self.lr_placeholder:self.params.learning_rate,self.training_mode:True}
                if step_no%log_train_every_n_iter==0 and set_logging:
                    summaries,loss,acc,new_step_no,_=sess.run([self.summaries,self.loss,self.accuracy,self.step_no,self.train_op],feed_dict=feed_dict)
                    train_writer.add_summary(summaries,step_no)
                else:
                    loss,acc,new_step_no,_=sess.run([self.loss,self.accuracy,self.step_no,self.train_op],feed_dict=feed_dict)
                print ("Step={} and loss occured= {} and acc= {} \n".format(str(step_no),str(loss),str(acc)))
                feed_dict=None #freeing memory
                x=y=None
#                 if step_no%log_validation_every_n_iter==0 and set_logging:
#                     x,y=get_validation_set_fn()
#                     feed_dict={self.X:x,self.Y:y,self.training_mode:False}
#                     [summaries]=sess.run([self.summaries],feed_dict=feed_dict)
#                     validation_writer.add_summary(summaries, step_no)

                if (step_no)%save_every_n_iter==0:
                    print("saving model\n")
                    self.save_model(sess,save_dir,self.step_no)
                step_no=new_step_no
#                 print ("""loss= "+str(loss)+"\n")
            _,epoch=sess.run([self.increment_epoch_op,self.epoch_no])
    
                

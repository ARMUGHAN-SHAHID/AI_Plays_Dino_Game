
# coding: utf-8

# In[1]:


import tensorflow as tf
import os 
import numpy as np
import pandas as pd
from model import CNN_Model,Params
from Env import Env
from time import sleep


# In[2]:


class experience_replay_buffer:
#     def __init__(self,size,dtypes):
#         self.column_names=['state','action','next_state','reward','done']
#         self.buffer={self.column_names[i]:np.empty(size,dtype=dtypes[i]) for i in np.arange(len(self.column_names)) }
#         self.num_items=0
#         self.capacity=size
    def __init__(self,buffer_len,sample):
        self.column_names=['state','action','next_state','reward','done']
        self.buffer={col_name:np.empty(shape=[buffer_len,*np.array(item).shape],dtype=np.array(item).dtype) for col_name,item in zip(self.column_names,sample) }
        self.num_items=0
        self.capacity=buffer_len
        self.add_ind=0
    def add_experience(self,state,action,next_state,reward,done):
        ind=self.add_ind
        self.add_ind=(self.add_ind+1)%self.capacity
#         ind=self.num_items
        if self.num_items<self.capacity:
            self.num_items+=1
#         else:
#             ind=np.random.randint(low=0,high=self.capacity,size=1,dtype=np.int32)
            
        self.buffer['state'][ind]=state
        self.buffer['action'][ind]=action
        self.buffer['next_state'][ind]=next_state
        self.buffer['reward'][ind]=reward
        self.buffer['done'][ind]=done
           
    
    def get_batch(self,batch_size):
        inds=np.random.randint(low=0,high=self.num_items,size=batch_size,dtype=np.int32)
        return self.buffer['state'][inds],self.buffer['action'][inds],self.buffer['next_state'][inds],self.buffer['reward'][inds],self.buffer['done'][inds]
    


# In[3]:


class Q_Network(CNN_Model):
    def __init__(self,max_experience_buffer_len=120,param_dict={},restore_params=False,pickle_file_path=""):
        CNN_Model.__init__(self,param_dict,restore_params,pickle_file_path)
        self.max_experience_buffer_len=max_experience_buffer_len
        
    def form_loss(self,logits,targets):
        entropies=self.params.loss_fn(labels=targets,logits=logits)
        return entropies
        
    def Build_model(self):
        self.build_model_till_logits()
        with tf.variable_scope(self.params.name_scope):
            #logits are q values]
            self.max_q_value_actions=tf.squeeze(tf.argmax(self.logits,axis=1)) #value which has the highest q value
            self.max_q_values=tf.reduce_max(self.logits,axis=1)
            
            #placeholder for action at current timestep
            self.actions=self.form_placeholder((None),tf.int32)
            one_hot_actions=tf.one_hot(indices=self.actions,depth=self.params.num_outputs)
            q_vals=tf.reduce_sum(self.logits*one_hot_actions,axis=1)
            
            
            
            #placeholder for max next state q values,rewards and discount rate
            self.max_q_values_next_state=self.form_placeholder((None),tf.float32)
            self.rewards=self.form_placeholder((None),tf.float32)
            self.notended=self.form_placeholder((None),tf.float32)
            self.discount_rate=self.form_placeholder([],tf.float32)
            
            self.loss=tf.reduce_mean(tf.square((self.rewards+(self.discount_rate*(self.max_q_values_next_state*self.notended)))-q_vals))
#             self.qvalues_next=self.max_q_values_next_state*self.notended
#             self.discounted_qvalues_next=self.discount_rate*self.qvalues_next
#             self.targets=self.rewards+self.discounted_qvalues_next
#             self.diff=self.targets-q_vals
#             self.squared_diff=tf.square(self.diff)
#             self.loss=tf.reduce_mean(self.squared_diff)
        
            #computing gradients 
            optimizer=self.params.optimizer_fn(learning_rate=self.lr_placeholder)
            self.grads_and_vars=optimizer.compute_gradients(loss=self.loss,var_list=self.model_trainable_variables)
            
            self.train_op=optimizer.apply_gradients(grads_and_vars=self.grads_and_vars,global_step=self.step_no)
            self.model_variables=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.params.name_scope)
            self.saver=tf.train.Saver(var_list=self.model_variables)

            self.initializer=tf.global_variables_initializer()
    
    def add_to_experience_replay(self,state,action,next_state,reward,done):
        
        if not hasattr(self,"experience_replay_buffer"):
            sample=[state,action,next_state,reward,done]
            self.experience_replay_buffer=experience_replay_buffer(buffer_len=self.max_experience_buffer_len,sample=sample)
        self.experience_replay_buffer.add_experience(state,action,next_state,reward,done)



    def train(self,sess,episodes,steps,initial_epsilon,final_epsilon,epsilon_dec,train_t,discount_rate,batch_size,env,save_dir,save_every_n_iter,log_every_n_iter,initialize=False,set_logging=True,num_frames_to_repeat_action=4):
        if initialize:
            print ("Initializing.....\n")
            sess.run([self.initializer])
        if set_logging:
            print ("Setting up for Logging ...\n")
            log_dir,set_logging=self.create_log_directory_if_doesnt_exist(save_dir)
        if set_logging: #creating file handlers if dir cretaed or found in above statement
            print("Logging called but no code implemented")
#                 train_writer = tf.summary.FileWriter(os.path.join(log_dir,'train'), sess.graph)
#                 validation_writer = tf.summary.FileWriter(os.path.join(log_dir ,'validation'))
        print ("Retreiveing step no...\n")
        [iter_no]=sess.run([self.step_no]) 
        epsilon=initial_epsilon
        for episode in np.arange(episodes):
            state=env.reset()
            step=0
            episode_reward=0
            episode_loss=0
            previous_action=None
            for step in np.arange(steps):
                #choosing action 
#                 action=-1
                if step%num_frames_to_repeat_action==0:
                    if epsilon>final_epsilon and iter_no>train_t:
                            epsilon-=epsilon_dec

    #                 if  iter_no<train_t or (np.random.random(1)<epsilon):
                    if  (np.random.random(1)<epsilon):
                        action=np.random.randint(low=0,high=self.params.num_outputs,size=1,dtype=np.int32)
                    else:
                        feed_dict={self.X:np.expand_dims(state,axis=0),self.lr_placeholder:self.params.learning_rate,self.training_mode:True}
                        [action]=sess.run([self.max_q_value_actions],feed_dict=feed_dict)
                    action=np.squeeze(action)
                else:
                    action=previous_action
#                 print(action)
                next_state,reward,done,info=env.step(action)
                episode_reward+=reward
                self.add_to_experience_replay(state,action,next_state,reward,done)
                previous_action=action
                episode_has_finished=done

                state=next_state

                if self.experience_replay_buffer.num_items>train_t: #perform training if there are enough experiences
#                     print("buffer filled")
                    
                    
                    #performing training step
                    states,actions,next_states,rewards,dones=self.experience_replay_buffer.get_batch(batch_size=batch_size)
                    

                    #finding vals of next states
#                     print (next_states.shape)
                    feed_dict={self.X:next_states,self.lr_placeholder:self.params.learning_rate,self.training_mode:True}
                    [max_q_vals_next_state]=sess.run([self.max_q_values],feed_dict=feed_dict)

                    feed_dict={self.X:states,self.actions:actions,self.max_q_values_next_state:max_q_vals_next_state,self.rewards:rewards,self.notended:((np.logical_not(dones)).astype(np.int32)),self.discount_rate:discount_rate,self.lr_placeholder:self.params.learning_rate,self.training_mode:True}
                    loss,_=sess.run([self.loss,self.train_op],feed_dict=feed_dict)
                    episode_loss+=loss
                    iter_no+=1
                    if (iter_no)%save_every_n_iter==0:
                        print("^^^^ saving model ^^^^ \n")
                        self.save_model(sess,save_dir,self.step_no)
                    if (iter_no)%log_every_n_iter==0:
                        print ("Trainaing Step:\t Iteration no={} Game Step ={} loss={} ".format(iter_no,step,loss))
                if episode_has_finished:
                    break
            print ("===================>Episode {} Ended <===================\n".format(episode)) 
            print ("=======>\t Episode Length={} \t<=======\n".format(step))   
            print ("=======>\t Episode Reward={} \t<=======\n".format(episode_reward))
            print ("=======>\t Mean Episode Loss={} \t<=======\n".format(episode_loss/step))
            
    def test(self,sess,initialize,env,sleep_time=0.5):
        if initialize:
            print ("Initializing.....\n")
            sess.run([self.initializer])
            
        state=env.reset()
        done=False
        while not done:
            feed_dict={self.X:np.expand_dims(state,axis=0),self.lr_placeholder:self.params.learning_rate,self.training_mode:False}
            [action]=sess.run([self.max_q_value_actions],feed_dict=feed_dict)
            action=np.squeeze(action)
            state,reward,done,info=env.step(action)
            env.render()
            sleep(sleep_time)
        env.close()


# In[4]:


# env=Env('SpaceInvaders-v0',convert_to_grayscale=True,crop=True,valid_Y=[20,-10],valid_X=[10,-10],resize=True,resize_Y=90,resize_X=70,normalize=True,num_of_frames_per_stack=4)
# params={
#     'input_shape':[None, *env.image_shape],
#     'num_outputs':env.action_space,
    
#     'layer_hierarchy':[
#         {'layer_type':'conv_layer','kernel_size':8,'kernel_strides':4,'num_filters':32,'padding':'valid'},
# #         {'layer_type':'batch_normalization_layer'},
#         {'layer_type':'activation_layer'},
#         {'layer_type':'conv_layer','kernel_size':4,'kernel_strides':2,'num_filters':64,'padding':'valid'},
# #         {'layer_type':'batch_normalization_layer'},
#         {'layer_type':'activation_layer'},
#         {'layer_type':'conv_layer','kernel_size':3,'kernel_strides':2,'num_filters':64,'padding':'valid'},
# #         {'layer_type':'batch_normalization_layer'},
#         {'layer_type':'activation_layer'},
#         {'layer_type':'flattening_layer'},
#         {'layer_type':'fc_layer','num_hidden_units':512},
# #         {'layer_type':'batch_normalization_layer'},
#         {'layer_type':'activation_layer'}
# #         {'layer_type':'dropout_layer','dropout_probability':0.2},
# #         {'layer_type':'fc_layer','num_hidden_units':50},
# # #         {'layer_type':'batch_normalization_layer'},
# #         {'layer_type':'activation_layer'}
# #         {'layer_type':'dropout_layer','dropout_probability':0.2}
        
#     ],
#     'initializer_fn':tf.contrib.layers.variance_scaling_initializer,
#     'activation_fn':tf.nn.elu,
# #     'loss_fn':tf.nn.sparse_softmax_cross_entropy_with_logits, #carefull
#     'learning_rate':0.001,
#     'optimizer_fn':tf.train.AdamOptimizer,
#     'logdir':'/tf_logs_rnn/run/',
#     'name_scope':'q_network_with_frames'
# }
# print (params['num_outputs'])


# In[6]:


# n_episodes=50
# max_steps=50000
# save_every_n_iter=50
# log_every_n_iter=50
# initialize=False
# save_dir="deep_q_saves"
# max_experience_buffer_len=10000
# initial_epsilon=0.5#1
# final_epsilon=0.0001
# epsilon_dec=0.00001
# train_t=1000
# discount_rate=0.9
# batch_size=120

# tf.reset_default_graph()

    

# model=""
# with tf.Session() as sess:
#     params['input_shape']
#     if(not initialize):
#         model=Q_Network(max_experience_buffer_len,params,restore_params=True,pickle_file_path="deep_q_saves/q_network_with_frames/model_object.pkl")
#         model.Build_model()
#         model.restore_model(sess,save_dir)
#         model.params.learning_rate=0.001
#     else:
#         model=Q_Network(max_experience_buffer_len,params,restore_params=False,pickle_file_path="deep_q_saves/q_network_with_frames/model_object.pkl")
#         model.Build_model()
    
#     model.train(sess=sess,episodes=n_episodes,steps=max_steps,initial_epsilon=initial_epsilon,final_epsilon=final_epsilon,epsilon_dec=epsilon_dec,train_t=train_t,discount_rate=discount_rate,batch_size=batch_size,env=env,save_dir=save_dir,save_every_n_iter=save_every_n_iter,log_every_n_iter=log_every_n_iter,initialize=initialize,set_logging=True)
# #     model.test(initialize=True,env=env)
#     env.close()


# In[6]:


# # a=np.arange(6).reshape(3,2)
# # b=np.empty(6,type(a))
# # print (b)
# # b[1]=a
# # print (b.shape)
# # print(type(b))
# # np.isnan(np.array(b))
# a=np.array([[1,2],[3,4]])
# # a=np.array(a)
# print (1,*a.shape)
# print (a.dtype)
# a=np.array([6])
# print (a)
# print (np.squeeze(a))


# In[5]:


# n_episodes=50
# max_steps=50000
# save_every_n_iter=20
# log_every_n_iter=50
# initialize=False
# save_dir="deep_q_saves"
# max_experience_buffer_len=10000
# initial_epsilon=1
# final_epsilon=0.0001
# epsilon_dec=0.00001
# train_t=1000
# discount_rate=0.9
# batch_size=120

# tf.reset_default_graph()

    

# model=""
# with tf.Session() as sess:
#     params['input_shape']
#     if(not initialize):
#         model=Q_Network(max_experience_buffer_len,params,restore_params=True,pickle_file_path="deep_q_saves/q_network_with_frames/model_object.pkl")
#         model.Build_model()
#         model.restore_model(sess,save_dir)
#         model.params.learning_rate=0.0005
#     else:
#         model=Q_Network(max_experience_buffer_len,params,restore_params=False,pickle_file_path="deep_q_saves/q_network_with_frames/model_object.pkl")
#         model.Build_model()
    
# #     model.train(sess=sess,episodes=n_episodes,steps=max_steps,initial_epsilon=initial_epsilon,final_epsilon=final_epsilon,epsilon_dec=epsilon_dec,train_t=train_t,discount_rate=discount_rate,batch_size=batch_size,env=env,save_dir=save_dir,save_every_n_iter=save_every_n_iter,log_every_n_iter=log_every_n_iter,initialize=initialize,set_logging=True)
#     model.test(initialize=True,env=env,sleep_time=0.05)


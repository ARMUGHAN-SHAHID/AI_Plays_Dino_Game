
# coding: utf-8

# In[26]:


import numpy as np
import gym
import cv2


# In[12]:


class Env:
    def __init__(self,env_name,convert_to_grayscale=True,crop=True,valid_Y=[0,-1],valid_X=[0,-1],resize=False,resize_Y=None,resize_X=None,normalize=True):
        
        self.steps=0
        self.env_name=env_name
        self.env=gym.make(self.env_name)
        self.convert_to_grayscale=convert_to_grayscale
        self.crop=crop
        self.valid_Y=valid_Y
        self.valid_X=valid_X
        self.resize=resize
        self.resize_Y=resize_Y
        self.resize_X=resize_X
        self.normalize=normalize
        self.action_space=self.env.action_space.n
    
        
        self.done=False
        self.image_shape=self.reset().shape
        self.done=True
    def preprocess(self,state):
        if self.convert_to_grayscale:
            state=cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            
#         state=state[40:-10,10:]
        if self.crop:
            state=state[self.valid_Y[0]:self.valid_Y[1],self.valid_X[0]:self.valid_X[1]]
        
        if self.resize:
            state=cv2.resize(state,(self.resize_X,self.resize_Y))
            
        if self.normalize:
            state=state/255.0
            
        state=np.expand_dims(state,axis=-1)
#         print (state.shape)
        return state
    
    def reset(self):
        print ("Resetting Environment...\n")
        state=self.env.reset()
        state=self.preprocess(state)
        self.steps=0
        self.done=False
        return state       
    
    def step(self,action):
        if not self.done:
            next_state,reward,self.done,info=self.env.step(action)
            next_state=self.preprocess(next_state)
            self.steps+=1
#             print('stepping')
#             cv2.imshow("Over the Clouds - gray", next_state)
#             cv2.waitKey(0)
            return next_state,reward,self.done,info
        else:
            print ("episode finished.Try Resetting the environment.Returning null values\n")
            return None,None,None,None
    def render(self):
#         cv2.imshow("Over the Clouds - gray", gray_image)
        self.env.render()
    def close(self):
        self.env.close()
        
    def __del__(self):
        self.close()


# In[19]:


# # convert_to_grayscale=True,crop=True,valid_Y=[0,-1],valid_X=[0,-1],resize=False,resize_Y=None,resize_X=None
# env=Env('SpaceInvaders-v0',convert_to_grayscale=True,crop=True,valid_Y=[20,-10],valid_X=[10,-10],resize=False,resize_Y=140,resize_X=100)
# print (env.image_shape)
# print (env.action_space)
# a_nums=env.action_space
# print (int(a_nums))
# img=env.reset()
# for i in np.arange(100):
#     env.render()
#     ns,r,d,i=env.step(2)
#     if d:
#         env.reset()
# env.close()
# cv2.destroyAllWindows()
     



# coding: utf-8

# In[2]:


import numpy as np
import gym
import cv2


# In[3]:


class Env:
    def __init__(self,env_name,convert_to_grayscale=True,crop=True,valid_Y=[0,-1],valid_X=[0,-1],resize=False,resize_Y=None,resize_X=None,normalize=True,num_of_frames_per_stack=1):
        
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
        
        self.num_of_frames_per_stack=num_of_frames_per_stack
        self.frame_stack=None
        
        self.done=False
        self.image_shape=self.reset().shape
        self.done=True
    def preprocess(self,frame):
        if self.convert_to_grayscale:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#         state=state[40:-10,10:]
        if self.crop:
            frame=frame[self.valid_Y[0]:self.valid_Y[1],self.valid_X[0]:self.valid_X[1]]
        
        if self.resize:
            frame=cv2.resize(frame,(self.resize_X,self.resize_Y))
            
        if self.normalize:
            frame=frame/255.0
            
#         state=np.expand_dims(state,axis=-1)
#         print (state.shape)
        return frame
    
    def reset(self):
        print ("Resetting Environment...\n")
        frame=self.env.reset()
        frame=self.preprocess(frame)
        self.steps=0
        self.done=False
        
        self.frame_stack=np.stack((frame  for i in np.arange(self.num_of_frames_per_stack)),axis=2)
        return self.frame_stack       
    
    def step(self,action):
        if not self.done:
            next_frame,reward,self.done,info=self.env.step(action)
            next_frame=self.preprocess(next_frame)
            self.steps+=1
#             print('stepping')
#             cv2.imshow("Over the Clouds - gray", next_state)
#             cv2.waitKey(0)
            next_frame=np.expand_dims(next_frame,axis=-1)
            self.frame_stack=np.append(next_frame,self.frame_stack[:,:,:-1],axis=-1)
            return self.frame_stack,reward,self.done,info
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


# In[7]:


# # # convert_to_grayscale=True,crop=True,valid_Y=[0,-1],valid_X=[0,-1],resize=False,resize_Y=None,resize_X=None
# env=Env('SpaceInvaders-v0',convert_to_grayscale=True,crop=True,valid_Y=[20,-10],valid_X=[10,-10],resize=False,resize_Y=140,resize_X=100,normalize=True,num_of_frames_per_stack=4)
# print (env.image_shape)
# print (env.action_space)
# a_nums=env.action_space
# print (int(a_nums))
# img=env.reset()
# print("img shape={}".format(img.shape))
# for i in np.arange(10):
# #     env.render()
#     ns,r,d,i=env.step(2)
#     print('next_stack shape={}'.format(ns.shape))
#     if d:
#         env.reset()
# env.close()
# cv2.destroyAllWindows()
     


# In[13]:


# d=deque()
# a=np.arange(6).reshape(3,2)
# b=np.arange(6,12).reshape(3,2)
# c=np.arange(12,18).reshape(3,2)
# print(a)
# print(b)
# print(c)
# d=np.stack((a,b),axis=2)
# print(d)
# e=np.append(d,np.expand_dims(c,axis=-1),axis=2)
# print (e)


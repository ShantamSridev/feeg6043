from matplotlib import pyplot as plt
from math_feeg6043 import Vector,Matrix,Identity,Transpose,Inverse,v2t,t2v,HomogeneousTransformation, polar2cartesian,gaussian, eigsorted
import matplotlib.patches as patches
import matplotlib as mpl
import numpy as np
from matplotlib.colors import hsv_to_rgb
from matplotlib.colors import LogNorm
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from collections import Counter
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
from collections import Counter
import copy
plt.rcParams["figure.figsize"] = (5,3) #make plots look nice
plt.rcParams["figure.dpi"] = 150 #make plots look nice

class plot_2dframe:
    'Library to plot 2D coordinate frames                                                    '
    'Considers a fixed references frame                                                      '
    'where h0 and h1 are pose(s) or a pose and a point                                       '
    'metadata contains  object type as point or pose and two IDs for h0 and h1, respectively '
    'Poses are defined as homogeneous matrices                                               '
    'Points are defined as homogeneous vectors relative to a homogeneous matrix (i.e., pose) '
    'edge_flag toggles whether lines between poses, or a pose and a point are shown          '

    def __init__(self, metadata, h, edge_flag = False, legend_flag = True):
        
        if metadata[0]=='point_only':            
            self.H=HomogeneousTransformation().H
            self.t=h[0]            
        
        if metadata[0]=='point' or metadata[0]=='map':            
            self.H=h[0]
            self.t=h[1]
        elif metadata[0]=='pose' or metadata[0]=='pose_gt' or metadata[0]=='pose_ref' or metadata[0]=='pose_ends':
            self.H0=h[0]
            if len(h)>1: self.H1=h[1]
            else: self.H1=self.H0
        
        elif metadata[0]=='observation':
            self.H0=h[0]
            self.H1=h[1]
            self.t = v2t(polar2cartesian(h[2][0],h[2][1]))
            self.id2=metadata[3]            
            
        self.object_type=metadata[0]
        self.id0=metadata[1]
        if len(h)>1: self.id1=metadata[2]
        else: self.id1 = self.id0
        self.edge_flag=edge_flag

        # plot
        self._fixed_frame()
        
        if metadata[0]=='point' or metadata[0]=='map' or metadata[0]=='point_only':        
            self._point()
        elif metadata[0]=='pose' or metadata[0]=='pose_gt':
            self._pose()            
        elif metadata[0]=='pose_ref' or metadata[0]=='pose_ends' :  
            self._pose_coloured() 
        elif metadata[0]=='observation':
            self._observation()                        
           
        if legend_flag == True: self._legend_2dframe()
          
    def _fixed_frame(self):
        # A coordinate frame defined by its origin & unit vectors
        origin = Vector(2)
        xhat = Vector(2)
        xhat[0] = 1
        yhat = Vector(2)
        yhat[1] = 1
        
        # Plotting 2 unit vectors    
        plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2, color='k')
        plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1, color='k')
        circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color='k',label='$e$')
        plt.gca().add_patch(circle)

    def _point(self):
        # plot the point as id1 and the pose it was seen from as id0
        if self.object_type == 'point' or self.object_type == 'map' or self.object_type == 'point_only':              
            c=self.H@self.t
            c=t2v(c)                           
            
            if self.object_type == 'map': 
                size = 0.1
                colour = 'k'            
            elif self.object_type == 'point' or self.object_type == 'point_only': 
                size = 0.05
                colour = 'r'            

            circle = plt.Circle((c[1], c[0]), size, color=colour, label=self.id1)
            plt.gca().add_patch(circle)
                
                
            H_ = HomogeneousTransformation()
            H_.H=self.H
            origin = Vector(2)
                
            origin[0] = H_.t[0]
            origin[1] = H_.t[1]
                
            xhat = Vector(2)
            xhat[0] = 1
            xhat = v2t(xhat)            
                
            yhat = Vector(2)
            yhat[1] = 1        
            yhat = v2t(yhat)
                
            xhat=H_.H_R@xhat
            yhat=H_.H_R@yhat

            xhat = t2v(xhat)
            yhat = t2v(yhat)     

            if self.object_type == 'point' or self.object_type == 'map':            
                if np.all(H_.H == HomogeneousTransformation().H):
                    plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2, color='k')
                    plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1, color='k')
                    circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color='k', label=self.id0)

                else:
                    plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2, color='b')
                    plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1, color='b')
                    circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color='b', label=self.id0)
                plt.gca().add_patch(circle)                                

                
        # visualise the edge (or line) connecting point id1 and pose id0
        if self.edge_flag == True:                
            H_ = HomogeneousTransformation()
            H_.H=self.H            
                                
            plt.plot([H_.t[1],c[1]],[H_.t[0],c[0]], 'r--', linewidth = 1)
            
    def _pose(self):
        # plot pose id0 and pose id1 that it moves to
        if self.object_type == 'pose' or self.object_type == 'pose_gt' or self.object_type == 'pose_ref':  

            if self.object_type == 'pose':                
                colour = 'b'    
                unit_factor=1

            elif self.object_type == 'pose_gt':                 
                colour = 'g'            
                unit_factor=0.5
                
            elif self.object_type == 'pose_ref':                 
                colour = 'r'            
                unit_factor=0.5                
                

            H0_ = HomogeneousTransformation()
            H0_.H=self.H0
            origin = Vector(2)
                
            origin[0] = H0_.t[0]
            origin[1] = H0_.t[1]
                
            xhat = Vector(2)
            xhat[0] = unit_factor
            xhat = v2t(xhat)            
                
            yhat = Vector(2)
            yhat[1] = unit_factor
            yhat = v2t(yhat)
                
            xhat=H0_.H_R@xhat
            yhat=H0_.H_R@yhat

            xhat = t2v(xhat)
            yhat = t2v(yhat)                
                
            plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2*unit_factor, color=colour)
            plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1*unit_factor, color=colour)
            circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color=colour, label=self.id0)
            plt.gca().add_patch(circle)
            
            H1_ = HomogeneousTransformation()
            H1_.H=self.H1
            origin = Vector(2)
                
            origin[0] = H1_.t[0]
            origin[1] = H1_.t[1]
                
            xhat = Vector(2)
            xhat[0] = unit_factor
            xhat = v2t(xhat)            
                
            yhat = Vector(2)
            yhat[1] = unit_factor
            yhat = v2t(yhat)
                
            xhat=H1_.H_R@xhat
            yhat=H1_.H_R@yhat

            xhat = t2v(xhat)
            yhat = t2v(yhat)                
                
            plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2*unit_factor, color=colour)
            plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1*unit_factor, color=colour)
            circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color=colour, label=self.id1)
            plt.gca().add_patch(circle)

            # visualise the edge (or line) connecting pose id0 and pose id1
            if self.edge_flag == True:                                
                if self.object_type == 'pose': plt.plot([H0_.t[1],H1_.t[1]],[H0_.t[0],H1_.t[0]], 'b--', linewidth = 1*unit_factor)
                if self.object_type == 'pose_gt': plt.plot([H0_.t[1],H1_.t[1]],[H0_.t[0],H1_.t[0]], 'g--', linewidth = 1*unit_factor)
                if self.object_type == 'pose_ref': plt.plot([H0_.t[1],H1_.t[1]],[H0_.t[0],H1_.t[0]], 'r--', linewidth = 1*unit_factor)                    

    def _pose_coloured(self):
        # plot pose id0 as blue and pose id1 as different colours
        if self.object_type == 'pose' or self.object_type == 'pose_gt' or self.object_type == 'pose_ref' or self.object_type == 'pose_ends':  

            if self.object_type == 'pose':                
                colour = 'b'    
                unit_factor=1

            elif self.object_type == 'pose_gt':                 
                colour = 'g'            
                unit_factor=0.5
                
            elif self.object_type == 'pose_ref':                 
                colour = 'r'            
                unit_factor=0.5                
            elif self.object_type == 'pose_ends':  
                colour = 'r'            
                unit_factor=1                                
                

            H0_ = HomogeneousTransformation()
            H0_.H=self.H0
            origin = Vector(2)
                
            origin[0] = H0_.t[0]
            origin[1] = H0_.t[1]
                
            xhat = Vector(2)
            xhat[0] = unit_factor
            xhat = v2t(xhat)            
                
            yhat = Vector(2)
            yhat[1] = unit_factor
            yhat = v2t(yhat)
                
            xhat=H0_.H_R@xhat
            yhat=H0_.H_R@yhat

            xhat = t2v(xhat)
            yhat = t2v(yhat)                
                
            plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2*unit_factor, color='c')
            plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1*unit_factor, color='c')
            circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color='c', label=self.id0)
            plt.gca().add_patch(circle)
            
            H1_ = HomogeneousTransformation()
            H1_.H=self.H1
            origin = Vector(2)
                
            origin[0] = H1_.t[0]
            origin[1] = H1_.t[1]
                
            xhat = Vector(2)
            xhat[0] = unit_factor
            xhat = v2t(xhat)            
                
            yhat = Vector(2)
            yhat[1] = unit_factor
            yhat = v2t(yhat)
                
            xhat=H1_.H_R@xhat
            yhat=H1_.H_R@yhat

            xhat = t2v(xhat)
            yhat = t2v(yhat)                
                
            plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2*unit_factor, color=colour)
            plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1*unit_factor, color=colour)
            circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color=colour, label=self.id1)
            plt.gca().add_patch(circle)

            # visualise the edge (or line) connecting pose id0 and pose id1
            if self.edge_flag == True:                                
                if self.object_type == 'pose': plt.plot([H0_.t[1],H1_.t[1]],[H0_.t[0],H1_.t[0]], 'c--', linewidth = 1*unit_factor)
                if self.object_type == 'pose_gt': plt.plot([H0_.t[1],H1_.t[1]],[H0_.t[0],H1_.t[0]], 'g--', linewidth = 1*unit_factor)
                if self.object_type == 'pose_ref': plt.plot([H0_.t[1],H1_.t[1]],[H0_.t[0],H1_.t[0]], 'r--', linewidth = 1*unit_factor)                    

                    
    def _observation(self):
        # plot pose id0 as blue and pose id1 as different colours

            H0_ = HomogeneousTransformation()
            H0_.H=self.H0
            origin = Vector(2)
                
            origin[0] = H0_.t[0]
            origin[1] = H0_.t[1]
                
            xhat = Vector(2)
            xhat[0] = 1
            xhat = v2t(xhat)            
                
            yhat = Vector(2)
            yhat[1] = 1
            yhat = v2t(yhat)
                
            xhat=H0_.H_R@xhat
            yhat=H0_.H_R@yhat

            xhat = t2v(xhat)
            yhat = t2v(yhat)                
                
            plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2, color='b')
            plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1, color='b')
            circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color='b', label=self.id0)
            plt.gca().add_patch(circle)
            
            H1_ = HomogeneousTransformation()
            H1_.H=self.H1
            origin = Vector(2)
                
            origin[0] = H1_.t[0]
            origin[1] = H1_.t[1]
                
            xhat = Vector(2)
            xhat[0] = 0.2
            xhat = v2t(xhat)            
                
            yhat = Vector(2)
            yhat[1] = 0.2
            yhat = v2t(yhat)
                
            xhat=H1_.H_R@xhat
            yhat=H1_.H_R@yhat

            xhat = t2v(xhat)
            yhat = t2v(yhat)                
                
            plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.1, color='r')
            plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.05, color='r')
            circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color='r', label=self.id1)
            plt.gca().add_patch(circle)
            
            # calculate the cartesian vector of 
            c=self.H1@self.t
            c=t2v(c)                           

            circle = plt.Circle((c[1], c[0]), 0.05, color='r', label=self.id2)
            plt.gca().add_patch(circle)
                    

            if self.edge_flag == True:                                
#                 plt.arrow(*origin.reshape(2)[::-1], *xhat.reshape(2)[::-1], head_width=0.2, color='b')
#                 plt.arrow(*origin.reshape(2)[::-1], *yhat.reshape(2)[::-1], head_width=0.1, color='b')
#                 circle = plt.Circle((*origin.reshape(2)[::-1], *origin.reshape(2)[::-1]), 0.05, color='b', label=self.id0)
#                 plt.gca().add_patch(circle)                   
                plt.plot([H1_.t[1],c[1]],[H1_.t[0],c[0]], 'r--', linewidth = 1)
                    
    def _legend_2dframe(self):
        # Get handles and labels to remove duplicates in the legend

        handles, labels = plt.gca().get_legend_handles_labels()

        # Create a dictionary to track unique labels
        unique_labels = {}

        # Remove duplicates
        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_labels[label] = handle

        plt.legend(handles=unique_labels.values(), labels=unique_labels.keys(),bbox_to_anchor=(1.05, 1.0),loc="upper left")
        plt.axis('equal')
        plt.ylabel('Northings, m'); plt.xlabel('Eastings, m')


def show_information(matrix,n_pose,pose_size,n_landmark,landmark_size, display_type = 'both', matrix_compare = None):
    
    if display_type == 'both' or display_type == 'intensity':        
        show_information_intensity(matrix,n_pose,pose_size,n_landmark,landmark_size)
    if display_type == 'both' or display_type  == 'source':        
        show_information_source(matrix,n_pose,pose_size,n_landmark,landmark_size,matrix_compare)    


def show_information_intensity(matrix,n_pose,pose_size,n_landmark,landmark_size):   

    # note this only looks at absolute values of the matrix and ignores negatives
    fig, ax = plt.subplots()

    matrix = np.array(matrix)
#     cax = ax.matshow(abs(matrix), cmap='Greys')
    cax = ax.matshow(abs(matrix), norm=LogNorm(), cmap='Greys')

#     cbar = fig.colorbar(cax)
    
    # Add labels to the axes          
    labels = []
    if len(matrix[0])>1:
        for i in range(len(matrix[0])):
            if i<n_pose*pose_size:
                n = np.floor_divide(i,pose_size) 
                m = np.remainder(i,pose_size)
                if m==0: labels.append('x'+str(n+1))
                else: labels.append('')
            else:
                n = np.floor_divide((i-n_pose*pose_size),landmark_size)
                m = np.remainder((i-n_pose*pose_size),landmark_size)
                if m==0: labels.append('m'+str(n+1))
                else: labels.append('')                 
    else: labels.append('') #nothing if just 1d
    ax.set_xticks(np.arange(len(matrix[0])))        
    ax.set_xticklabels(labels)
        

    labels = []    #reset label
    for i in range(len(matrix)):
        if i<n_pose*pose_size:
            n = np.floor_divide(i,pose_size) 
            m = np.remainder(i,pose_size) 
            if m==0: labels.append('x'+str(n+1))
            else: labels.append('')
        else:
            n = np.floor_divide((i-n_pose*pose_size),landmark_size)
            m = np.remainder((i-n_pose*pose_size),landmark_size)
            if m==0: labels.append('m'+str(n+1))
            else: labels.append('') 
    ax.set_yticks(np.arange(len(matrix)))                
    ax.set_yticklabels(labels)
                
    # Show grid lines with equal dimensions
    ax.set_xticks(np.arange(len(matrix[0])) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix)) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

    # Make the axis tight
    plt.axis('tight')
    ax.set_aspect('equal')
    
    if 1.5*np.max(abs(matrix))>0.1:
        cax.set_clim(vmin=0.1,vmax=1.5*np.max(abs(matrix)))
    else:
        cax.set_clim(vmin=0.1,vmax=3)

    # Show the matrix
    plt.show()
    
def show_information_source(matrix,n_pose,pose_size,n_landmark,landmark_size,matrix_compare = None):      

    hue_node = 0.66 #blue
    hue_motion = 0.33 #green
    hue_landmark = 0.0 #red

    matrix = np.array(matrix)
    #colourise only needed for matrices and not vectors    
    hsv_matrix = np.zeros((matrix.shape[0],matrix.shape[1], 3))

    if matrix.shape[0]==matrix.shape[1]:#square matrix
        # now produced the coloured form
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i,j] != 0:#if the matrix is not empty
                    if i < n_pose*pose_size and j < n_pose*pose_size: # must be diagonal block or motion
                        if np.floor(i/pose_size) == np.floor(j/pose_size): # same diagonal block so a node
                            hsv_matrix[i,j, :] = [hue_node,0.4,0.9]
                        else: # must be motion
                            hsv_matrix[i,j, :] = [hue_motion,0.4,0.9]
                    else: # at least one part is in the landmark area
                        if np.floor((i-n_pose*pose_size)/landmark_size) == np.floor((j-n_pose*pose_size)/landmark_size): # same diagonal block so a node
                            hsv_matrix[i,j, :] = [hue_node,0.4,0.9]
                        else: #must be a landmark
                            hsv_matrix[i,j, :] = [hue_landmark,0.4,0.9]            
                else:
                    hsv_matrix[i,j, :] = [0,0,1]#white if there is no information
    else:
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i,j] != 0:#if the matrix is not empty
                    hsv_matrix[i,j, :] = [hue_node,0.4,0.9]
                else:
                    hsv_matrix[i,j, :] = [0,0,1]#white if there is no information

    
    
    if np.all(matrix_compare == None) == False:#there is a matrix        
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):                
                if matrix[i,j] != matrix_compare[i,j]:
                    # change to dark hue any place the matrix differs from the comparison
                    hsv_matrix[i,j, :] = [0,0,0.5]#dark                     

    # Set up the plot
    fig, ax = plt.subplots()

    # Convert HSV to RGB and plot the matrix
    rgb_matrix = hsv_to_rgb(hsv_matrix)
    cax = ax.matshow(rgb_matrix)

    # Add a colorbar
    # cbar = fig.colorbar(cax)

    # Add labels to the axes
    labels = []
    if len(matrix[0])>1:
        for i in range(len(matrix[0])):
            if i<n_pose*pose_size:
                n = np.floor_divide(i,pose_size) 
                m = np.remainder(i,pose_size)
                if m==0: labels.append('x'+str(n+1))
                else: labels.append('')
            else:
                n = np.floor_divide((i-n_pose*pose_size),landmark_size)
                m = np.remainder((i-n_pose*pose_size),landmark_size)
                if m==0: labels.append('m'+str(n+1))
                else: labels.append('')                 
    else: labels.append('') #nothing if just 1d
    ax.set_xticks(np.arange(len(matrix[0])))        
    ax.set_xticklabels(labels)
        

    labels = []    #reset label
    for i in range(len(matrix)):
        if i<n_pose*pose_size:
            n = np.floor_divide(i,pose_size) 
            m = np.remainder(i,pose_size) 
            if m==0: labels.append('x'+str(n+1))
            else: labels.append('')
        else:
            n = np.floor_divide((i-n_pose*pose_size),landmark_size)
            m = np.remainder((i-n_pose*pose_size),landmark_size)
            if m==0: labels.append('m'+str(n+1))
            else: labels.append('') 
    ax.set_yticks(np.arange(len(matrix)))                
    ax.set_yticklabels(labels)
    
    # Show grid lines with equal dimensions
    ax.set_xticks(np.arange(len(matrix[0])) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(matrix)) - 0.5, minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

    # Show the matrix
    plt.show()        

# def plot_motion_uncertainty(sigma_motion,u,dt=1.0,n=100):
   
#     if np.all(sigma_motion == 0.0) == False and np.all(u == 0.0) == False:
        
#         sigma_u=Matrix(2,2)
        
#         sigma_u[0,0]=(sigma_motion@(u*dt))[0]
#         sigma_u[1,1]=(sigma_motion@(u*dt))[1]

#         #visualise noise        
#         u_ = (u[:,0]*dt).tolist()
#         sigma_u_ = sigma_u.tolist()       
            
#         x1_range = [u_[0]-5*np.max(sigma_u)**0.5,u_[0]+5*np.max(sigma_u)**0.5,10*np.max(sigma_u)**0.5/100]
#         x2_range = [u_[1]-5*np.max(sigma_u)**0.5,u_[1]+5*np.max(sigma_u)**0.5,10*np.max(sigma_u)**0.5/100]
    
#         shade = 'Greens' #'Greys', 'Blues', 'Greens', 'Reds'
#         c = 'k'#'k', 'b', 'g', r'
                        
#         f_x = multivariate_normal(u_, sigma_u_)

#         plot_probability(f_x, u_, sigma_u_, x1_range, x2_range, shade, c); 

#         u_hat = np.random.multivariate_normal([0,0], sigma_u_, n)
#         u_hat[:,0]=u[0]*dt+u_hat[:,0]
#         u_hat[:,1]=u[1]*dt+u_hat[:,1]

#         plt.plot(u_hat[:,1],u_hat[:,0],'g.',markersize = 2)

#         H_eb = HomogeneousTransformation(0,0)     

#         cf=plot_2dframe(H_eb.H,v2t(u_hat),['map','b','m0'],True)        
#         cf.fixed_frame()
#         plt.axis('equal')
#         plt.show()
        
#         gamma_hat=np.random.normal(0, (sigma_motion@(u*dt))[2][0], n)
#         gamma_hat = u[1]*dt + gamma_hat    
    
#         m=(u[1]*dt)[0]
#         s=(sigma_motion@(u*dt))[2][0]
#         c='g'
    
#         x_range = np.arange(-6*s+m, 6*s+m,12*s/100)
    
#         plt.plot(np.rad2deg(x_range),gaussian(m, s, x_range),c,linewidth=3, 
#                      label ='$\mu = $' + str(m) + ', $\sigma = $' + str(s))
#         # show sigma region (region of 68.3% confidence) and mean
#         roi = np.arange(-s+m, s+m, 2*s/100)
#         plt.fill_between(np.rad2deg(roi),gaussian(m, s, roi),color=c,alpha=0.2)
#         plt.plot([np.rad2deg(m),np.rad2deg(m)],[0, gaussian(m, s, m)],color=c,linestyle='--')
#         plt.xlabel('$\Delta \gamma$, degrees'); plt.ylabel('p($\gamma$)');
#         plt.plot(np.rad2deg(gamma_hat),np.zeros(len(gamma_hat))-0.1*np.max(gaussian(m, s, x_range)),'g.',markersize = 1)
#         plt.plot(0,0,'ko',markersize = 5)
#         plt.show()
def plot_motion_uncertainty(sigma_motion,u,dt=1.0,n=100):
   
    if np.all(sigma_motion == 0.0) == False and np.all(u == 0.0) == False:
        
        sigma=(sigma_motion@u)        
        sigma_u=Matrix(3,3)

        sigma_u[0,0] = sigma[0]*dt
        sigma_u[1,1] = sigma[1]*dt
        sigma_u[2,2] = sigma[2]*dt
        
        #calculate the noise component
        J = Matrix(3,3)

        dx_dx = 1
        dx_dy = 0
        if u[1]!=0: dx_dg = (u[0]/u[1])*(-np.cos(0)+np.cos(0+u[1]*dt))
        else: dx_dg = 0
        dy_dx = 0
        dy_dy = 1
        if u[1]!=0: dy_dg = (u[0]/u[1])*(-np.sin(0)+np.sin(0+u[1]*dt))
        else: dy_dg = 0
        dg_dx = 0
        dg_dy = 0       
        dg_dg = 1
                
        J[0,0] = dx_dx
        J[0,1] = dx_dy
        J[0,2] = dx_dg        
        J[1,0] = dy_dx
        J[1,1] = dy_dy
        J[1,2] = dy_dg        
        J[2,0] = dg_dx
        J[2,1] = dg_dy
        J[2,2] = dg_dg        

        sigma_xy = J@sigma_u@J.T
        
        #motion 
        H_eb = HomogeneousTransformation(0,0)
        H_eb_ = HomogeneousTransformation()

        v = u[0] #surge rate
        w = u[1] #yaw rate

        # calculate centre of rotation from the initial body position
        t_bc = Vector(2) # [2x1] matrix of 0                       
        t_bc[1]=v/w      # centre of rotation is v/w in the +ve y direction of the body Eq(A1.2.12)
            
        # the centre of rotation 'c' keeps the heading of the body, so is 0 as seen from the body
        
        H_bc = HomogeneousTransformation(t_bc,0)
                
        # to rotate the body about 'c' so need 'b' as seen from the centre of rotation
        H_cb = HomogeneousTransformation() 
        H_cb.H = Inverse(H_bc.H)  

        # to rotate the body b around the centre of rotation w*dt while maintain the same radius
        H_cb_ = HomogeneousTransformation(H_cb.t,w*dt)
            
        # we rotate first, and then translate
        H_eb_.H = H_eb.H@H_bc.H@H_cb_.H_R@H_cb_.H_T 
        

        #visualise noise
        xy_ = H_eb_.t[:,0].tolist()
        sigma_xy_ = sigma_xy[0:2,0:2].tolist()       

        x1_range = [xy_[0]-5*np.max(sigma_xy[0:2,0:2])**0.5,xy_[0]+5*np.max(sigma_xy[0:2,0:2])**0.5,10*np.max(sigma_xy[0:2,0:2])**0.5/100]
        x2_range = [xy_[1]-5*np.max(sigma_xy[0:2,0:2])**0.5,xy_[1]+5*np.max(sigma_xy[0:2,0:2])**0.5,10*np.max(sigma_xy[0:2,0:2])**0.5/100]
    
        shade = 'Greens' #'Greys', 'Blues', 'Greens', 'Reds'
        c = 'k'#'k', 'b', 'g', r'

                                
        f_x = multivariate_normal(xy_, sigma_xy_)

        plot_probability(f_x, xy_, sigma_xy_, x1_range, x2_range, shade, c); 

        xy_hat = np.random.multivariate_normal([0,0], sigma_xy_, n)
        xy_hat[:,0]=xy_[0]+xy_hat[:,0]
        xy_hat[:,1]=xy_[1]+xy_hat[:,1]

        plt.plot(xy_hat[:,1],xy_hat[:,0],'k.',markersize = 2)

        cf=plot_2dframe(['pose','b','b_'],[H_eb.H,H_eb_.H],True)        
        cf._fixed_frame()
        cf._pose()

        plt.axis('equal')
        plt.show()
        
        gamma_hat=np.random.normal(0, sigma_u[2,2], n)
        gamma_hat = u[1]*dt + gamma_hat    
    
        m=(u[1]*dt)[0]
        s=sigma_u[2,2]
        c='g'
    
        x_range = np.arange(-6*s+m, 6*s+m,12*s/100)
    
        plt.plot(np.rad2deg(x_range),gaussian(m, s, x_range),c,linewidth=3, 
                     label ='$\mu = $' + str(m) + ', $\sigma = $' + str(s))
        # show sigma region (region of 68.3% confidence) and mean
        roi = np.arange(-s+m, s+m, 2*s/100)
        plt.fill_between(np.rad2deg(roi),gaussian(m, s, roi),color=c,alpha=0.2)
        plt.plot([np.rad2deg(m),np.rad2deg(m)],[0, gaussian(m, s, m)],color=c,linestyle='--')
        plt.xlabel('$\Delta \gamma$, degrees'); plt.ylabel('p($\gamma$)');
        plt.plot(np.rad2deg(gamma_hat),np.zeros(len(gamma_hat))-0.1*np.max(gaussian(m, s, x_range)),'k.',markersize = 1)
        plt.plot(0,0,'ko',markersize = 5)
        plt.show()

def plot_probability(f_x, mu, sigma, ax_range=None, ay_range=None,shade='Greys',c='k'):
    # prepare 2d state variables for plot
    
    if ax_range == None:
        ax_range=[mu[0]-20*np.max((sigma)),mu[0]+20*np.max((sigma)),40*np.max((sigma))/100]
    if ay_range == None:
        ay_range=[mu[1]-20*np.max((sigma)),mu[1]+20*np.max((sigma)),40*np.max((sigma))/100]                
    
    x_1, x_2 = np.mgrid[ax_range[0]:ax_range[1]:ax_range[2],ay_range[0]:ay_range[1]:ay_range[2]]
    pos = np.empty(x_1.shape + (2,))
    pos[:, :, 0] = x_1; pos[:, :, 1] = x_2

    # plot colour map 
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    plt.contourf(x_2, x_1, f_x.pdf(pos), 100, cmap=shade) # Swap x_1 and x_2 here
    text_mu0=str("{:.3f}".format(round(mu[0], 3)))
    text_mu1=str("{:.3f}".format(round(mu[1], 3)))    
    text_sigma00 = str("{:.3f}".format(round(sigma[0][0],3)))
    text_sigma01 = str("{:.3f}".format(round(sigma[0][1],3)))
    text_sigma10 = str("{:.3f}".format(round(sigma[1][0],3)))
    text_sigma11 = str("{:.3f}".format(round(sigma[1][1],3)))                      
    title = 'mean = ['  + text_mu0 + ' , ' + text_mu1 + '] \n cov = [' + text_sigma00 + ' , ' + text_sigma01 + ' ; ' + text_sigma10 + ' , ' + text_sigma11 + ']'    
    plt.ylabel('$\Delta x_b$, m'); plt.xlabel('$\Delta y_b$, m'); plt.title(title) # Swap the labels here
    colour = plt.colorbar(); colour.set_label('$p(\Delta x_b, \Delta y_b)$')
    e=sigma_contour([mu[1], mu[0]],[[sigma[1][1],sigma[1][0]],[sigma[0][1],sigma[0][0]]],c)
    e.set_facecolor('none')
    ax.add_patch(e)
    ax.legend()
    ax.set_aspect('auto')
#     plt.show()


# show sigma region (region of 68.3% confidence)
def sigma_contour(mu,sigma,c):
    # calculate 1sigma uncertainty bound
    e_val, e_vec = np.linalg.eigh(sigma)      
    # sort eigen values and eigen vectors in order of size
    idx=np.argsort(e_val)[::-1]
    e_val=e_val[idx]; e_vec=e_vec[:,idx]
    # standard deviation is square root of covariance 
    e_val=np.sqrt(e_val)
    w=e_val[0]*2; h=e_val[1]*2; theta=np.degrees(np.arctan2(*e_vec[:,0][::-1]))
    return Ellipse(xy=mu, width = w, height  = h, angle=theta, label = r'$\sigma$',color=c, linestyle='-')  
    



def plot_path(P, legend_flag = True, trackline_flag = True, verbose = False):
# Function cycles through the entire path, create a homogeneous matrix for each entry and plot all the poses using a for loop    
    for i in range(len(P)-1):
        
        if verbose == True:
            print('Path entry P',i,' has coordinates ',P[[i],0:2].T)
            print('and heading ',P[i,2])

            print('Path entry P',i+1,' has coordinates ',P[[i+1],0:2].T)
            print('and heading ',P[i+1,2])                    
            
        H_eb=HomogeneousTransformation(P[[i],0:2].T,P[i,2])
        H_eb_=HomogeneousTransformation(P[[i+1],0:2].T,P[i+1,2])

        plot_2dframe(['pose','p'+str(i),'p'+str(i+1)],[H_eb.H,H_eb_.H],trackline_flag,legend_flag)
    
    H_eb = HomogeneousTransformation(P[[0],0:2].T,P[0,2])
    H_eb_ = HomogeneousTransformation(P[[-1],0:2].T,P[-1,2])
    plot_2dframe(['pose_ends','Start','End'],[H_eb.H,H_eb_.H],False,legend_flag)

def plot_zero_order(t,u, c = None,label = None):
    # plotting function for control sequence with zero order steps between commands
    # first list is assumed to be t of form t=[]
    # second list is assumed to be a single control parameter with the same length as t
    
    t_step = []
    u_step = []    
    for i in range(len(t)):
        
        t_step.append(t[i]) 
        if i == 0: u_step.append(u[0])               
        else: u_step.append(u[i-1])                          
            
        t_step.append(t[i])                        
        u_step.append(u[i])                          

    if c == None: c = 'b'
    plt.plot(t_step,u_step,c,label = label)
    if label != None: plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # overlay the actual commands
    plt.plot(t,u, 'o', color = c)        
    plt.xlabel('Time, s')

def plot_trajectory(s, arc_radius = None, t_ref = None, p_ref = None, u_ref = None, p_robot = None, accept_radius = None, show_time_history = True):

    if show_time_history == True:
        plt.plot(s.Tp,s.V,'o-b',label = 'Straight lines')
        if np.all(np.isnan(s.Tp_arc) == False): plt.plot(s.Tp_arc,s.V_arc,'o-r',label = 'Turning arcs')   
        if isinstance(u_ref, np.ndarray): plt.plot(t_ref,u_ref[0],'og',label = 'Reference')        
        plt.xlabel('Time, s')
        plt.ylabel('Velocity, m/s')
        plt.legend()
        plt.show()

        plt.plot(s.Tp,s.W,'o-b',label = 'Straight lines')
        if np.all(np.isnan(s.Tp_arc) == False): 
            plot_zero_order(s.Tp_arc.T[0],np.rad2deg(s.W_arc[:,0]),'r',label = 'Turning arcs')                
        if isinstance(u_ref, np.ndarray): plt.plot(t_ref,np.rad2deg(u_ref[1]),'og',label = 'Reference')        
        plt.xlabel('Time, s')
        plt.ylabel('Angular velocity, deg/s')
        plt.legend()
        plt.show()


        plt.plot(s.Tp,s.P[:,0],'o-b',label = 'Straight lines')
        if np.all(np.isnan(s.Tp_arc) == False): plt.plot(s.Tp_arc,s.P_arc[:,0],'or',label = 'Turning arcs')    
        if isinstance(p_ref, np.ndarray): plt.plot(t_ref,p_ref[0],'og',label = 'Reference')                    
        plt.xlabel('Time, s')
        plt.ylabel('Northings, m')
        plt.legend()
        plt.show()

        plt.plot(s.Tp,s.P[:,1],'o-b',label = 'Straight lines')
        if np.all(np.isnan(s.Tp_arc) == False): plt.plot(s.Tp_arc,s.P_arc[:,1],'or',label = 'Turning arcs')       
        if isinstance(p_ref, np.ndarray): plt.plot(t_ref,p_ref[1],'og',label = 'Reference')                                
        plt.xlabel('Time, s')
        plt.ylabel('Eastings, m')
        plt.legend()
        plt.show()

        plot_zero_order(s.Tp.T[0],np.rad2deg(s.P[:,2]),'b',label = 'Straight lines')
        if np.all(np.isnan(s.Tp_arc) == False): plt.plot(s.Tp_arc.T[0],np.rad2deg(s.P_arc[:,2]),'o-r',label = 'Turning arcs')       
        if isinstance(p_ref, np.ndarray): plt.plot(t_ref,np.rad2deg(p_ref[2]),'og',label = 'Reference')                                            
        plt.xlabel('Time, s')
        plt.ylabel('Heading, deg')
        plt.legend()
        plt.show()

        plt.plot(s.Tp,'o-b',label = 'Straight lines')
        if np.all(np.isnan(s.Tp_arc) == False): plt.plot(s.Tp_arc,'o-r',label = 'Turning arcs')
        if isinstance(p_ref, np.ndarray): plt.plot([0,len(s.Tp_arc)],[t_ref,t_ref],'--g',label = 'Reference')                                             
        plt.xlabel('Waypoint number')
        plt.ylabel('Expected arrival, s')
        plt.legend()
        plt.show()
    

    # Create a figure and axis        
    fig, ax = plt.subplots()

    ax.plot(s.P[:,1],s.P[:,0],'o-k')
    ax.plot(s.P_arc[:,1],s.P_arc[:,0],'ok')
    ax.plot(s.Arc[:,1],s.Arc[:,0],'+k')
    ax.plot(s.P[0,1],s.P[0,0],'oc',label = 'Start')
    ax.plot(s.P[-1,1],s.P[-1,0],'or',label = 'End')
    ax.set_xlabel('Eastings, m')
    ax.set_ylabel('Northings, m')
    # Add the circle patch to the axis
    if arc_radius != None: 
        for i in range(len(s.Arc)):
            circle = patches.Circle((s.Arc[i,1], s.Arc[i,0]), radius=arc_radius, edgecolor='black', facecolor='none')
            ax.add_patch(circle)
    
    if isinstance(p_ref, np.ndarray): ax.plot(p_ref[1],p_ref[0],'om',label = 'p_ref')
    if isinstance(p_robot, np.ndarray): ax.plot(p_robot[1],p_robot[0],'+r',label = 'p_robot')
            
    if s.wp_id != None and accept_radius != None:
        circle = patches.Circle((s.P_arc[s.wp_id,1], s.P_arc[s.wp_id,0]), radius=accept_radius, edgecolor='blue', facecolor='none', label='wp accept')
        ax.add_patch(circle)
                                
    plt.axis('equal')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	
def plot_1d_gauss(m,s,x=np.arange(-5,5,0.05),c='b'):
# Plot the curves and shade between +/- sigma     
    plt.plot(x,gaussian(m, s, x),c,linewidth=3, 
             label ='$\mu = $' + str(m) + ', $\sigma = $' + str(s))
    # show sigma region (region of 68.3% confidence) and mean
    roi = np.arange(-s+m, s+m, 0.05)
    plt.fill_between(roi,gaussian(m, s, roi),color=c,alpha=0.2)
    
    plt.plot([m,m],[0, gaussian(m, s, m)], color = c,linestyle= '--')              
    plt.xlabel('x'); plt.ylabel('f(x)'); # plt.legend()  


def plot_2d_gauss(mu, cov, x1_range=None, x2_range=None,shade='Greys',c='k'):
    # prepare 2d state variables for plot
    
    if x1_range == None:
        x1_range=[mu[0]-5*np.max((cov)),mu[0]+5*np.max((cov)),10*np.max((cov))/100]
    if x2_range == None:
        x2_range=[mu[1]-5*np.max((cov)),mu[1]+5*np.max((cov)),10*np.max((cov))/100]        
        
    
    x1, x2 = np.mgrid[x1_range[0]:x1_range[1]:x1_range[2],x2_range[0]:x2_range[1]:x2_range[2]]
    pos = np.empty(x1.shape + (2,))
    pos[:, :, 0] = x1; pos[:, :, 1] = x2

    fx = multivariate_normal(mu, cov)

    # plot colour map 
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    plt.contourf(x1,x2, fx.pdf(pos), 100, cmap=shade)
    text_mu0=str("{:.3f}".format(round(mu[0], 3)))
    text_mu1=str("{:.3f}".format(round(mu[1], 3)))    
    text_cov00 = str("{:.3f}".format(round(cov[0][0],3)))
    text_cov01 = str("{:.3f}".format(round(cov[0][1],3)))
    text_cov10 = str("{:.3f}".format(round(cov[1][0],3)))
    text_cov11 = str("{:.3f}".format(round(cov[1][1],3)))                      
#     title = 'mean = ['  + str(mu[0]) + ' , ' + str(mu[1]) + '] \n cov = [' + str(cov[0][0]) + ' , ' + str(cov[0][1]) + ' ; ' + str(cov[1][0]) + ' , ' + str(cov[1][1]) + ']'
    title = 'mean = ['  + text_mu0 + ' , ' + text_mu0 + '] \n cov = [' + text_cov00 + ' , ' + text_cov01 + ' ; ' + text_cov10 + ' , ' + text_cov11 + ']'    
    plt.xlabel('$x_1$'); plt.ylabel('$x_2$'); plt.title(title)
    colour = plt.colorbar(); colour.set_label('$f(x_1,x_2)$')
    e=cov_contour(mu,cov,c)
    e.set_facecolor('none')
    ax.add_patch(e)
    ax.legend()
    ax.set_aspect('auto')
#     plt.show()

# show sigma region (region of 68.3% confidence)
def cov_contour(mu,cov,c):
    # calculate 1sigma uncertainty bound
    e_val, e_vec = np.linalg.eigh(cov)      
    # sort eigen values and eigen vectors in order of size
    idx=np.argsort(e_val)[::-1]
    e_val=e_val[idx]; e_vec=e_vec[:,idx]
    # standard deviation is square root of covariance 
    e_val=np.sqrt(e_val)
    w=e_val[0]*2; h=e_val[1]*2; theta=np.degrees(np.arctan2(*e_vec[:,0][::-1]))
    return Ellipse(xy=mu, width = w, height  = h, angle=theta, label = r'$\sigma$',color=c, linestyle='-')  

    plt.fill_between(roi,gaussian(m, s, roi),color=c,alpha=0.2)
    
    plt.plot([m,m],[0, gaussian(m, s, m)], color = c,linestyle= '--')              
    plt.xlabel('x'); plt.ylabel('f(x)'); # plt.legend()      


# illustrate input output relationships through a function
def plot_gaussians_through_functions(x,fx,pd_in,function_label,ax_range=[-2, 2],ay_range=[-2, 2],fx_lin=None,point_projections=False,key_points=None):
    
    
    # fig, (main_ax, dist_out, dist_in) = plt.subplots(2, 1, )  # Adjust the figsize as needed
    fig = plt.figure(figsize=(8, 8), dpi=150)
    plt.rcParams['font.size'] = 24


    grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)

    
    main_ax = fig.add_subplot(grid[:-1, 1:])
    dist_out = fig.add_subplot(grid[:-1, 0])
    dist_in = fig.add_subplot(grid[-1, 1:])

    # plot function
    main_ax.plot(x, fx, 'k', linewidth=4, label=function_label)
    if fx_lin is not None:#,function_label_lin=None
        main_ax.plot(x, fx_lin, 'r', linewidth=3, label='linearized')
    
    main_ax.legend()
    main_ax.grid(True)
    main_ax.set_xticklabels([]) 
    main_ax.set_yticklabels([]) 
    main_ax.set_xlim(ax_range)
    main_ax.set_ylim(ay_range)    

    # input distribution
    idx=np.argmax(pd_in)
    dist_in.plot(x, pd_in,'k',linewidth=4)
    dist_in.plot([x[idx],x[idx]],[0,pd_in[idx]],'k-',linewidth=3)
    
    # output distribution, 100 bins is ok but curve needs sufficient points to look ok
    n=100 #bin size
    fx_bins=np.arange(min(fx),max(fx),1/n)

    pd_hist=[]
    for i in range(len(fx_bins)-1):
        j=0; p=0
        while j < len(fx):
            if fx[j] > fx_bins[i] and fx[j] <= fx_bins[i+1]:
                p+=pd_in[j]
            j+=1                
        pd_hist.append(p*n)
    pd_hist.append(0) # adds a zero on the end of the histogram
    
    idx=pd_hist.index(max(pd_hist))
    dist_out.plot(pd_hist,fx_bins,'k',linewidth=4, label = '$f(x)$')    
    dist_out.plot([0,pd_hist[idx]],[fx_bins[idx],fx_bins[idx]],'k-',linewidth=3)            
    
    if fx_lin is not None:#,function_label_lin=None 
        n=100
        # output distribution, 100 bins is ok but curve needs sufficient points to look ok
        pd_lin=[]
        fx_lin_bins=np.arange(min(fx_lin),max(fx_lin),1/n)                
        for i in range(len(fx_lin_bins)-1):        
            j=0; p=0
            while j < len(fx_lin):
                if fx_lin[j] > fx_lin_bins[i] and fx_lin[j] <= fx_lin_bins[i+1]:
                    p+=pd_in[j]
                j+=1                    
            pd_lin.append(p*n)
        pd_lin.append(0) # adds a zero on the end of the histogram

        idx=pd_lin.index(max(pd_lin))
        
        dist_out.plot(pd_lin,fx_lin_bins,'r-',linewidth=4)            
        dist_out.plot([0,pd_lin[idx]],[fx_lin_bins[idx],fx_lin_bins[idx]],'r--',linewidth=3)                    
    
    if point_projections is True:        
        for i in range(len(fx_bins)-1):
            if fx_bins[i] < key_points[3] and fx_bins[i+1] > key_points[3]:
                ukf_mu_out=[fx_bins[i],pd_hist[i]]
            if fx_bins[i] < key_points[4] and fx_bins[i+1] > key_points[4]:                
                ukf_n_sigma_out=[fx_bins[i],pd_hist[i]]
            if fx_bins[i] < key_points[5] and fx_bins[i+1] > key_points[5]:
                ukf_p_sigma_out=[fx_bins[i],pd_hist[i]]

        for i in range(len(x)-1):
            if x[i] < key_points[0] and x[i+1] > key_points[0]:
                ukf_mu_in=[x[i],pd_in[i]]
            if x[i] < key_points[1] and x[i+1] > key_points[1]:
                ukf_n_sigma_in=[x[i],pd_in[i]]
            if x[i] < key_points[2] and x[i+1] > key_points[2]:
                ukf_p_sigma_in=[x[i],pd_in[i]]
                
        dist_in.plot([ukf_n_sigma_in[0],ukf_n_sigma_in[0]],[0,ukf_n_sigma_in[1]],'k--',linewidth=3)
        dist_in.plot([ukf_p_sigma_in[0],ukf_p_sigma_in[0]],[0,ukf_p_sigma_in[1]],'k--',linewidth=3)                    
        dist_in.plot([ukf_n_sigma_in[0]],[ukf_n_sigma_in[1]],'ko') 
        dist_in.plot([ukf_p_sigma_in[0]],[ukf_p_sigma_in[1]],'ko') 
        dist_in.plot([ukf_mu_in[0]],[ukf_mu_in[1]],'ko') 
        dist_in.set_ylim([0, 1.2*ukf_mu_in[1]])

        dist_out.plot([ukf_n_sigma_out[1]],[ukf_n_sigma_out[0]],'ro') 
        dist_out.plot([ukf_p_sigma_out[1]],[ukf_p_sigma_out[0]],'ro')
        dist_out.plot([ukf_mu_out[1]],[ukf_mu_out[0]],'ro')

        def gaus_fit(x, a, b, c): return a*np.exp(-np.power(x - b, 2)/(2*np.power(c, 2)))
        pars, cov = curve_fit(f=gaus_fit, xdata=[ukf_n_sigma_out[0],ukf_mu_out[0],ukf_p_sigma_out[0]], ydata=[ukf_n_sigma_out[1],ukf_mu_out[1],ukf_p_sigma_out[1]], p0=[0, 0.5, 2], bounds=(-np.inf, np.inf))
        # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
        stdevs = np.sqrt(np.diag(cov))
        # Calculate the residuals
        res = [ukf_n_sigma_out[1],ukf_mu_out[1],ukf_p_sigma_out[1]] - gaus_fit([ukf_n_sigma_out[0],ukf_mu_out[0],ukf_p_sigma_out[0]], *pars)
        
        # Show plot
        idx=gaus_fit(fx_bins, *pars).argmax()                        
        dist_out.plot(gaus_fit(fx_bins, *pars),fx_bins,'r-',linewidth=4)
        dist_out.plot([0,gaus_fit(fx_bins, *pars)[idx]],[fx_bins[idx],fx_bins[idx]],'r--',linewidth=3)                    

        
    dist_out.set_ylim(ay_range)
    dist_out.set_xticklabels([]) # Force this empty !
    dist_out.set_ylabel('$f(x)$') 
    
    dist_in.set_yticklabels([]) # Force this empty !
    dist_in.set_xlabel('$x$') 
    dist_in.set_xlim(ax_range)



def plot_kalman(mu1, Sigma1, mu2, Sigma2, x, xlim, ylim, z=None, C=None, Q=None): 

    plt.rcParams["figure.figsize"] = (3,2) #make plots look nice
    plt.rcParams["figure.dpi"] = 300 #make plots look nice
    plt.rcParams['font.size'] = 12 #return to normal
    # note Sigma is the covariance and the gaussian function implemented here is for the standard deviation (i.e., sqrt(cov))
    if z is None:
                

        plt.plot(x,gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), x),'k',linewidth=3, alpha=1, label ='Previous state')
        roi = np.arange(-np.sqrt(Sigma1[0,0])+mu1[0,0], np.sqrt(Sigma1[0,0])+mu1[0,0], 0.05)
        plt.fill_between(roi,gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), roi),color='k',alpha=0.1)   
        
        plt.plot([mu1[0,0],mu1[0,0]],[0, gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), mu1[0,0])],color='k',linewidth=2,linestyle='--', alpha=1,)
        plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylim([ylim[0],ylim[1]])        
        plt.xlim([xlim[0],xlim[1]])                
        plt.show()    

        plt.plot(x,gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), x),'k',linewidth=2, alpha=0.2, label ='Previous state')
        roi = np.arange(-np.sqrt(Sigma1[0,0])+mu1[0,0], np.sqrt(Sigma1[0,0])+mu1[0,0], 0.05)
        plt.fill_between(roi,gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), roi),color='k',alpha=0.1)
        plt.plot([mu1[0,0],mu1[0,0]],[0, gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), mu1[0,0])],color='k',linewidth=2,linestyle='--', alpha=0.3,)
        
        plt.plot(x,gaussian(mu2[0,0], np.sqrt(Sigma2[0,0]), x),'b',linewidth=3, label ='Prediction')        
        roi = np.arange(-np.sqrt(Sigma2[0,0])+mu2[0,0], np.sqrt(Sigma2[0,0])+mu2[0,0], 0.05)
        plt.fill_between(roi,gaussian(mu2[0,0], np.sqrt(Sigma2[0,0]), roi),color='b',alpha=0.1)
        plt.plot([mu2[0,0],mu2[0,0]],[0, gaussian(mu2[0,0], np.sqrt(Sigma2[0,0]), mu2[0,0])],color='b',linewidth=2,linestyle='--', alpha=1,)
        plt.xlim([xlim[0],xlim[1]])                        
        
    else:
        plt.plot(x,gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), x),'k',linewidth=2, alpha=0.2, label ='Prediction')
        roi = np.arange(-np.sqrt(Sigma1[0,0])+mu1[0,0], np.sqrt(Sigma1[0,0])+mu1[0,0], 0.05)
        plt.fill_between(roi,gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), roi),color='k',alpha=0.1)
        plt.plot([mu1[0,0],mu1[0,0]],[0, gaussian(mu1[0,0], np.sqrt(Sigma1[0,0]), mu1[0,0])],color='k',linewidth=2,linestyle='--', alpha=0.3,)

        plt.plot(x,gaussian(1/C*z[0,0], np.sqrt(1/C*Q[0,0]*1/C), x),'g',linewidth=3, label ='Measurement')
        s=np.sqrt(1/C*Q[0,0]*1/C)
        m=1/C*z[0,0]
        roi = np.arange(-s+m,s+m, 0.05)
        plt.fill_between(roi,gaussian(1/C*z[0,0], np.sqrt(1/C*Q[0,0]*1/C), roi),color='g',alpha=0.1)
        plt.plot([1/C*z[0,0],1/C*z[0,0]],[0, gaussian(1/C*z[0,0], np.sqrt(1/C*Q[0,0]*1/C), 1/C*z[0,0])],color='g',linewidth=2,linestyle='--', alpha=1,)        
        
        plt.plot(x,gaussian(mu2[0,0], np.sqrt(Sigma2[0,0]), x),'r',linewidth=3, label ='Update')
        roi = np.arange(-np.sqrt(Sigma2[0,0])+mu2[0,0], np.sqrt(Sigma2[0,0])+mu2[0,0], 0.05)
        plt.fill_between(roi,gaussian(mu2[0,0], np.sqrt(Sigma2[0,0]), roi),color='r',alpha=0.1)
        plt.plot([mu2[0,0],mu2[0,0]],[0, gaussian(mu2[0,0], np.sqrt(Sigma2[0,0]), mu2[0,0])],color='r',linewidth=2,linestyle='--', alpha=1,)
        plt.xlim([xlim[0],xlim[1]])                        
        

    plt.xlabel('x'); plt.ylabel('f(x)'); plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylim([ylim[0],ylim[1]])
    plt.xlim([xlim[0],xlim[1]])                    

    plt.show()


def plot_EKF_trajectory(states, covariances, flip=False, measurements=None, keyframe = 50):
    _ = plt.figure(figsize=(5, 5))
    ax = plt.subplot(111, aspect="equal")
    i = 0
    for s in states:
        if i % keyframe == 0:
            j = len(s)                        
            vals, vecs = eigsorted(covariances[i*j:i*j+2, 0:2])
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            if flip:
                e = Ellipse(
                    xy=(s[1], s[0]), width=vals[1], height=vals[0], angle=-theta
                )
            else:
                e = Ellipse(
                    xy=(s[0], s[1]), width=vals[0], height=vals[1], angle=theta
                )
            e.set_alpha(0.2)
            e.set_facecolor("red")
            ax.add_patch(e)
        i += 1
    xpos = [s[0] for s in states]
    ypos = [s[1] for s in states]
    if flip:
        plt.plot(ypos, xpos, "bo-")
    else:
        plt.plot(xpos, ypos, "bo-")
    plt.xlabel("Eastings (m)")
    plt.ylabel("Northings (m)")
    plt.axis("equal")

    if measurements is not None:
        for i in range(len(measurements[0])):
            if not flip:
                circle = plt.Circle(                    
                    (measurements[0][i,0], measurements[0][i,1]), measurements[1][i,0], color="g"
                )
                circle.set_alpha(0.2)
                ax = plt.gca()
                ax.add_patch(circle)
                plt.plot(measurements[0][i,0], measurements[0][i,1], "gx")
            else:
                circle = plt.Circle(
                    (measurements[0][i,1], measurements[0][i,0]), measurements[1][i,0], color="g"
                )
                circle.set_alpha(0.2)
                ax = plt.gca()
                ax.add_patch(circle)
                plt.plot(measurements[0][i,1], measurements[0][i,0], "gx")
    plt.show()    
    

def plot_cumsum(a):
    fig = plt.figure()
    N = len(a)
    cmap = mpl.colors.ListedColormap(
        [[0.0, 0.4, 1.0], [0.0, 0.8, 1.0], [1.0, 0.8, 0.0], [1.0, 0.4, 0.0]]
        * (int(N / 4) + 1)
    )
    cumsum = np.cumsum(np.asarray(a) / np.sum(a))
    cumsum = np.insert(cumsum, 0, 0)

    # fig = plt.figure(figsize=(6,3))
    fig = plt.gcf()
    ax = fig.add_axes([0.05, 0.475, 0.9, 0.15])
    norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)
    bar = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        drawedges=False,
        spacing="proportional",
        orientation="horizontal",
    )
    if N > 10:
        bar.set_ticks([])


def plot_systematic_resample(a, random_number=None):
    N = len(a)

    cmap = mpl.colors.ListedColormap(
        [[0.0, 0.4, 1.0], [0.0, 0.8, 1.0], [1.0, 0.8, 0.0], [1.0, 0.4, 0.0]]
        * (int(N / 4) + 1)
    )
    cumsum = np.cumsum(np.asarray(a) / np.sum(a))
    cumsum = np.insert(cumsum, 0, 0)

    fig = plt.figure()
    ax = plt.gcf().add_axes([0.05, 0.475, 0.9, 0.15])
    norm = mpl.colors.BoundaryNorm(cumsum, cmap.N)
    bar = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        drawedges=False,
        spacing="proportional",
        orientation="horizontal",
    )
    xs = np.linspace(0.0, 1.0 - 1.0 / N, N)
    ax.vlines(xs, 0, 1, lw=2)

    # make N subdivisions, and chose a random position within each one
    if random_number is None:
        random_number = random()
    b = (random_number + np.array(range(N))) / N
    plt.scatter(b, [0.5] * len(b), s=60, facecolor="k", edgecolor="k")
    bar.set_ticks([])
    plt.title("systematic resampling")



def plot_pf_trajectory(estimates, past_northings, past_eastings, past_weights, position_list=None, past_neff=None):
    stamp = [x[0] for x in estimates]
    est_northings = [x[1] for x in estimates]
    est_eastings = [x[2] for x in estimates]
    northings_std = [x[3] for x in estimates]
    eastings_std = [x[4] for x in estimates]
    max_weight = np.max(past_weights, axis=1)

    plt.figure(figsize=[8, 8])

    mid = int(len(past_northings) / 2.0)
    plt.plot(past_eastings[0], past_northings[0], 'r.', alpha=1, markersize=6, label='Particles')
    plt.plot(past_eastings[round(mid/2)], past_northings[round(mid/2)], 'r.', alpha=1, markersize=6)
    plt.plot(past_eastings[mid], past_northings[mid], 'r.', alpha=1, markersize=6)
    plt.plot(past_eastings[round(mid/2)+mid], past_northings[round(mid/2)+mid], 'r.', alpha=1, markersize=6)    
    plt.plot(past_eastings[-1], past_northings[-1], 'r.', alpha=1, markersize=6)
    plt.plot(est_eastings, est_northings, "bo", markersize=8, markeredgewidth=0, label="KDE")
    ax = plt.gca()
    if position_list is not None:
        for p in position_list:
            e = Ellipse(xy=(p['eastings'], p['northings']), width=2*p['eastings_std'], height=2*p['northings_std'], angle=0)
            e.set_alpha(0.3)
            e.set_facecolor("green")
            ax.add_patch(e)
            plt.plot(p['eastings'], p['northings'], "gx")            
            
    plt.xlabel("Eastings (m)",fontsize=14); plt.ylabel("Northings (m)",fontsize=14)
    plt.legend(); plt.axis('equal'); plt.show()

    plt.figure(figsize=[8, 4])
    plt.bar(past_eastings[0], past_weights[0], width=0.05, alpha=0.6, label='Init')
    plt.bar(past_eastings[mid], past_weights[mid], width=0.05, alpha=0.6, label='Mid')
    plt.bar(past_eastings[-1], past_weights[-1], width=0.05, alpha=0.6, label='End')
    plt.xlabel("Northings (m)",fontsize=14); plt.ylabel("Weights",fontsize=14)
    plt.legend(); plt.show()

    plt.figure(figsize=[8, 4])
    plt.plot(stamp, northings_std, '.' , markersize=12,label='Northings')
    plt.plot(stamp, eastings_std, '.', markersize=12,label='Eastings')
    plt.xlabel("Time (s)",fontsize=14); plt.ylabel("Standard deviation (m)",fontsize=14)
    plt.legend(); plt.show()
    
    if past_neff is None:
        return
    
    plt.figure(figsize=[8, 4])
    plt.plot(stamp, past_neff, label='NEFF')
    plt.plot(stamp, [len(past_northings[0])/2]*len(stamp), "--", label="N/2")
    plt.xlabel("Time (s)"); plt.ylabel("NEFF")
    plt.legend(); plt.show()
	
def plot_gpc(test_data, posterior = None, posterior_compare = None, posterior_uncert = None, training =None):

    [X_test, z_predict, class_1_test, class_0_test] = test_data 
    
    plt.scatter(X_test[class_1_test], z_predict[class_1_test], label='Test Data Actual Class 1', marker='.', color='red')
    plt.scatter(X_test[class_0_test], z_predict[class_0_test], label='Test Data Actual Class 0', marker='.', color='blue')        
        
    if training: 
        [X,z,class_0,class_1] = training
        # plot training data
        plt.scatter(X[class_1], z[class_1]+0.1, label='Training Class 1', marker='o', color='red')
        plt.scatter(X[class_0], z[class_0]-0.1, label='Training Class 0', marker='o', color='blue')

    if posterior: 
        [c_mean, label0] = posterior

        # plot posterior functions
        plt.plot(X_test, c_mean, label=label0, color='green')  
        
    if posterior_uncert: 
        [c_upper, c_lower] = posterior_uncert

        # plot posterior functions        
        plt.fill_between(X_test[:,0], #1D format needed to plot
                         c_upper[:,0], #1D format needed to plot
                         c_lower[:,0], #1D format needed to plot
                         label='Class 1 prediction uncertainty', 
                         color='lightgreen', alpha=0.3)
        
    if posterior_compare: 
        [c_mean, label0] = posterior_compare

        # plot posterior functions
        plt.plot(X_test, c_mean, label=label0, color='red')  
        
        
    plt.xlabel('$X$')
    plt.ylabel('Predicted class')
    plt.axhline(0.5, X_test.min(), X_test.max(), color='black', ls='--', lw=0.5)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
    
    
def plot_gpr_model(model):
    x = model[0]
    y = model[1]
    s = model[2]
    plt.plot(x, y, "b-", label="Prediction")

    plt.fill(
        np.concatenate([x, x[::-1]]),
        np.concatenate([y - 1.96*s, 
                        (y + 1.96*s)[::-1]]),
        alpha=0.1,
        fc="b",
        ec="None",
        label="95% confidence interval",
    )
    plt.xlabel("Bearing, deg$")
    plt.ylabel("Range, m")

    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))        
    plt.show()


def plot_gpr_likelihood(new_obs, modelled, likelihood):
    
    new_bearing = new_obs[0]
    new_range_obs = new_obs[1]
    new_range_std = new_obs[2]
    
    new_range_pred = modelled[0]
    new_range_std_pred = modelled[1]
    
    fig, (ax_main, ax_likelihood) = plt.subplots(2, 1, sharex = True)

    # Main plot with error bars
    ax_main.errorbar(new_bearing, new_range_obs.flatten(), yerr=new_range_std.flatten(), fmt="o", c="k", label="New Observations")
    ax_main.errorbar(new_bearing, new_range_pred, yerr=1.96 * new_range_std_pred.flatten(), fmt="o", c="b", label="Predicted Observations")
    ax_main.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot likelihood on the second subplot
    likelihood_mean = np.mean(likelihood)
    ax_likelihood.plot(new_bearing, likelihood, linestyle = '', marker='o', color='red', label='Observation Likelihood')
    ax_likelihood.axhline(y=likelihood_mean, linestyle='-', color='red', label=f'Mean Likelihood: {likelihood_mean:.2f}')
    ax_likelihood.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Set titles
    ax_main.set_title(f'Observation likelihood: {likelihood_mean:.2f}')

    # Set axis labels
    ax_main.set_ylabel('Range')
    ax_likelihood.set_xlabel('Bearing')
    ax_likelihood.set_ylabel('Likelihood')

    # Adjust layout
    plt.tight_layout()
    plt.show() 

# show sigma region (region of 68.3% confidence)
def sigma_contour(mu,sigma,c):
    # calculate 1sigma uncertainty bound
    e_val, e_vec = np.linalg.eigh(sigma)      
    # sort eigen values and eigen vectors in order of size
    idx=np.argsort(e_val)[::-1]
    e_val=e_val[idx]; e_vec=e_vec[:,idx]
    # standard deviation is square root of covariance 
    e_val=np.sqrt(e_val)
    w=e_val[0]*2; h=e_val[1]*2; theta=np.degrees(np.arctan2(*e_vec[:,0][::-1]))
    return Ellipse(xy=mu, width = w, height  = h, angle=theta, label = r'$\sigma$',color=c, linestyle='-')  


def plot_observation_uncertainty(sigma_observe,z,n=100):
   
    if np.all(sigma_observe == 0.0) == False and np.all(z == 0.0) == False:
        sigma_z = Matrix(2,2)        
        sigma_z = sigma_observe@z

        sigma_rtheta = Matrix(2,2) 

        sigma_rtheta[0,0] = sigma_z[0]
        sigma_rtheta[1,1] = sigma_z[1]           
        
        # observation Jacobian
        J = Matrix(2,2)
        dx_dr = np.cos(z[1])
        dx_dtheta = -z[0]*np.sin(z[1])    
        dy_dr = np.sin(z[1])
        dy_dtheta = z[0]*np.cos(z[1])    
        
        J[0,0] = dx_dr
        J[0,1] = dx_dtheta    
        J[1,0] = dy_dr
        J[1,1] = dy_dtheta        
        
        sigma_z = J@sigma_rtheta@J.T      
        
        #visualise noise in the cartesian frame      
        xy = polar2cartesian(z[0],z[1])         
        sigma_z_ = sigma_z.tolist() 

        z_ = [xy[0][0],xy[1][0]]         

        x1_range = [z_[0]-5*np.max(sigma_z)**0.5,z_[0]+5*np.max(sigma_z)**0.5,10*np.max(sigma_z)**0.5/100]
        x2_range = [z_[1]-5*np.max(sigma_z)**0.5,z_[1]+5*np.max(sigma_z)**0.5,10*np.max(sigma_z)**0.5/100]
    
        shade = 'Reds' #'Greys', 'Blues', 'Greens', 'Reds'
        c = 'k'#'k', 'b', 'g', r'
        f_x = multivariate_normal(z_, sigma_z_)

        plot_probability(f_x, z_, sigma_z_, x1_range, x2_range, shade, c);         
        z_hat = np.random.multivariate_normal([0,0], sigma_z_, n)
        z_hat[:,0]=z_[0]+z_hat[:,0]
        z_hat[:,1]=z_[1]+z_hat[:,1]

        plt.plot(z_hat[:,1],z_hat[:,0],'r.',markersize = 2)

        H_eb = HomogeneousTransformation(0,0)    
        m = Vector(2)
        m[0] = xy[0][0]
        m[1] = xy[1][0]
        
        cf=plot_2dframe(['map','b','m0'],[H_eb.H,v2t(m)],True)        
        cf._fixed_frame()#plt.plot(0,0,'bo',markersize = 6)
        plt.axis('equal')
        plt.show()


def show_observation(H_eb,t_bm,sigma,feature_label,ax, track_lines = True):
    
    t_bm=(v2t(t_bm))

    cf=plot_2dframe(['point','b',feature_label],[H_eb.H,t_bm], track_lines, False)        
        
    sigma_em = Matrix(2,2)
    sigma_em = H_eb.R@sigma@H_eb.R.T
        
    x = (H_eb.H@t_bm)[0:2].tolist()
    s = sigma_em[0:2,0:2].tolist()
        
    e=sigma_contour([x[1],x[0]],[[s[1][1],s[1][0]],[s[0][1],s[0][0]]],'r')
    e.set_facecolor('none')
    ax.add_patch(e) 


def plot_1d_binary_classification(x,h,c1,c0, binary_threshold, model_label, x_train = None, c_train = None):
    
    plt.rcParams["figure.dpi"] = 150 #make plots look nice
    plt.rc('font', family='sans serif', size=12)

    fig, ax = plt.subplots(3, 1, figsize = (8,6),  gridspec_kw={'height_ratios': [2, 1, 1]})

    # plot results for linear function
    ax[0].plot(x,h,'b', linewidth=3, label = model_label)
    ax[0].plot([min(x),max(x)],[binary_threshold, binary_threshold],'k--', linewidth=3, label = 'Decision boundary')
    if np.any(c_train != None):
        ax[0].plot(x_train,c_train,'ro', label = 'Measurements')        
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Model value')
    ax[0].set_yticks([0,0.5,1])    
    ax[0].set_ylim([-0.2,1.2])        
    ax[0].legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")

    # plot c1 probability
    ax[1].plot(x,c1,'ko',label='Predict')
    ax[1].set_yticks([0,1]); ax[1].set_yticklabels(['False', 'True'])
    if np.any(c_train != None):
        x_gt = []
        c_gt = []
        for i in range(len(c_train)):
            if c_train[i] == 1:
                x_gt.append(x_train[i])
                c_gt.append(1)         
        ax[1].plot(x_gt,c_gt,'ro',label='True')
        ax[1].legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")
        
    ax[1].set_ylabel('c+'); ax[1].set_ylim([-0.2,1.2]);
    ax[1].set_xticklabels([])    

    # plot c1 probability    
    ax[2].plot(x,c0,'ko',label='Predict')
    ax[2].set_yticks([0,1]); ax[2].set_yticklabels(['False', 'True'])
    if np.any(c_train != None):
        x_gt = []
        c_gt = []
        for i in range(len(c_train)):
            if c_train[i] == 0:
                x_gt.append(x_train[i])
                c_gt.append(1)         
        ax[2].plot(x_gt,c_gt,'ro',label='True')    
        ax[2].legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")
        
    ax[2].set_ylabel('c-'); ax[2].set_ylim([-0.2,1.2]);
    ax[2].set_xlabel('x')

    plt.show()

def plot_gp_functions(x,fx_samples, mu, std, xlim = [0,1],ylim = [-1,1],n_show = 4, true_fx = None, measurements = None):
    
    plt.rcParams["figure.figsize"] = (5,3) #make plots look nice
    plt.rcParams["figure.dpi"] = 150 #make plots look nice
    plt.rc('font', family='sans serif', size=18)
    

    # Plot the mean and the shaded sigma band
    plt.plot(x, mu, 'k-',linewidth=3, label = 'Mean fn')
    plt.fill_between(x.flat, (mu - 2*std).flat, (mu + 2*std).flat, color='gray', alpha=0.2, edgecolor='none',  label = '2 sigma')
    
    for i in range(n_show):
        if i == 0: plt.plot(x, fx_samples[:,i:i+1], 'k--', linewidth=0.5,  label = 'Sampled fn')
        else: plt.plot(x, fx_samples[:,i:i+1], 'k--', linewidth=0.5)            
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.xlim([xlim[0],xlim[1]]); plt.ylim([1.8*ylim[0],1.8*ylim[1]])
    plt.xticks([xlim[0],(xlim[1]+xlim[0])/2,xlim[1]]); plt.yticks([ylim[0],0,ylim[1]])
    
    if not np.all(true_fx == None): plt.plot(x, true_fx, 'r-',linewidth=1, label = 'Actual fn')
        
    if not np.all(measurements == None): 
        x = measurements[0]
        fx = measurements[1]
        fx_error = measurements[2]
        plt.errorbar(x.flatten(), fx.flatten(), abs((2*fx_error).flatten()), fmt='o', color='r', ecolor='r', label = 'Measurements')
    
    plt.legend(bbox_to_anchor=(1.05, 1.0),loc="upper left")
    plt.show()

def plot_graph(graph_object, p_gt_path, H_em, m_gt, m_labels):
    
    graph=copy.copy(graph_object)
    #establish the map
    fig = plt.figure()
    ax = plt.subplot(111, aspect='equal')

    for i in range(len(m_gt)):        
        cf=plot_2dframe(['map','e',m_labels[i]],[H_em.H,v2t(m_gt[i])],False)
        cf._point()
    
    for k in range(len(graph.edge)):  
        if graph.edge[k][0] == 'landmark':
            i = graph.edge[k][1]
            l = graph.edge[k][2]
            z_il = graph.edge[k][3]

            X_igt = HomogeneousTransformation(p_gt_path[i][0:2],p_gt_path[i][2])
            X_i = HomogeneousTransformation(graph.pose[i][0:2],graph.pose[i][2])  
            show_observation(X_i,z_il,(graph.landmark_covariance[l]),'m',ax)           

        if graph.edge[k][0] == 'motion':

            i = graph.edge[k][1]
            j = graph.edge[k][2]

            X_igt = HomogeneousTransformation(p_gt_path[i][0:2],p_gt_path[i][2])
            X_i = HomogeneousTransformation(graph.pose[i][0:2],graph.pose[i][2])                            

            x = (graph.pose[i][0:2]).tolist()
            s = (graph.pose_covariance[i])[0:2,0:2].tolist()

            e=sigma_contour([x[1],x[0]],[[s[1][1],s[1][0]],[s[0][1],s[0][0]]],'g')
            e.set_facecolor('none')
            ax.add_patch(e)            


            X_jgt = HomogeneousTransformation(p_gt_path[j][0:2],p_gt_path[j][2])
            X_j = HomogeneousTransformation(graph.pose[j][0:2],graph.pose[j][2])                    

            x = (graph.pose[j][0:2]).tolist()
            s = (graph.pose_covariance[j])[0:2,0:2].tolist()

            e=sigma_contour([x[1],x[0]],[[s[1][1],s[1][0]],[s[0][1],s[0][0]]],'g')
            e.set_facecolor('none')
            ax.add_patch(e)            

            cf=plot_2dframe(['pose','b','b_'],[X_i.H,X_j.H],True)
            cf._fixed_frame()
            cf._pose()        

            cf=plot_2dframe(['pose_gt','B','B_'],[X_igt.H,X_jgt.H],True)
            cf._fixed_frame()
            cf._pose()        


    # make plots
    plt.axis('equal')
    plt.ylabel('Primary')
    plt.xlabel('Secondary')

    # Get handles and labels to remove duplicates in the legend
    handles, labels = plt.gca().get_legend_handles_labels()

    # Create a dictionary to track unique labels
    unique_labels = {}

    # Remove duplicates
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels[label] = handle

    plt.legend(handles=unique_labels.values(), labels=unique_labels.keys(),bbox_to_anchor=(1.05, 1.0),loc="upper left")
    plt.show()    
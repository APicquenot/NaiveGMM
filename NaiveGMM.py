import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import copy
import warnings
import itertools
import pandas as pd # Necessary for PDF calculation only


def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
    return g1 + g2

def single_gaussian(x, A1, mu1, sigma1):
    g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
    return g1


def chi(y_,y_fit_,params_): # used to estimate quality of the fit
    resid = (y_ - y_fit_)
    return np.sum(resid**2/(y_+1e-3))

def rotate(x, y, theta): # rotate a system of coordinates from angle theta within the plan

    x_rot = (x * np.cos(theta) + y * np.sin(theta))
    y_rot = (y * np.cos(theta) - x * np.sin(theta))
    
    return x_rot, y_rot


class RotationTree:
    # Class to generate new dimensions through successive rotations.
    # Level 1 rotations all generate 2 new dimensions (for each angle) from rotating each couple of dimensions in original data
    # Level 2 rotations generate new dimensions by rotating dimensions obtained in Level 1 with dimensions in original data
    # etc, until Level L, with L ≤ n-1
    # =======> Handled through a tree of depth L
    def __init__(self, dim_list, dim_used=None, new_dims=None,proj=None, parent=None):
        self.dim_list = dim_list
        self.dim_used = dim_used or []
        self.new_dims = new_dims or []
        self.proj = proj or []
        self.parent = parent
        self.children = []
        if parent is None:
            self.level = 0
        else:
            self.level = parent.level + 1

    def add_child(self, dim_list, dim_used=None, new_dims=None,proj=None):

        child = RotationTree(dim_list,dim_used, new_dims,proj, parent=self)
        self.children.append(child)
        return child

    def __repr__(self):
        return f"RotationTree(dim_list={self.dim_list})"

def traverse(node): #debugging tool
    print(node.dim_list)
    print(len(node.new_dims))

    for child in node.children:
        traverse(child)
def count_nodes(node): #debugging tool
    total = 1  # count this node
    for child in node.children:
        total += count_nodes(child)
    return total

    
def create_tree(node,max_depth,ndim):
    for child in node.children:
        
        if (child.level<max_depth) & (child.children==[]):
            dim_list=child.dim_list
            
            for n in (dim_list):

                new_list=copy.copy(dim_list)
                new_list.remove(n)
                child.add_child(dim_list=new_list,dim_used=[n],new_dims=[])
            create_tree(child,max_depth,ndim)

def rotate_tree(node,ALL_data,Angles,normalize_rot=True):
    for child in node.children:
        if (child.new_dims==[]):
            
            new_dims=[]
            proj=[]
            
            dim_used=child.dim_used
            prev_dims=node.new_dims
            prev_proj=node.proj
            
            for d in dim_used:  # Dimensions still not rotated
                for i,D in enumerate(prev_dims): # Already rotated dimensions in previous node
                
                    for theta in Angles:  # NEW DIMS DEFINED BY ROTATING WITH CHOSEN ANGLES
                    
                        x_rot, y_rot = rotate(ALL_data[d], D, theta)
                        if normalize_rot:
                            x_rot = (x_rot-x_rot.min())/(x_rot.max()-x_rot.min())
                            y_rot = (y_rot-y_rot.min())/(y_rot.max()-y_rot.min())
                        new_dims.append(x_rot)
                        new_dims.append(y_rot)
                        proj_theta=prev_proj[i]*np.sin(theta) #Y
                        proj_theta[d]=np.cos(theta) #X
                        proj.append(proj_theta)
                        proj_theta=prev_proj[i]*np.cos(theta) #Y
                        proj_theta[d] = -np.sin(theta) #X
                        proj.append(proj_theta)
                        
            child.new_dims = new_dims
            child.proj=proj
            
            rotate_tree(child,ALL_data,Angles)

def collect_dims(node):
    NEW_DIMS = [node.new_dims]
    for child in node.children:
        NEW_DIMS.extend(collect_dims(child))
    return NEW_DIMS

def collect_proj(node):
    PROJ = [node.proj]
    for child in node.children:
        PROJ.extend(collect_proj(child))
    return PROJ

def assemble_dims(new_dims_list):
    new_dims_=[]
    for d in new_dims_list:
        for dims in d:
            new_dims_.append(dims)
    new_dims_=np.asarray(new_dims_)
    return new_dims_
    
def NaiveGMM(ALL_data,dim_plot=[],Rotation_Level=0,Angles=[np.pi/4],normalize_rot=False,out_param=False, show_warnings=True,max_run=1000):

    # ------------------------------------------------------ NaiveGMM ----------------------------------------------------- #
    #
    # ALL_data is a [n_dim,n_events] array. Each Dimension is considered individually.
    # Output is [n_events] array containing the estimated probability for each event to belong to the largest distribution.
    # => Be careful of the order when sizes are similar.
    #
    # dim_plot = dim (int) or [dim1,dim2,...] (int array) containing dimensions to plot. (dim starts at 0)
    # If dim_plot == 'All', all dimensions will be plotted.
    #
    # You can generate new dimensions through rotation with Rotation_Level.
    #    # Level 1 => 2D rotations
    #    # Level 2 => 3D rotations
    #    # Level L => L+1D rotations, thus L ≤ n-1
    # Level 1 or 2 recommended, higher levels increase computing time considerably
    # By default, only one angle : 45° to maximize difference between original dimensions and rotated ones.
    #
    # normalize_rot (bool): Option to normalize extra dimensions obtained through rotation. Non-normalized dimensions are more 
    # likely to be skipped during the fitting step, so this is useful if you but this option should not be used when using the functional 
    # form of NaiveGMM, in order to preserve linearity of the change of coordinates.
    #
    # out_param (bool): Option to return the parameters necessary to write the functional form of the calculated probabilities.
    # If True, NaiveGMM returns Probas, parameters instead of just Probas
    #
    # show_warnings (bool): Option to return which dimensions were not properly fitted.
    #
    # max_run (int): Maximum number of iteration in the fits. 
    #
    # --------------------------------------------------------------------------------------------------------------------- #


    all_probas=[] # Probas along each dimension, used to obtain global proba as an output
    all_ratios=[] # Weights to determine how useful each dimension is for the separation.
                  # Defined as chi2(one Gaussian) / chi2(two Gaussians)
                  # => Two clearly separated peaks give a large weight, overlapping distributions give a small weight.

    if (isinstance(ALL_data,np.ndarray)==False) or (len(ALL_data.shape)!=2):
        raise ValueError("ALL_data must be a [n_dim,n_events] array.")
    
    n_dim = ALL_data.shape[0] # Dimensionality of original data, can be increased with rotations

    
    if ALL_data.shape[1]<n_dim:
       warnings.warn(f"More dimensions than events. ALL_data must be a [n_dim,n_events] array, have you checked the order?")


    if out_param:
        d = {"A1": [], "mu1": [], "sigma1": [], "A2": [], "mu2": [], "sigma2": []}
        output_parameters = pd.DataFrame(data=d)
        
        columns = ['X_%i'%i for i in range(n_dim)]
        data = [[int(i==j) for i in range(n_dim)]for j in range(n_dim)]
        dimensions = pd.DataFrame(data=data, columns=columns)
    
    # -------------------------------------------------- Rotation Option -------------------------------------------------- #
    
    if Rotation_Level > 0:

    
        if Rotation_Level>n_dim-1: raise ValueError("Rotation_Level cannot be higher than data_dim - 1")
 
        rotation_root = RotationTree(dim_list=list(np.arange(n_dim)),new_dims=[])

        couple_ind = np.asarray(list(itertools.combinations(range(n_dim), 2)))
        
        for coup in couple_ind:  # All combinations of two dimensions to rotate for Level 1 rotations
    
            dim_list=copy.copy(rotation_root.dim_list)
            dim_list.remove(coup[0]) # DON'T USE TWICE THE SAME DIM IN SUCCESSIVE ROTATIONS
            dim_list.remove(coup[1])
            new_dims=[]
            proj=[]    
            for theta in Angles:  # NEW DIMS DEFINED BY ROTATING WITH CHOSEN ANGLES
            
                x_rot, y_rot = rotate(ALL_data[coup[0]], ALL_data[coup[1]], theta)
                if normalize_rot:
                    x_rot = (x_rot-x_rot.min())/(x_rot.max()-x_rot.min())
                    y_rot = (y_rot-y_rot.min())/(y_rot.max()-y_rot.min())
                new_dims.append(x_rot)
                new_dims.append(y_rot)
                proj_theta=np.zeros(shape=(n_dim))
                proj_theta[coup[0]] = np.cos(theta)#X
                proj_theta[coup[1]] = np.sin(theta)#Y
                proj.append(proj_theta)
                proj_theta=np.zeros(shape=(n_dim))
                proj_theta[coup[0]] = -np.sin(theta)#X
                proj_theta[coup[1]] = np.cos(theta)#Y
                proj.append(proj_theta)
            rotation_root.add_child(dim_list=dim_list,new_dims=new_dims,dim_used=[couple_ind],proj=proj)
            
            
        create_tree(rotation_root,max_depth=Rotation_Level,ndim=n_dim) # Create tree of depth L to handle the rotations if L > 1

        for child in rotation_root.children:
            rotate_tree(child,ALL_data,Angles,normalize_rot=normalize_rot) # Generate new dimensions through rotations
        
        new_dims_ = assemble_dims(collect_dims(rotation_root)) # Collect all the new dimensions generated through rotations
        all_proj_ = assemble_dims(collect_proj(rotation_root)) # Collect all the new projections information

        if out_param:
            new_rows = pd.DataFrame(data=all_proj_, columns=columns)
            dimensions = pd.concat([dimensions, new_rows], ignore_index=True)
                        
        ALL_data=np.concatenate((ALL_data,new_dims_),axis=0)  # Add new dimensions to the data
        n_dim = ALL_data.shape[0] # Update data dimension

    # ---------------------------------------------------- Main Loop ---------------------------------------------------- #
    
    for i in range(n_dim): # Loop on all dimensions of original data + new dimensions obtained through rotations
        
        x_data = ALL_data[i]
        
        bins_gauss=np.linspace(x_data.min()-x_data.min()*0.1,x_data.max()+x_data.max()*0.1,60)
        
        y, x_edges = np.histogram(x_data, bins=bins_gauss, density=True)
        x = 0.5 * (x_edges[1:] + x_edges[:-1])  # bin centers
    
        # --- Model: sum of two Gaussians ---
    
        # Initial guesses: works well in general, can be modified if needed
        p0 = [
            0.5, np.percentile(x_data, 25), np.percentile(x_data, 25)/12,
            0.5, np.percentile(x_data, 75), np.percentile(x_data, 25)/12
        ]
        Bounds=([0,0,0,0,0,0],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
        
        # --- Fit ---
        
        flag=0    #Security flag to avoid a Value Error in case of terribly bad fit
        
        try:
            params, _ = curve_fit(double_gaussian, x, y, p0=p0,maxfev=max_run,bounds=Bounds)
            A1, mu1, sigma1, A2, mu2, sigma2 = params
        
        except (ValueError, RuntimeError):
            if show_warnings:
                warnings.warn(f"Bad fit in dimension {i} (2 gaussians): dimension ignored.")
            A1, mu1, sigma1, A2, mu2, sigma2 = p0
            flag=1 #SECURITY
            
        g1 = A1 * np.exp(-(x - mu1)**2 / (2 * sigma1**2))
        g2 = A2 * np.exp(-(x - mu2)**2 / (2 * sigma2**2))
        
        y_fit = g1+g2

        if np.isnan(y_fit).sum()>0:
            flag=1  #SECURITY
            if show_warnings:
                warnings.warn(f"Bad fit in dimension {i} (2 gaussians): dimension ignored.")
        
    
        # --- Model: one Gaussian to compare with the two Gaussians model---
    
        # Initial guess: works well in general, can be modified if needed
        p0_alt = [
            0.5, np.percentile(x_data, 50), np.percentile(x_data, 50)/6,
        ]
        Bounds=([0,0,0],[np.inf,np.inf,np.inf])
        
        # --- Fit ---
        
        try:
            params_alt, _ = curve_fit(single_gaussian, x, y, p0=p0_alt,maxfev=max_run,bounds=Bounds)
            A1_alt, mu1_alt, sigma1_alt = params_alt
        
        except (ValueError, RuntimeError):
            if show_warnings:
                warnings.warn(f"Bad fit in dimension {i} (1 gaussian): dimension ignored.", UserWarning)
            params_alt = p0_alt
            flag=1 #SECURITY

        y_fit_alt = single_gaussian(x, *params_alt)
    

        if np.isnan(y_fit_alt).sum()>0:
            flag=1 #SECURITY
            if show_warnings:
                warnings.warn(f"Bad fit in dimension {i} (1 gaussian): dimension ignored.", UserWarning)
        
        if out_param:
            new_row = pd.DataFrame({"A1": [A1], "mu1": [mu1], "sigma1": [sigma1], "A2": [A2], "mu2": [mu2], "sigma2": [sigma2]})
            output_parameters = pd.concat([output_parameters, new_row], ignore_index=True)
        
        # ------------------------------------------- Estimate probas along dim i ------------------------------------------- #
                
        if flag==1: # Ignore this dimension if one of the fits goes wrong
            ratio=0
            P_gauss1=np.zeros(shape=(len(x_data)))
                      
        else:
            ratio=(chi(y,y_fit_alt,params_alt)/chi(y,y_fit,params))**2 # Ratio used to estimate how useful a dimension is for the separation
            g1_ = A1 * np.exp(-(x_data - mu1)**2 / (2 * sigma1**2))
            g2_ = A2 * np.exp(-(x_data - mu2)**2 / (2 * sigma2**2))
            P_gauss1=g1_/(g1_+g2_+1e-5)
            

        
        all_probas.append(P_gauss1)
        all_ratios.append(ratio)

        # --------------------------------------------------- Plot Option --------------------------------------------------- #
        
        if isinstance(dim_plot,int): dim_plot=[dim_plot]
            
        if (dim_plot=='All') or (i in dim_plot):
            
            
            ## --- Plot ---

            print(" Ratio = %f" %ratio)
            
            plt.hist(x_data, bins=50, density=True, alpha=0.4, label="Data")
            plt.plot(x, y_fit, 'k-', label="Total fit")
            plt.plot(x, g1, '--', label="Gaussian 1")
            plt.plot(x, g2, '--', label="Gaussian 2")
            plt.plot(x, y_fit_alt, '--', label="Single Gaussian")
            
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("Density")
            plt.title("Fits along dim %i"%i)
            plt.show()
            
    
    
    all_probas=np.asarray(all_probas)
    all_ratios=np.asarray(all_ratios)

    all_ratios/=np.sum(all_ratios) #ratio nornalization
    
    # ---------------------------------------- Ordering of probas in each dimension ------------------------------------------ #

    # Use most trustful events from most trustful dimension to make sure "gaussian 1" and "gaussian 2" are always the same.
    # => Define trusted events and search in which distribution they fall for each dimension.
    
    trusted_dim = all_ratios.argmax() # Find most trustful dim
    trusted_events = np.where(all_probas[trusted_dim]>0.7*all_probas[trusted_dim].max()) # Most trusted events in most trusted dim

    all_probas_ordered=[]

    for i in range(n_dim):
        
        if np.sum(all_probas[i][trusted_events])>np.sum(1-all_probas[i][trusted_events]): # Where are most trusted events?
            all_probas_ordered.append(all_probas[i])
        else:
            all_probas_ordered.append(1-all_probas[i])
            if out_param:
                output_parameters.loc[i,'A1'],output_parameters.loc[i,'A2'] = output_parameters.loc[i,'A2'],output_parameters.loc[i,'A1']
                output_parameters.loc[i,'mu1'],output_parameters.loc[i,'mu2'] = output_parameters.loc[i,'mu2'],output_parameters.loc[i,'mu1']
                output_parameters.loc[i,'sigma1'],output_parameters.loc[i,'sigma2'] = output_parameters.loc[i,'sigma2'],output_parameters.loc[i,'sigma1']
            
    
    all_probas_ordered=np.asarray(all_probas_ordered) # Probas along all dimensions ordered, finally

    if out_param: output_parameters['W_d'] = all_ratios

    # ------------------------------- Summing probas to get global PROBAS and order by size ---------------------------------- #
    
    PROBAS = (all_probas_ordered*all_ratios[:,np.newaxis]).sum(axis=0) # Sum of probas on each dimension weighted with ratio
    
    if len(PROBAS[PROBAS>0.5]) < len( (1-PROBAS)[(1-PROBAS)>0.5] ): 
        PROBAS = 1-PROBAS # Order by size
        if out_param:
            output_parameters['A1'],output_parameters['A2'] = output_parameters['A2'],output_parameters['A1']
            output_parameters['mu1'],output_parameters['mu2'] = output_parameters['mu2'],output_parameters['mu1']
            output_parameters['sigma1'],output_parameters['sigma2'] = output_parameters['sigma2'],output_parameters['sigma1']
            
    if out_param:
        output_parameters = pd.concat([dimensions,output_parameters],axis=1)
        return PROBAS, output_parameters
            
    return PROBAS
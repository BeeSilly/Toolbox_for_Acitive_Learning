import os
import time
import json
import pickle
import itertools
import numpy as np
import tensorflow as tf
import scipy.stats as st
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d
import sklearn.gaussian_process as gp

from matplotlib import cm
from tensorflow import keras
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.neighbors import KernelDensity
from scipy.stats import norm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut


"""The structure of the script is as following:

def est_max_distance(ndarry):
def max_distance(ndarry):
def nparray_process(x_train, x_all, y_all=False, process='in'): 
def plotting_2D(x, y, all_x=False, all_true_y=False, ...):
def plotting_3D_x_in_2D(x, y, all_x=False, all_true_y=False, ...):
def plotting_3D_y_in_2D(x, y, all_x=False, all_true_y=False, ...):
def plotting_3D_stem_contourf(x, y, xlabel='x', ylabel='y', zlabel='y', ...):
def plotting(x, y, all_x=False, all_true_y=False, all_ptd_y=False, ...):

class MyCallback(tf.keras.callbacks.Callback):
    __init__(self, loss_threshold, x_train, y_train, interval, no_of_epochs...):
    def evaluation_by_interval(self, y_predictions):
    def on_epoch_end(self, epoch, ...,logs=None):

class active_learning_solver():
    def __init__(self):
    def define_a_single_model(self, nn_struc_para_list, ...):
    def add_model_to_QBC(self, model_and_its_parameter, number=1):
    def delete_model_from_QBC(self, number_of_the_model):    
    def fit(self, x_train, y_train, no_of_epochs, batch_size, ...):
    def predict(self, x_range, pdf_type='Gaussian', ...):
    def predict_n_steps(self, true_y_next=False, n_step=False, ...):
    def gaussian_est(self, y_np_2d_array):
    def kernel_density_est(self, y_np_2d_array):
    def pred_x_next_by_rand_guess(self, x_train, y_train, x_range, ...):
    def gaussian_processes_solver(self, x_train, y_train, x_range, ...):
    def get_parameters(self):
    def save_model(self, model_positions=False, model_names=False, path=False):
    def load_model(self, model_names=False, path=False):
"""


def function_generator(x_min, x_max, dimension=1, random_seed=None):
    """Generate 1D or 2D function with the given range and random seed.

    Args:
    x_min:
        The lower bound of the function;
    x_max:
        The upper bound of the function;
    dimension:
        The dimension type of generated function, only can be 1 or 2;
    random_seed: 
        The random seed to generate parameters for the function.

    Returns:
        A 1D or 2D function object.
    """
    # Check the bound, if min is not less than max, then raise error.
    if x_max <= x_min:
        raise Exception('x_min should be strictly greater than x_max.')
    # Check whether the random seed is assigned or not.
    if random_seed:
        np.random.seed(random_seed)

    # Generate the intermediate parameters for the basic functions
    n = np.random.randint(5, 25, 1).item()
    w = np.random.uniform(low=x_min, high=x_max, size=3*n)
    m = np.sort(np.random.randint(low=x_min, high=x_max, size=3))
    v = np.random.uniform(low=1, high=30, size=3)
    a = np.random.uniform(high=10, size=3)

    # Define some basic 1D and 2D functions to for combining a complex function;
    # Here p is the 1D function, and f1 ro f4 are 2D functions;
    p = lambda a, m, v, x: sum([a[i]*np.exp(-(((x-m[i])/v[i])**2)/2)\
                           for i in range(3)])
    f1 = lambda x, w: 0.005*w[2]*np.log((x[:,0]-w[0])**2 + (x[:,1]-w[1])**2 +\
                      0.01*np.abs(w[0])/(np.abs(w[0])+np.abs(w[1])))
    f2 = lambda x, w: 0.2*np.exp(-(x[:,0]+x[:,1]-w[0])**2/3) + \
                      0.1*np.exp(-(x[:,0]-x[:,1]-w[1])**2/3)
    f3 = lambda x, w: 0.01*np.exp(-(x[:,0]+x[:,1]-w[0])**2/5) + \
                      0.02*np.log((x[:,0]-w[1])**2 + (x[:,1]-w[2])**2 + 0.005)
    f4 = lambda x, w: 0.01*np.exp(-(x[:,0]+x[:,1]-w[0])**2/5) - \
                      0.02*np.log((x[:,0]-w[1])**2 + (x[:,1]-w[2])**2 + 0.03)

    # Check the dimension information for the output;
    if dimension == 1:
        function = lambda x: -1*p(a,m,v,x) + a[0]*np.sin(x)/(1+np.abs(x)) +\
                             np.sin(x)/(1+np.abs(x)) - 2*np.cos(0.01*x)
    if dimension == 2:
        f = np.random.choice([f1, f2, f3, f4], 1).item()
        function = lambda x: np.sum(np.array([f(x, w[i:i+3]) \
                            for i in range(n)]), axis=0)

    return function

def x_in_space_check(x, search_space_info):
    """Check whether a point in the search space, if not make it vaild.

    Args:
    x:
        A data point in the designed dimensions and data types;
    search_space_info:
        A special designed np.array with shape in nx3 to indicate the search
        space for the solution. n is the dimension information for x. The
        first column of the array indicates the data type. '1' stands for 
        continuous data, '2' stands for categorized data and '3' stands for
        encoded data. For continuous data, the second and third column are
        the minimum and maximum limit, any number between them is valid; 
        For categorized data, the second and third column are the lowest and
        highest limit, only integer number between them is valid; For 
        encoded data, the second and third columns are the beginning and
        ending positions of the encoding group, each encoded group should be
        in continuous rows, and only a '1' and others in '0' is valid; An
        example for search_space is as following:
        np.array([[1, 1.1, 5.1],  # continuous data between 1.1 to 5.1
                [3, 1, 3],        # encoded data group 1 from row 1
                [3, 1, 3],        # encoded data group 1 with row 2
                [3, 1, 3],        # encoded data group 1 ends in row 3
                [2, 0, 5],        # categorized data in 0, 1, 2, 3, 4, 5.
                [1, -10, 20],     # continuous data between -10 to 20
                [3, 6, 7],        # encoded data group 2 from row 6
                [3, 6, 7],        # encoded data group 2 ends in row 7
                [2, 2, 4],        # categorized data in 2, 3, 4.
                ]])  
        In this case, a valid data point for x is as following:
        np.array([2.2, 0, 1, 0, 3, 8.5, 1, 0, 4])

    Returns:
        An updated x fulfill the requirement as a point in the search space.
    """
    # Define an empty list container to store intermediate information.
    x_all_parts = []
    # Find out how many data type in the data.
    x_types = np.unique(search_space_info[:,0])
    # 1 for continuous_data; 2 for categorized_data; 3 for encoded_data;
    for i in [1, 2, 3]:
        # Create an array to filter out the data in desired type
        data_type_mask = i * np.ones_like(x)
        flags = data_type_mask == search_space_info[:,0]
        data_in_x = x * flags

        # Filter out the constraints only for the data in desired type;
        # Set the other positions into np.nan to avoid issues from value in 0.
        flags_all = np.column_stack((flags, flags, flags))
        data_constrain = search_space_info * flags_all
        for j in range(len(data_constrain)):
            if data_constrain[j][0] == 0:
                data_constrain[j] = np.nan * np.ones_like(data_constrain[0])
        # Correct the outer-bound issues for continuous and categorized data
        if (i == 1 or i == 2) and (i in x_types):
            min_error = data_in_x < data_constrain[:,1]
            max_error = data_in_x > data_constrain[:,2]
            errors = min_error + max_error
            corrects = np.zeros_like(x)

            # np.sum(errors) being not 0 means there are values out of range
            if np.sum(errors) != 0:
                corrects = data_constrain[:,1] + np.random.uniform(size=x.size)\
                             * (data_constrain[:,2] - data_constrain[:,1])
            x_con_cat_part = data_in_x * (errors == 0) + corrects * (errors != 0)

            # For categorized data, it should be in integer, thus use np.round
            # Append the updated continuous and categorized part to the list
            if i == 2: x_con_cat_part = np.round(x_con_cat_part)
            x_all_parts.append(x_con_cat_part)

        # Correct the encoded data, only one place can be 1 in an encoded group
        elif (i == 3) and (i in x_types):
            encoded_part = []
            # Check how many encoded groups in the data; np.nan does not count
            for j in np.unique(data_constrain[:,1]):
                if not np.isnan(j):
                    ones_j = j * np.ones_like(data_constrain[:,1])
                    x_part_j = data_in_x * (ones_j == data_constrain[:,1])

                    # For data just with one column in 0/1, use random method
                    # For others use sigmoid function to set the maximum into 1
                    if np.sum(x_part_j) == 1:
                        encode_j = np.random.choice([0, 1], 1) * x_part_j
                    else:
                        sigmoid_j = 1/(1 + np.exp(-1 * x_part_j))
                        encode_j = sigmoid_j == np.max(sigmoid_j)
                    encoded_part.append(encode_j)
            # Put all the updated encoded groups together
            encoding = np.zeros_like(x)
            for part in encoded_part:
                encoding += part
            x_all_parts.append(encoding)

        # Put the continuous, categorized and encoded parts together to return
        x_checked = np.zeros_like(x)
        for part in x_all_parts:
            part[np.isnan(part)] = 0
            x_checked += part

    return x_checked

def generate_particle_from_search_space(search_space_info, nr_particles):
    """Randomly generate the assigned number of particles from the search space.

    Args:
    search_space_info:
        A special designed np.array with shape in nx3 to indicate the search
        space for the solution. n is the dimension information for x. The
        first column of the array indicates the data type. '1' stands for 
        continuous data, '2' stands for categorized data and '3' stands for
        encoded data. For continuous data, the second and third column are
        the minimum and maximum limit, any number between them is valid; 
        For categorized data, the second and third column are the lowest and
        highest limit, only integer number between them is valid; For 
        encoded data, the second and third columns are the beginning and
        ending positions of the encoding group, each encoded group should be
        in continuous rows, and only a '1' and others in '0' is valid; An
        example for search_space is as following:
        np.array([[1, 1.1, 5.1],  # continuous data between 1.1 to 5.1
                [3, 1, 3],        # encoded data group 1 from row 1
                [3, 1, 3],        # encoded data group 1 with row 2
                [3, 1, 3],        # encoded data group 1 ends in row 3
                [2, 0, 5],        # categorized data in 0, 1, 2, 3, 4, 5.
                [1, -10, 20],     # continuous data between -10 to 20
                [3, 6, 7],        # encoded data group 2 from row 6
                [3, 6, 7],        # encoded data group 2 ends in row 7
                [2, 2, 4],        # categorized data in 2, 3, 4.
                ]])  
        In this case, a valid data point for x is as following:
        np.array([2.2, 0, 1, 0, 3, 8.5, 1, 0, 4])
    nr_particles:
        The needed number of particles, should be a positive integer.

    Returns:
        A np.array containing the needed number of points from the search space.
    """
    # Find out the dimension information for a single particle
    dimensions = len(search_space_info)
    all_particle_element = np.zeros([dimensions, nr_particles])

    # Use the information in the search space to generate element of a 
    # particle in randomness and put it into all_particle_element list
    for i in range(dimensions):
        low = search_space_info[i][1]
        high = search_space_info[i][2]
        all_particle_element[i] = np.random.uniform(low, high, nr_particles)
    # print(all_particle_element)
    # Transpose all_particle_element to have an array containing all the raw
    # particles which not meet the category and encoding data requirements
    raw_all_particle = np.transpose(all_particle_element, axes=(1,0))
    # print('raw_all_particle', raw_all_particle)
    all_particles = np.zeros_like(raw_all_particle)

    # Iterate all the raw particles and use x_in_space_check function to check 
    # and update a raw particle into a real one matching the data requirement.
    for j in range(len(raw_all_particle)):
        raw_particle = raw_all_particle[j]

        particle = x_in_space_check(raw_particle, search_space_info)
        # print('raw_particle', raw_particle, 'particle', particle)
        all_particles[j] = particle

    return all_particles.reshape(nr_particles, -1)


class Particle:
    """This class defines the particles in a swarm for PSO method.

    The methods of this class includes getting some critical information and
    the step function for the movement of the particle.

    Attributes:
    id:
        A integer identity to name/number the particles in a swarm.
    x:
        The particle's position information in the given search space.
    search_space_info:
        A special designed np.array with shape in nx3 to indicate the search
        space for the solution. n is the dimension information for x. The
        first column of the array indicates the data type. '1' stands for
        continuous data, '2' stands for categorized data and '3' stands for
        encoded data. For continuous data, the second and third column are
        the minimum and maximum limit, any number between them is valid;
        For categorized data, the second and third column are the lowest and
        highest limit, only integer number between them is valid; For
        encoded data, the second and third columns are the beginning and
        ending positions of the encoding group, each encoded group should be
        in continuous rows, and only a '1' and others in '0' is valid; An
        example for search_space is as following:
        np.array([[1, 1.1, 5.1],  # continuous data between 1.1 to 5.1
                [3, 1, 3],        # encoded data group 1 from row 1
                [3, 1, 3],        # encoded data group 1 with row 2
                [3, 1, 3],        # encoded data group 1 ends in row 3
                [2, 0, 5],        # categorized data in 0, 1, 2, 3, 4, 5.
                [1, -10, 20],     # continuous data between -10 to 20
                [3, 6, 7],        # encoded data group 2 from row 6
                [3, 6, 7],        # encoded data group 2 ends in row 7
                [2, 2, 4],        # categorized data in 2, 3, 4.
                ]])
        In this case, a valid data point for x is as following:
        np.array([2.2, 0, 1, 0, 3, 8.5, 1, 0, 4])
    v_max:
        The upper (v_max) and lower (-v_max) bound of the speed limit when a
        particle moves.
    c1:
        Individual velocity term, controls how much the particle should move
        in the direction of personal best position found so far.
    c2:
        Social velocity term, controls how much the particle should move in
        the direction of the global best location found by the swarm so far.
    w:
        Velocity inertia parameter, controls how much the particle should
        continue going in the same direction as the previous time step.
    obj_func:
        The objective function to determin the function value of the partice.
    
    """
    def __init__(self, id, x, search_space_info, v_max, c1, c2, obj_func, w=1):
        self.id = id
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.search_space_info = search_space_info
        self.v_max = v_max * np.ones_like(x)
        # The particel's current location, initialized by the given x
        self.x_v = x.reshape(1, -1) 
        self.obj_func = obj_func
        # The particel's current function value, initialized by the f(x)
        self.fx = self.obj_func(x.reshape(1, -1))
        # The particel's current speed, initialized all in 0.1
        self.v_next = 0.1 * np.ones_like(x.reshape(1, -1)) 
        # The particel's personal best place, initialized by the given x
        self.p_best_x = self.x_v
        # The function value for the particel's personal best place
        # initialized by f(x)
        self.p_best_fx = obj_func(self.x_v)
        # The swarm's global best place, initialized by the given x
        self.g_best_x = x.reshape(1, -1)
        # An list object to store the particle's movement
        self.trajectory = []

    def particle_info(self):
        """Output the particle's information into an list.

        Return:
        The particle's current location, function value, current speed (step 
        length), personal best postion and the its function value.
        """
        print('This is the No. {} particle.'.format(self.id))
        return [self.x_v, self.fx, self.v_next, self.p_best_x, self.p_best_fx]

    def step(self):
        """The particle forward a step to update its position and velocity.

        Return:
        The personal best position and its function value.
        """
        # Generate random numbers from 0 to 1 to update v_next 
        r = np.random.uniform(size=self.x_v.size)

        # update v_next by formula 2 (as following);
        # v(t+1) = w*v(t) + c1*r1(t)*(xpb-x(t)) + c2*r2(t)*(xgb-x(t))
        v_next_np = self.w * self.v_next + \
                    self.c1 * r * (self.p_best_x - self.x_v) + \
                    self.c2 * r * (self.g_best_x - self.x_v)

        # check the speed and implement capped velocity
        v_max_flag = self.v_max * (np.absolute(v_next_np) > self.v_max)
        v_next_updated = (v_next_np * (np.absolute(v_next_np) < self.v_max)) +\
                         (v_max_flag * np.sign(v_next_np))

        self.v_next = v_next_updated
        # print(v_next_np, self.v_next)

        # update the particle's position and evaluate x_next whether in the 
        # search space or not, if not update it by x_in_space_check.
        if len(self.search_space_info) > 1:
            temp_new_position = np.squeeze(self.x_v + self.v_next)
        else:
            temp_new_position = self.x_v + self.v_next
        new_position = x_in_space_check(temp_new_position, self.search_space_info)
        # Reshape the updated particle and calculate its function value
        self.x_v = new_position.reshape(1, -1)
        self.fx = self.obj_func(self.x_v)

        self.trajectory.append([self.x_v, self.fx])

        # update p_best_x if case by using L2 norm in case of the function 
        # value in multiple dimensions.
        if np.linalg.norm(self.p_best_fx) < np.linalg.norm(self.fx):
            self.p_best_x, self.p_best_fx = self.x_v, self.fx

        return self.p_best_x, self.p_best_fx


class PSO:
    """This class defines a solver for Particle Swarm Optimization.

    The methods of this class includes the swarm forwarding a single step, 
    running forward by the assigned steps and the strategy of automatic 
    running to find out the optimized minimum point.

    Attributes:
    obj_func:
        The objective function to determin the function value of the partice.
    nr_particles:
        The needed number of particles, should be a positive integer.
    search_space_info:
        A special designed np.array with shape in nx3 to indicate the search
        space for the solution. n is the dimension information for x. The
        first column of the array indicates the data type. '1' stands for
        continuous data, '2' stands for categorized data and '3' stands for
        encoded data. For continuous data, the second and third column are
        the minimum and maximum limit, any number between them is valid;
        For categorized data, the second and third column are the lowest and
        highest limit, only integer number between them is valid; For
        encoded data, the second and third columns are the beginning and
        ending positions of the encoding group, each encoded group should be
        in continuous rows, and only a '1' and others in '0' is valid; An
        example for search_space is as following:
        np.array([[1, 1.1, 5.1],  # continuous data between 1.1 to 5.1
                [3, 1, 3],        # encoded data group 1 from row 1
                [3, 1, 3],        # encoded data group 1 with row 2
                [3, 1, 3],        # encoded data group 1 ends in row 3
                [2, 0, 5],        # categorized data in 0, 1, 2, 3, 4, 5.
                [1, -10, 20],     # continuous data between -10 to 20
                [3, 6, 7],        # encoded data group 2 from row 6
                [3, 6, 7],        # encoded data group 2 ends in row 7
                [2, 2, 4],        # categorized data in 2, 3, 4.
                ]])
        In this case, a valid data point for x is as following:
        np.array([2.2, 0, 1, 0, 3, 8.5, 1, 0, 4])
    v_max:
        The upper (v_max) and lower (-v_max) bound of the speed limit when a
        particle moves.
    c1:
        Individual velocity term, controls how much the particle should move
        in the direction of personal best position found so far.
    c2:
        Social velocity term, controls how much the particle should move in
        the direction of the global best location found by the swarm so far.
    w:
        Velocity inertia parameter, controls how much the particle should
        continue going in the same direction as the previous time step.
    print_gb_update_info:
        An boolen flag to control the printing updating information of the
        global best point among the particles in the swarm.
    
    """
    def __init__(self, obj_func, nr_particles, search_space_info, v_max,
                 c1, c2, w=1, print_gb_update_info=True):
        self.obj_func = obj_func
        self.nr_particles = nr_particles
        self.search_space_info = search_space_info
        self.v_max = v_max
        self.c1 = c1
        self.c2 = c2
        self.w = w
        # An internal counter to calculate how many steps the swarm 
        # have been forwarded, thus initialized in 0
        self.step_i = 0
        self.print_gb_update_info = print_gb_update_info
        # Using the defined generate_particle_from_search_space to generate 
        # the assigned number of particels for the swarm
        all_initial_particles = generate_particle_from_search_space(
                                search_space_info, nr_particles)
        # print('particles', all_initial_particles.shape, all_initial_particles)
        # Initially randomly choose a particle from the swarm as the best point
        # And initialize the global best function value
        rand_int = np.random.randint(low=0, high=nr_particles-1, size=1).item()
        self.g_best_x = all_initial_particles[rand_int].reshape(1, -1)
        self.g_best_fx = obj_func(self.g_best_x)
        # Attach all the particles into the list as a swarm
        self.swarm = []
        for pid in range(nr_particles):
            p = Particle(pid+1, all_initial_particles[pid].reshape(1, -1),
                         search_space_info, v_max, c1, c2, self.obj_func, w)
            self.swarm.append(p)

    def step(self):
        """Method to control the particles in the swarm forward one step.
        """

        # A counter for record how many times the global best point updates
        g_best_update_flag = 0

        # Move a step forward for all the particles in the swarm and
        # Update the global minimum by comapring with every personal best point
        # in L2 norm in case of the function value in multiple dimensions;
        for particle in self.swarm:
            p_best_x, p_best_fx = particle.step()
            if np.linalg.norm(self.g_best_fx) < np.linalg.norm(p_best_fx):
                self.g_best_x, self.g_best_fx = p_best_x, p_best_fx
                g_best_update_flag += 1

        # Update the global minimum into every particle;
        for particle in self.swarm:
            particle.g_best_x = self.g_best_x

        # Print update information when global minimum is changed if activated;
        if g_best_update_flag and self.print_gb_update_info:
            print('Global Best is updated: x={}, y={}.'
                  .format(self.g_best_x, self.g_best_fx))

        self.step_i += 1  # add 1 to step counter

    def run(self, nr_steps):
        """Assign the number of steps to run to find out the optimized point.
        
        Args:
        nr_steps:
            The assigned number of steps to run.

        Return:
        The found global best point along its function value.
        """
        for i in range(nr_steps):
            self.step()
        return self.g_best_x, self.g_best_fx

    def auto_run(self, nr_steps=False, acceptable_maximum=False):
        """A method to run with some default settings for the optimum.
        
        Args:
        nr_steps:
            The assigned number of steps to run.
        acceptable_minimum:
            An acceptable minimum function value. If the self.g_best_fx is
            lower or equal than this value, the swarm stops moving.

        Return:
        The found global best point along its function value.
        """
        # If the running steps is not assigned, then 10000 is set.
        if (not nr_steps):
            print('Steps is not assigned, a max 10000 steps will be done!')
            n = 10000
        else:
            n = nr_steps

        # Define an list container to track the L2 norm of self.g_best_fx
        # for the policy of stop moving after moving 200 steps further without
        # any update of the optimum
        improvement_flag_list = []
        while n:
            self.step()
            n -= 1
            if acceptable_maximum and (self.g_best_fx >= acceptable_maximum):
                print('An acceptable_maximum fx is found after {} steps. '
                      'Swarm stops moving.'.format(self.step_i))
                break
            if n % 200 == 0:
                improvement_flag_list.append(np.linalg.norm(self.g_best_fx))
            if (len(improvement_flag_list) > 1 and
               (improvement_flag_list[-1] == improvement_flag_list[-2])):
                print('No better solution is found after another 200 steps, '
                      'the swarm stops moving.')
                break

        print('The optimal found by the swarm is: x={}, y={}.'
              .format(self.g_best_x, self.g_best_fx))

        return self.g_best_x, self.g_best_fx


def est_max_distance(ndarray):
    """Calculate the estimated max distance in O(N) among points in an array.

    Args:
    ndarray: 
        A np.array instance with maximum 2 axes, and the second axis
        can contain multiple values.

    Returns:
        A non-negative floating number to indicate the distance.
    """
    # Find out the dimension number
    n = int(ndarray.size / len(ndarray))
    if n == 1:
        # For data in 1 dimension, return the gap of max and min
        return np.max(ndarray) - np.min(ndarray)
    else:
        candidate_distance = []
        # Finding out the maxi distance in each dimension and add to the list
        for i in range(n):
            n_max, n_min = np.max(ndarray[:, i]), np.min(ndarray[:, i])
            candidate_distance.append(np.power(n_max - n_min, 2))
        # Calculate a distance by using the maximum distance in each dimension
        return np.sqrt(np.sum(candidate_distance)).item()

def max_distance(ndarray):
    """Calculate the exact maximum distance in O(n^2) among points in an array.

    Args:
    ndarray: 
        A np.array instance with maximum 2 axes, and the second axis
        can contain multiple values.

    Returns:
        A non-negative floating number to indicate the distance.
    """
    # Find out the dimension number
    n = int(ndarray.size / len(ndarray))
    if n == 1:
        # For data in 1 dimension, return the gap of max and min
        return np.max(ndarray) - np.min(ndarray)
    else:
        # Calculate and record the L2 distance between any two point
        distance = np.zeros([len(ndarray), len(ndarray)])
        for i in range(len(ndarray)):
            distance[i] = np.sqrt(
                np.sum(np.power(ndarray - ndarray[i], 2), axis=1))
        return np.max(distance).item()

def model_structure_update(model_structure, nr_neurons):
    """A function to update a NN model's architecture list by a policy.

    The policy is that if the existed NN model is not able to fit with the given
    constraint, then one neuron will be added to the model. This extran neuron
    is started to put from hidden layer to hidden layer which gradually away 
    the input layer. A updating of [1, 2, 2, 2, 1] from 6 to 11 neurons is as 
    following:
        [1, 2, 2, 2, 1]
        [1, 3, 2, 2, 1]
        [1, 3, 3, 2, 1]
        [1, 3, 3, 3, 1]
        [1, 4, 3, 3, 1]
        [1, 4, 4, 3, 1]

    Args:
    model_structure: 
        A list object with integer in each position to indicate the number of
        input variables, neurons of the specific hidden layer and the output
        variables of the last layer. The minimum length of the list is 3.
    nr_neurons:
        The assigned total number of neurons.

    Returns:
        A list object for a NN model's architecture .
    """
    # Find out the number of hidden layers from the given architecture list
    nr_hidden_layers = len(model_structure) - 2
    # Find out the minimum number of neurons for each hidden layer
    each_layer = nr_neurons // nr_hidden_layers
    # Find out how many extra neurons needed to add to the architecture list
    extra_neurons = nr_neurons % nr_hidden_layers
    # Put the minimum number of neurons to each hidden layer 
    for i in range(nr_hidden_layers):
        model_structure[i + 1] = each_layer
    # Add the extra neurons from the left side to the right side
    for i in range(extra_neurons):
        model_structure[i + 1] += 1
    return model_structure

def nparray_process(x_train, x_all, y_all=False, process='in'):
    """Process an array by the appointed method.

    Args:
    x_train: 
        A np.array instance.
    x_all: 
        A np.array instance. It should have the same size as x_train in
        the second axis.
    y_all: Default in False. Normally it is an array with the same length as
        x_all and will be a constant from 0 to 1 when process is 'threshold'.
    process: Indicate the appointed method, and can be 'in', 'inf', 'del' and
        'threshold'. When it is 'in', the function will tell whether x_all
        contains any element in x_train; When it is 'inf', the data in y_all
        with the same positions of x_train in x_all will be set into -np.inf;
        when it is 'del', the data in the positions of the element from x_train
        in x_all will be deleted; When it is 'value', the data in y_all
        with the same positions of x_train in x_all will be returned;
        When it is 'threshold', the data within the range of 
        threshold*maximum_distance_among_all_point_in_x_train of a 
        training point in x_all will be removed. 

    Returns:
        When process is 'in', the return will be True if any element in x_train 
        is contained in x_all; Otherwise, a np.array will be returned.

        For example: When x_train=np.array([0.1, 0.3]), x_all=np.array([0.1,
        0.2, 0.3, 0.4]) and y_all=np.array([1.1, 2.2, 3.3, 4.4]),
        if process='in', the return will be True since both 0.1, 0.3 are in
        x_all; if process='inf', the return will be np.array([-np.inf, 2.2,
        -np.inf, 4.4]); if process='del', the return will be np.array([2.2,
        4.4]). if process='value', the return will be np.array([1.1, 3.3]); 
        When x_train=np.array([0.1, 0.3]), x_all=np.array([0.1, 0.11,
        0.22, 0.3, 0.5]), y_all=0.1 and process='threshold', then the return
        will be np.array[(0.3, 0.5)] since the max_distance in x_train is
        0.5-0.1=0.4, and 0.1, 0.11 in (0.1-0.4*0.1, 0.1+0.4*0.1) and 0.22 in
        (0.2-0.4*0.1, 0.2+0.4*0.1) making 0.1, 0.11 and 0.22 in x_all
        supposed to be deleted.
    """
    # List object to record the indices met the requirement
    index = []
    # Case for process in 'threshold'
    if isinstance(y_all, float) and process.lower() == 'threshold':
        cp_y_all = np.copy(x_all)
        # Using the defined max_distance to get the maxi distance in x_train
        threshold_distance = max_distance(x_train)
        for i, element in enumerate(x_all):
            for j in x_train: 
                # Check whether a point in x_train is close to that in x_all 
                percentage = np.linalg.norm(element-j) / threshold_distance
                # percentage = np.sqrt(np.sum(np.power((element - j), 2))) \
                #              / threshold_distance
                if percentage <= y_all:
                    index.append(i)
        return np.delete(cp_y_all, index, axis=0)
    else:
        # Case for process in 'in', 'value', 'inf' and 'del'
        # cp_y_all, dimension = np.copy(y_all), x_train[0].size
        cp_y_all = np.copy(y_all)
        for i, element in enumerate(x_all):
            # Case for element in x_all or x_train in 1 dimension
            # if dimension == 1:
            #     if element in x_train:
            #         index.append(i)
            # # Case for that in 2 dimension and process in 'value'
            # elif dimension == 2 and len(x_train) == 1:
            #     if (x_train == element).all():
            #         index.append(i)
            # else:
            for j in x_train:
                if (element == j).all():
                    index.append(i)
        if isinstance(y_all, bool) and process.lower() == 'in':
            if len(index):
                return True
        if (not isinstance(y_all, bool)) and process.lower() == 'value':
            return cp_y_all[index]
        if (not isinstance(y_all, bool)) and process.lower() == 'inf':
            cp_y_all[index] = -np.inf
            return cp_y_all
        if (not isinstance(y_all, bool)) and process.lower() == '0':
            cp_y_all[index] = np.zeros_like(cp_y_all[0])
            return cp_y_all
        if (not isinstance(y_all, bool)) and process.lower() == 'del':
            return np.delete(cp_y_all, index, axis=0)

def plotting_2D(x, y, all_x=False, all_true_y=False, all_ptd_y=False,
                xlabel='x', ylabel='y', predicted_point=False, legend=False,
                plot_size=(12, 9), title=False, save_name=False, show=True):
    """Plot 2D graphs with x and y in both 1D during the solver processing data.

    Args:
    x: 
        A np.array instance. In this project, x_train is put here;
    y: 
        A np.array instance. In this project, y_train is put here;
    all_x:
        A np.array instance. In this project, x_range is put here;
    all_true_y: 
        A np.array instance. In this project, all the real function value of
        x_range is put here;
    all_ptd_y:
        A list instance containing np.array instance. Each of the np.array
        corresponds to a model's prediction over x_range;
    xlabel:
        A string instance to indicate the label of x-axis; 
    ylabel:
        A string instance to indicate the label of y-axis;
    predicted_point:
        A np.array instance in 1x1 as the solver's prediction for x_next
    legend:
        A list instances containing strings to indicate the legends in the
        plot; Default in False, and it will stop the plt.legend().
    plot_size:
        A list/tuple instance to control the size of the figure;
    title:
        A string object to set the tile of the figure. None if it is False;
    save_name:
        A string object to set the tile when saving the figure. The plot will
        not be saved if it is False;
    show:
        A boolean to control whether show the figure or not when running the
        code. False will not show the plot. When save_name is set in a string
        and show set in False, the desired plot will be saved but not showed.

    Returns:
        The return will return None no return is set.
    """
    pred_std = ''
    if isinstance(all_x, list) and isinstance(all_x[0], np.ndarray):
        all_x, pred_std = all_x[0], all_x[1]
    plt.figure(figsize=plot_size)
    flag_1 = isinstance(all_x, bool)
    flag_2 = isinstance(all_true_y, bool)
    flag_3 = isinstance(all_ptd_y, bool)
    if ((not flag_1) and (not flag_2)) or ((not flag_1) and (not flag_3)):
        if isinstance(all_true_y, list):
            legend.insert(0, 'true_value')
            legend.insert(1, 'noised_true_value')
            plt.plot(all_x, all_true_y[0], linestyle='dashed')
            plt.plot(all_x, all_true_y[1], linestyle='dashed')
        elif not isinstance(all_true_y, bool):
            legend.insert(0, 'true_value')
            plt.plot(all_x, all_true_y, linestyle='dashed')
        if not isinstance(all_ptd_y, bool):
            # for i, predicted_y_in_x_range in enumerate(all_ptd_y):
            #     if i == len(all_ptd_y):
            #         plt.plot(all_x, predicted_y_in_x_range, color='blue',
            #                  linewidth=2.5,  marker='o', linestyle='solid')
            #     else:
            #         plt.plot(all_x, predicted_y_in_x_range, linestyle='dashdot')
            if len(all_ptd_y) == 1:
                plt.plot(all_x, all_ptd_y[-1], color='blue',
                     linewidth=2.5, linestyle='solid')
            else:
                for predicted_y_in_x_range in all_ptd_y[:-3]:
                    plt.plot(all_x, predicted_y_in_x_range, linestyle='dashdot')
                for y_in_x_range in all_ptd_y[-3:]:
                    plt.plot(all_x, y_in_x_range, linewidth=2.5, linestyle='solid')
                # plt.plot(all_x, all_ptd_y[-1], color='blue',
                #          linewidth=2.5, linestyle='solid')
            if isinstance(pred_std, np.ndarray):
                plt.fill_between(all_x.ravel(), 
                                    (all_ptd_y[-1] - 1.96*pred_std).ravel(),
                                    (all_ptd_y[-1] + 1.96*pred_std).ravel(),
                                    color="tab:orange",
                                    alpha=0.15,
                            )
                legend.append(r"95% confidence interval")
                    
        if not isinstance(predicted_point, bool):
            legend.append(''.join(['pred_x=', str(np.round(np.squeeze(
                predicted_point), 2))]))
            plt.axvline(x=predicted_point, linestyle='dotted', color='r')
        legend.append(''.join([str(len(x)), '_training_points']))
        plt.scatter(x, y, s=30, c='r')
        for i in range(len(x)):
            plt.annotate(str(i+1), xy=(x[i], y[i]), xytext=(x[i], y[i]))
    else:
        plt.plot(x, y, label=legend)
        if not isinstance(predicted_point, bool):
            plt.axvline(x=predicted_point[0], linestyle='dotted', color='r')
    plt.grid()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if not isinstance(legend, bool) and len(legend)<71:
        if len(legend) <= 10:
            plt.legend(legend)
        else:
            plt.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    else:
        if not isinstance(predicted_point, bool):
            txt = 'pred_x=' + str(np.round(np.squeeze(predicted_point), 2))
            plt.annotate(txt, xy=[predicted_point, plt.ylim()[1]])
    if title:
        plt.title(title)
    if save_name:
        plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plotting_3D_x_in_2D(x, y, all_x=False, all_true_y=False, all_ptd_y=False,
                        zlabel='y', predicted_point=False, legend=False,
                        plot_size=(12, 9), title=False, save_name=False,
                        show=True):
    """Plot 3D graphs with x in 2D and y in 1D during the solver processing data.

    Args:
    x: 
        A np.array instance. In this project, x_train is put here;
    y: 
        A np.array instance. In this project, y_train is put here;
    all_x:
        A np.array instance. In this project, x_range is put here;
    all_true_y: 
        A np.array instance. In this project, all the real function value of 
        x_range is put here;
    all_ptd_y:
        A list instance contianing np.array instance. Each of the np.array
        corresponds to a model's prediction over x_range; 
    zlabel:
        A string instance to indicate the label of z-axis;
    predicted_point:
        A np.array instance in 1x2 as the solver's prediction for x_next
    legend:
        A list instances containing strings to indicate the legends in the
        plot; Default in False, and it will stop the plt.legend().
    plot_size:
        A list/tuple instance to control the size of the figure;
    title:
        A string object to set the tile of the figure. None if it is False;
    save_name:
        A string object to set the tile when saving the figure. The plot will
        not be saved if it is False;
    show:
        A boolean to control whether show the figure or not when running the
        code. False will not show the plot. When save_name is set in a string
        and show set in False, the desired plot will be saved but not showed.

    Returns:
        The return will retrun None no return is set.
    """
    xlabel, ylabel = 'x1', 'x2'
    ax = plt3d.Axes3D(plt.figure(figsize=plot_size))
    marker = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h',
              'H', 'D', 'd', 'P', 'X')
    if isinstance(all_true_y, list):
        ax.scatter(all_x[:, 0], all_x[:, 1], all_true_y[0], alpha=0.03,
                   label="true_value")
        ax.scatter(all_x[:, 0], all_x[:, 1], all_true_y[1], alpha=0.03,
                   label="noised_true_value")
    elif not isinstance(all_true_y, bool):
        ax.scatter(all_x[:, 0], all_x[:, 1], all_true_y, alpha=0.03,
                   label="true_value")
    if not isinstance(all_ptd_y, bool):
        for k, predicted_y_in_x_range in enumerate(all_ptd_y):
            ax.scatter(all_x[:, 0], all_x[:, 1], predicted_y_in_x_range,
                       marker=marker[k % 15], alpha=0.5, label=legend[k])
    ax.scatter(x[:, 0], x[:, 1], y,
               label=''.join([str(len(x)), '_training_pionts']))
    for i in range(len(x)):
        ax.text(x[i][0], x[i][1], y[i], '%s' % (str(i + 1)), size=30, zorder=1,
                color='k')
    if not isinstance(predicted_point, bool):
        pred_point = np.squeeze(predicted_point)
        x_limit, y_limit, z_limit = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        ax.scatter(pred_point[0], pred_point[1], z_limit[0], marker='X')
        ax.scatter(pred_point[0], pred_point[1], z_limit[1], marker='X')
        ax.plot([pred_point[0], pred_point[0]], [pred_point[1], pred_point[1]],
                [z_limit[0], z_limit[1]],
                label=''.join(['pred_x_next=', str(np.round(pred_point, 2))]))
        ax.plot([x_limit[0], x_limit[1]], [pred_point[1],
                 pred_point[1]], [z_limit[0], z_limit[0]])
        ax.plot([pred_point[0], pred_point[0]],
                [y_limit[0], y_limit[1]], [z_limit[0], z_limit[0]])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title: plt.title(title, y=1.01)
    if not isinstance(legend, bool):
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    if save_name: plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

    if (not isinstance(all_ptd_y, bool)) and len(all_ptd_y) == 1:
        num = int(np.sqrt(len(all_x)))
        X, Y = np.meshgrid(all_x[:num, 0], all_x[:num, 0])
        plt.figure(figsize=plot_size)
        surf = plt.contourf(X, Y, all_ptd_y[0].reshape(num, -1), cmap=cm.coolwarm,
                            antialiased=False)
        plt.colorbar(surf, shrink=0.5, aspect=5)
        if not isinstance(predicted_point, bool):
            pdt_point = np.squeeze(predicted_point)
        plt.scatter(pdt_point[0], pdt_point[1], marker='X', c='black')
        plt.annotate('pred_x_next', xy=(pdt_point[0], pdt_point[1]),
                                xytext=(pdt_point[0], pdt_point[1]))
        plt.title('The pred_x_next in mean_y_prediction contour graph')
        time_stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
        plt.savefig(''.join([time_stamp, '_', 'mean_y_prediction_contour_graph',
                                '_', save_name.split('-')[1]]))
        if not show:
            plt.close()

def plotting_3D_y_in_2D(x, y, all_x=False, all_true_y=False, all_ptd_y=False,
                        predicted_point=False, legend=False, plot_size=(12, 9),
                        title=False, save_name=False, show=True):
    """Plot 3D graphs with x in 1D and y in 2D during the solver processing data.

    Args:
    x: 
        A np.array instance. In this project, x_train is put here;
    y: 
        A np.array instance. In this project, y_train is put here;
    all_x:
        A np.array instance. In this project, x_range is put here;
    all_true_y: 
        A np.array instance. In this project, all the real function value of 
        x_range is put here;
    all_ptd_y:
        A list instance containing np.array instance. Each of the np.array
        corresponds to a model's prediction over x_range;
    predicted_point:
        A np.array instance in 1x1 as the solver's prediction for x_next
    legend:
        A list instances containing strings to indicate the legends in the
        plot; Default in False, and it will stop the plt.legend().
    plot_size:
        A list/tuple instance to control the size of the figure;
    title:
        A string object to set the tile of the figure. None if it is False;
    save_name:
        A string object to set the tile when saving the figure. The plot will
        not be saved if it is False;
    show:
        A boolean to control whether show the figure or not when running the
        code. False will not show the plot. When save_name is set in a string
        and show set in False, the desired plot will be saved but not showed.

    Returns:
        The return will return None no return is set.
    """
    xlabel, ylabel, zlabel = 'y1', 'y2', 'x'
    ax = plt3d.Axes3D(plt.figure(figsize=plot_size))
    marker = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*',
              'h', 'H', 'D', 'd', 'P','X')
    if isinstance(all_true_y, list):
        ax.scatter(all_true_y[0][:, 0], all_true_y[0][:, 1],
                   all_x, alpha=0.03, label="true_value")
        ax.scatter(all_true_y[1][:, 0], all_true_y[1][:, 1],
                   all_x, alpha=0.03, label="noised_true_value")
    elif not isinstance(all_true_y, bool):
        ax.scatter(all_true_y[:, 0], all_true_y[:, 1],
                   all_x, alpha=0.03, label="true_value")
    if not isinstance(all_ptd_y, bool):
        for k, predicted_y_in_x_range in enumerate(all_ptd_y):
            ax.scatter(predicted_y_in_x_range[:, 0],
                       predicted_y_in_x_range[:, 1],
                       all_x, marker=marker[k % 15], alpha=0.5, label=legend[k])
    ax.scatter(y[:, 0], y[:, 1], x,
               label=''.join([str(len(x)), '_training_pionts']))
    for i in range(len(x)):
        ax.text(y[i][0], y[i][1], x[i], '%s' % (str(i + 1)),
                alpha=1, size=30, zorder=1, color='k')
    if not isinstance(predicted_point, bool):
        y0_min, y0_max = np.min(all_true_y[:, 0]), np.max(all_true_y[:, 0])
        y1_min, y1_max = np.min(all_true_y[:, 1]), np.max(all_true_y[:, 1])
        y1 = np.linspace(y0_min, y0_max, 50) #* 1.1
        y2 = np.linspace(y1_min, y1_max, 50) #* 1.1
        y1, y2 = np.meshgrid(y1, y2)
        y1, y2 = y1.ravel(), y2.ravel()
        z = predicted_point * np.ones(len(y1))
        legend.append(''.join(
            ['pred_x_next=', str(np.squeeze(predicted_point))]))
        ax.scatter(y1, y2, z, alpha=0.3, label=''.join(
            ['pred_x_next=', str(np.squeeze(np.round(predicted_point, 2)))]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title:
        plt.title(title, y=1.01)
    if not isinstance(legend, bool):
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    if save_name: plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plotting_3D_stem_contourf(x, y, zlabel='y', predicted_point=False,
                              legend=False, plot_size=(12, 9), title=False,
                              save_name=False, show=True):
    """Plot 3D graphs with x in 2D and y in 1D during the solver processing data.

    This function is for plotting Expectation Improvement with stem to indicate 
    x_next and the contourf graph with x in 2D and y in 1D.

    Args:
    x: 
        A np.array instance. In this project, x_train is put here;
    y: 
        A np.array instance. In this project, y_train is put here;
    zlabel:
        A string instance to indicate the label of z-axis;
    predicted_point:
        Default in False. In the solver it is a list instance containing the
        predicted x_next and the true y value for all points in x_range to 
        plot the contourf graph.
    legend:
        A list instances containing strings to indicate the legends in the
        plot; Default in False, and it will stop the plt.legend().
    plot_size:
        A list/tuple instance to control the size of the figure;
    title:
        A string object to set the tile of the figure. None if it is False;
    save_name:
        A string object to set the tile when saving the figure. The plot will
        not be saved if it is False;
    show:
        A boolean to control whether show the figure or not when running the
        code. False will not show the plot. When save_name is set in a string
        and show set in False, the desired plot will be saved but not showed.

    Returns:
        The return will return None no return is set.
    """
    xlabel, ylabel = 'x1', 'x2'
    if isinstance(predicted_point, bool):
        ax = plt3d.Axes3D(plt.figure(figsize=plot_size))
        ax.scatter(x[:, 0], x[:, 1], y, label=legend[0], alpha=0.1)
    elif isinstance(predicted_point, list):
        all_y = predicted_point[1]
        predicted_point = np.squeeze(predicted_point[0])
        if not isinstance(all_y, bool):
            num = int(np.sqrt(len(x)))
            X, Y = np.meshgrid(x[:num, 0], x[:num, 0])
            plt.figure(figsize=plot_size)
            surf = plt.contourf(X, Y, all_y.reshape(num, -1), cmap=cm.coolwarm,
                                antialiased=False)
            plt.colorbar(surf, shrink=0.5, aspect=5)
            plt.scatter(predicted_point[0], predicted_point[1], marker='X',
                        c='black')
            plt.annotate('pred_x_next',
                         xy=(predicted_point[0], predicted_point[1]),
                         xytext=(predicted_point[0], predicted_point[1]))
            plt.title('The pred_x_next in all true y contour graph')
            time_stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
            plt.savefig(''.join([time_stamp, '_', 'all_true_y_contour_graph',
                                 '_', save_name.split('-')[1]]))
            if not show: plt.close()

            plt.figure(figsize=plot_size)
            surf1 = plt.contourf(X, Y, y.reshape(num, -1), cmap=cm.coolwarm,
                                antialiased=False)
            plt.colorbar(surf1, shrink=0.5, aspect=5)
            plt.scatter(predicted_point[0], predicted_point[1], marker='X',
                        c='black')
            plt.annotate('pred_x_next',
                         xy=(predicted_point[0], predicted_point[1]),
                         xytext=(predicted_point[0], predicted_point[1]))
            plt.title('The pred_x_next in EI contour graph')
            time_stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
            plt.savefig(''.join([time_stamp, '_', 'all_EI_contour_graph',
                                 '_', save_name.split('-')[1]]))
            if not show: plt.close()

        ax = plt3d.Axes3D(plt.figure(figsize=plot_size))
        ax.scatter(x[:, 0], x[:, 1], y, label=legend[0], alpha=0.1)
        x_limit, y_limit, z_limit = ax.get_xlim(), ax.get_ylim(), ax.get_zlim()
        ax.scatter(predicted_point[0], predicted_point[1], z_limit[0],
                   marker='X')
        ax.scatter(predicted_point[0], predicted_point[1],
                   z_limit[1], marker='X')
        ax.plot([predicted_point[0], predicted_point[0]], [predicted_point[1],
                 predicted_point[1]], [z_limit[0], z_limit[1]], label=''.join(
                ['pred_x_next=', str(np.round(predicted_point, 2))]))
        ax.plot([x_limit[0], x_limit[1]],
                [predicted_point[1], predicted_point[1]],
                [z_limit[0], z_limit[0]])
        ax.plot([predicted_point[0], predicted_point[0]],
                [y_limit[0], y_limit[1]],
                [z_limit[0], z_limit[0]])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    if title: plt.title(title, y=1.01)
    if not isinstance(legend, bool):
        plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    if save_name: plt.savefig(save_name, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def plotting(x, y, all_x=False, all_true_y=False, all_ptd_y=False, xlabel='x',
             ylabel='y', zlabel='y', predicted_point=False, legend=False,
             plot_size=(12, 9), title=False, save_name=False, show=True,
             dimension=False):
    """Plot during the solver processing data in different situations.

    Args:
    x: 
        A np.array instance. In this project, x_train is put here;
    y: 
        A np.array instance. In this project, y_train is put here;
    all_x:
        A np.array instance. In this project, x_range is put here;
    all_true_y: 
        A np.array instance. In this project, all the real function value of 
        x_range is put here;
    all_ptd_y:
        A list instance containing np.array instance. Each of the np.array
        corresponds to a model's prediction over x_range; 
    xlabel:
        A string instance to indicate the label of x-axis; 
    ylabel:
        A string instance to indicate the label of y-axis;
    zlabel:
        A string instance to indicate the label of z-axis;
    predicted_point:
        Can be a np.array instance in 1x1 or a list containing all the true
        value of y and the solver's prediction for x_next. Default in False.
    legend:
        A list instances containing strings to indicate the legends in the
        plot; Default in False, and it will stop the plt.legend().
    plot_size:
        A list/tuple instance to control the size of the figure;
    title:
        A string object to set the tile of the figure. None if it is False;
    save_name:
        A string object to set the tile when saving the figure. The plot will
        not be saved if it is False;
    show:
        A boolean to control whether show the figure or not when running the
        code. False will not show the plot. When save_name is set in a string
        and show set in False, the desired plot will be saved but not showed.
    dimension:
        An integer indicating the plotting type. If it is 2, then plotting 2D;
        If it is 3, plotting 3D; If it is over 3, no plotting; Default in False.
    """
    if dimension == 2:
        plotting_2D(x, y, all_x, all_true_y, all_ptd_y, xlabel, ylabel,
                    predicted_point, legend, plot_size, title, save_name, show)
    elif dimension == 3:
        if not (isinstance(all_x, bool) or isinstance(all_true_y, bool)):
            # y=f(x1, x2)
            if x.size > y.size:
                plotting_3D_x_in_2D(x, y, all_x, all_true_y, all_ptd_y, zlabel,
                                    predicted_point, legend, plot_size, title,
                                    save_name, show)
            # y1, y2=f(x)
            else:
                plotting_3D_y_in_2D(x, y, all_x, all_true_y, all_ptd_y,
                                    predicted_point, legend, plot_size,
                                    title, save_name, show)
        else:
            plotting_3D_stem_contourf(x, y, zlabel, predicted_point, legend,
                                      plot_size, title, save_name, show)
    elif dimension > 3:
        pass # print('Dimension is too high to plot!')

def indices_by_kmeans_process(x_range, x_train, plot_1D_data, fsize=(12, 6.75), 
                              show_plot=True):
    """Find out indices of initial or next traning point(s) for fitting models.

    Args:
    x_range:
        It should be a np.array instance containing all the points desired
        to be investigated and evaluated for prediction x_next by QBC.
    x_train:
        It can be both a positive integer or a np.array instance. When it is an
        int, then x_range will be divided into the assigned number of clusters,
        and the indices of points in x_range closest to the centroids will be
        returned. When it is an np.array, then x_range will be divided into
        len(x_train)+1 clusters, and the indices of points in x_range belonging
        to the cluster containing no training point while has the most points
        will be returned. 
    plot_1D_data:
        A list object should contain 2 elements. The first one is the y values
        for all x in x_range, the second one is the y values only for x_train

    Returns:
        A list object containing indices from x_range.
    """
    # Determine the number of clusters based on the type of x_train
    if isinstance(x_train, int):
        n_clusters = x_train
    else:
        n_clusters = len(x_train) + 1

    # Use sklearn.cluster.Kmeans to do the clustering
    kmeans_process = KMeans(n_clusters=n_clusters, random_state=9)
    clustered_all = kmeans_process.fit_predict(x_range.reshape(len(x_range), -1))
    # Get the information of all the centroids and labels
    centroid = kmeans_process.cluster_centers_
    labels = kmeans_process.labels_

    # Find out the points and their indices closest to the centroids
    points_closest_centroid = []
    points_closest_centroid_id = []
    for i in range(n_clusters):
        dis = np.zeros(len(x_range))
        for n, j in enumerate(x_range):
            dis[n] = np.linalg.norm(j-centroid[i])
        # print(np.argmin(dis))
        points_closest_centroid.append(x_range[np.argmin(dis)])
        points_closest_centroid_id.append(np.argmin(dis))
    
    # Create a intermediate dictionary to summarize the clustering results:
    # for each label, the index of point in x_range belongs to this label
    info_dic = {}
    for label in labels:
        info_dic[str(label)] = []
    for i, label in enumerate(clustered_all):
        info_dic[str(label)].append(i)

    # Find out the labels for all the points in x_train if it is an np.array
    if not isinstance(x_train, int):
        # Iterate in x_range and x_train to find out the corresponding label
        clustered_x_train = [clustered_all[i] for i, j in enumerate(x_range)\
                                    for point in x_train if (j == point).all()]
        # Find out clusters containing no training points
        cluster_without_trn_data = list(set(labels)-set(clustered_x_train))
        # Find out how many points the clusters without training points have
        nr_of_data = [len(info_dic[str(c)]) for c in cluster_without_trn_data]
        # Find out the indices of the biggest clusters have no training points 
        # and the point clostest to its centroid
        cluster_id = cluster_without_trn_data[nr_of_data.index(max(nr_of_data))]
        x_next_close_centroid = points_closest_centroid[cluster_id].reshape(1,-1)

        # Find out the indices of points in the biggest clusters have no 
        # training points and a random point from this cluster
        all_indices_in_cluster_id = info_dic[str(cluster_id)]
        rand_int = np.random.randint(len(all_indices_in_cluster_id))
        x_next_id = info_dic[str(cluster_id)][rand_int]
        x_next_random_in_cluster = x_range[x_next_id].reshape(1, -1)

    # Print figures for more information
    # Find out the dimension of a singe x, only 1D and 2D case can be plotted
    x_d = int(x_range.size / len(x_range))
    if x_d == 1:
        plt.figure(figsize=fsize)
        # Scatter the function plot, here the y is noise free
        plt.scatter(x_range, plot_1D_data[0], c=clustered_all, label='all_data')
        # Scatter the training points, here y_train is also noise free
        if not isinstance(x_train, int): 
            plt.scatter(x_train, plot_1D_data[1], marker='X',
                        color='red', label='training_points')
        # Scatter the points closest to the centroid
        plt.scatter(points_closest_centroid, np.zeros_like(centroid),
                    color='blue', label='points_closest_centroids')
        # Annotate the point
        for i in range(n_clusters):
            text = str(i)
            if (not isinstance(x_train, int)) and i == cluster_id:
                text = 'Selected'
            plt.annotate(text, xy=(centroid[i], 0), xytext=(centroid[i], 0))
        plt.grid()
    elif x_d == 2:
        plt.figure(figsize=fsize)
        # Plot the contour graph of x_range
        plt.scatter(x_range[:,0], x_range[:,1], c=clustered_all, label='all_data')
        # Scatter the training points, here y_train is also noise free
        if not isinstance(x_train, int):
            plt.scatter(x_train[:,0], x_train[:,1], marker='X', 
                        color='red', label='training_points')
        # Scatter the points closest to the centroid
        points = np.array(points_closest_centroid).reshape(-1, 2)
        plt.scatter(points[:,0], points[:,1],
                    color='blue', label='points_closest_centroids')
        # Annotate the point
        for i in range(n_clusters):
            text = str(i)
            if (not isinstance(x_train, int)) and i == cluster_id:
                text = 'Selected'
            plt.annotate(text, xy=(centroid[i, 0], centroid[i, 1]), 
                         xytext=(centroid[i, 0], centroid[i, 1]))
    if x_d <= 2:
        # Show the legend and save the figure
        plt.legend()
        stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
        plt.savefig(''.join([stamp, '_', 'kmeans_process_for_indices']))
        if show_plot:
            plt.show()
        else:
            plt.close()

    # Return the indices according to x_train's type
    if isinstance(x_train, int):
        return points_closest_centroid_id
    else:
        return all_indices_in_cluster_id

def rolling_mean(dimension, y, n=4, h=0, w=0):
    """Find out all the rolling mean with the specified window size for EI.

    Args:
    dimension:
        The dimension information of a training point, can only be 1 or 2;
    y:
        The corressponding Expected Improvement for all the x. 
    n:
        The rolling window size. For x in 2D, a square window will be used.
    h:
        The heigh of y np.array if x in 2D, and it should be assigned if y is
        not flattened.
    w:
        The width of y np.array if x in 2D, and it should be assigned if y is
        not flattened.

    Returns:
        A np.array with in the same shape of y contining the rolling mean.
    """
    # Gain the shape information of array y
    if len(y.shape) == 2:
        a, b = y.shape
    elif len(y.shape) == 1:
        a, b = y.shape[0], 1

    # Deal with the situation of a single x in 1D
    if dimension == 1:
        # Reshape y in case of its original shape in (a,) to pass it to the
        # extended new array as a whole
        y = y.reshape(-1, b)
        # Create the extended array according to the window size and then put 
        # the original array in the middle
        extend_y = np.zeros((a+n-1, b))
        s = int(n/2) # start_poisition
        extend_y[s-1: a+s-1] = y

        # Calculate the accumulate sum of the extended array
        acc_e_y = np.cumsum(extend_y, dtype=float)
        # Calculate the sum of elements within the 'window'.
        # This line needs more attention since it is hard to understand.
        # Referenced from https://stackoverflow.com/questions/14313510/how-to-calculate-rolling-moving-average-using-python-numpy-scipy
        acc_e_y[n:] = acc_e_y[n:] - acc_e_y[:-n]
        # print('extend_y',np.squeeze(extend_y))
        # print('acc_e_y', acc_e_y)
        return acc_e_y[n-1:] /n

    if dimension == 2:
        # Find out the height and width information of the array y
        # If y is flatten while in 2D situation, then y will be treated
        # in a square shape
        if not (h and w):
            h  = int(np.sqrt(len(y)))
            w = h

        # Create the extended array according to the window size and then put 
        # the original array in the middle
        extend_y = np.zeros((w+n, h+n))
        y = y.reshape(h, w)
        # print('copy_y', copy_y)
        # print('extend_y.shape', extend_y.shape)
        # print(print('copy_y[1,2]', copy_y[1,2]))
        s = int(n/2) # start_poisition
        extend_y[s:h+s, s:w+s] = y
        # print('s',s)
        # print('extend_y', extend_y)
        
        # Find out the sum of the elements within the 'window'
        new_y = np.zeros_like(y)
        for i in range(n,h+n):
            for j in range(n, w+n):
                # print('np.sum(extend_y[i-s:i, j-s:j])', np.sum(extend_y[i-n:i, j-n:j]))
                new_y[i-n,j-n] = np.sum(extend_y[i-n:i, j-n:j])
        # print('new_y', new_y)
        # Calculate the mean and reshape it as the shape of y
        return (new_y/(n*n)).reshape(a, b)

class MyCallback(tf.keras.callbacks.Callback):
    """The class customizes the output when the Neral Network stops training.

    This class is inherited from tf.keras.callbacks.Callback, and is used during
    training the Keras neural network model with setting a threshold.

    Attributes:
    loss_threshold:
        A decimal indicate the baseline of the training loss.
    x_train: 
        It a np.array instance containing all the training data points in x
        for the models in QBC when using fit method. 
    y_train: 
        It a np.array instance containing all the training data points in y
        for the models in QBC when using fit method. 
    max_allowed_pred_error:
        An number a to indicate an interval from [-a, a], which to tell whether
        the gap between the predictions and the real values fall in this area.
        If 95% of the gaps fall in this area, then stop training the model.
    no_of_epochs: 
        An integer for the maximum training epochs.
    method:
        The strategy of stop training the model. The default one is using
        threshold and the other one is using the interval method.
    training_time_policy:
        A training time constraint for fitting the model. An decimal scaled in
        minute will activate the policy, and the fitting process will stop after
        the assigned time.
    """

    def __init__(self, loss_threshold, x_train, y_train, max_allowed_pred_error,
                 no_of_epochs=100000, method='threshold',
                 training_time_policy=False):
        """Inits MyCallback Class with 5 attributes"""
        super(MyCallback, self).__init__()
        self.threshold = loss_threshold
        self.no_of_epochs = no_of_epochs
        self.x_train = x_train
        self.y_train = y_train
        self.max_allowed_pred_error = max_allowed_pred_error
        self.method = method
        self.training_time_policy = training_time_policy

    def evaluation_by_interval(self, y_predictions):
        """Evaluate whether fulfill the training stop criteria.

        Args:
        y_predictions:
            The model's predictions over x_train 

        Returns:
        Return True if criteria is satisfied otherwise False. 
        """
        gaps = np.abs(self.y_train - y_predictions)
        marks = self.max_allowed_pred_error * np.ones_like(self.y_train)
        # Use <= to get a masked np.array, then use np.sum to count how many
        # predictions are less or equal to the max_allowed_pred_error
        all_flags = gaps <= marks
        if np.sum(all_flags) / len(marks) == 1:
            return True

    def on_epoch_end(self, epoch, logs=None):
        """Print customized information at the end of training epoch.

        Args:
        epoch:
            An integer indicate the current number of training epoch.
        logs:
            A dictionary containing the training loss information.
        """
        loss = logs["loss"]
        # Check whether the stop fitting policy is the max_allowed_pred_error
        if self.method.lower() == 'interval':
            y_predictions = self.model.predict(self.x_train)
            if self.evaluation_by_interval(y_predictions):
                print('Terminated by loss-in-interval constrain in epoch {} '
                      'with MSE loss in {}'.format(epoch + 1, loss), end=' ')
                self.model.stop_training = True
        # Check whether the stop fitting policy is the loss_threshold
        elif self.method.lower() == 'threshold':
            if loss <= self.threshold:
                print('Terminated normally in epoch {} with MSE'
                      ' loss in {}'.format(epoch + 1, loss), end=' ')
                self.model.stop_training = True
         # Check whether the stop fitting policy is the number of epochs
        if epoch == self.no_of_epochs - 1:
            print('Terminated by the constrain of training epochs in epoch {}'
                  ' with MSE loss in {}'.format(epoch + 1, loss), end=' ')
        # Check whether the stop fitting policy is the training time
        if not isinstance(self.training_time_policy, bool):
            time_cost = time.time() - self.training_time_policy[0]
            training_duration = self.training_time_policy[1]
            if time_cost > training_duration:
                print('Terminated by training time constrain in epoch {}'
                      ' with MSE loss in {}'.format(
                        epoch + 1, loss, round(time_cost, 2)), end=' ')
                self.model.stop_training = True


class ActiveLearningSolver:
    """This class is for the designed active learning solver.

    The methods of the solver includes defining a single model, creating a
    query based committee (QBC), fitting the models in QBC, predicting x_next
    in one step or multiple steps, predicting by random guess, predicting by
    Gaussian process, loading and saving models in QBC.

    Attributes:
    ensemble_model_list:
        it is initialized with an empty list, and will put the model in the list
        as a member of QBC when using add_model method.
    all_model_parameters:
        it is initialized with an empty list, and will contain lists with each
        having all the required parameters to define a model in QBC when using
        add_model method.
    fit_parameters:
        it is initialized with an empty list, and will contain lists with each
        having all the required parameters to train/fit a model in QBC when
        using fit method.
    predict_parameters:
        it is initialized with an empty list, and will store all the input
        parameters of predict method.
    x_dimension:
        it is initialized in None, and will be an integer indicating the
        dimension information of a single element in x_train or x_range.
    y_dimension:
        it is initialized in None, and will be an integer indicating the
        dimension information of a single element in y_train.
    x_train:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in x for predict_by_QBC when using fit method.
    y_train:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in y for predict_by_QBC when using fit method.
    x_train_PSO:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in x for predict_by_PSO when using fit method.
    y_train_PSO:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in y for predict_by_PSO when using fit method.
    x_train_GP:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in x for predict_by_GP.
    y_train_GP:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in y for predict_by_GP.
    x_train_rand:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in x for predict_by_random_guess.
    y_train_rand:
        Initialized in an empty list and will be a np.array instance containing 
        all the training data in y for predict_by_random_guess.
    x_range:
        It is initialized in None and will be a np.array instance containing all
        the points desired to be investigated and evaluated for prediction in x
        when using the predict method.
    y_true_in_x_range:
        It is initialized in None and will be a np.array instance containing all
        the points desired to be investigated and evaluated for prediction in y
        when using the predict method.
    y_min:
        It is initialized in None and will be a np.array instance in 1x1 Or 1xN
        shape as a single element from y_train. For 1 dimension, it is the
        minimum of y_train; For multiple dimensions, it is the element in
        y_train with the minimum L2 distance from origin.
    pred_by_QBC:
        it is initialized with an empty list, and will store list/tuple
        containing all the predicted x_next and the corresponding y prediction
        value for each single predict step when using QBC as the solver.
    pred_by_random:
        it is initialized with an empty list, and will store list/tuple
        containing all the predicted x_next for each single predict step when
        using random guess solver.
    pred_by_GP:
        it is initialized with an empty list, and will store list/tuple
        containing all the predicted x_next for each single predict step when
        using Gaussian Process solver.
    pred_by_PSO:
        it is initialized with an empty list, and will store list/tuple
        containing all the predicted x_next for each single predict step when
        using PSO solver.

    """

    def __init__(self):
        """Inits the solver with 16 attributes for different methods"""
        self.model_list = []
        self.model_list_PSO = []
        self.model_types = []
        self.define_parameters = []
        self.fit_parameters = {}
        self.predict_parameters = []
        self.x_dimension = None
        self.y_dimension = None

        self.tree_forest_parameters = []
        self.tree_forest_models = []

        self.x_train = []
        self.y_train = []
        self.x_train_PSO = []
        self.y_train_PSO = []
        self.x_train_GP = []
        self.y_train_GP = []
        self.x_train_rand = []
        self.y_train_rand = []

        self.x_range = None
        self.y_range = None
        self.y_min = None
        self.y_max = None
        self.search_space = None

        self.pred_by_QBC = []
        self.pred_by_random = []
        self.pred_by_GP = []
        self.pred_by_PSO = []

    def define_a_keras_nn(self, nn_struc_para_list, activation_func_list,
                              l2_penalty, bias, learning_rate=0.01,
                              loss="MeanSquaredError", optimizer="adam"):
        """Define a single model for the QBC.

        Only Neural Network from keras.models.Sequential() can be defined here.
        
        Args:
        nn_struc_para_list:
            A list containing integers to indicate the neural network
            architecture information. For example [2, 50, 40, 30, 3] means that
            there are 2 input variables in the input layer, 3 outputs in the
            output layer, and three hidden layers with containing 50, 40, 30
            neurons respectively.
        activation_func_list:
            A list containing strings to indicate the neural network activation
            function information for each layer. For example ["Relu", "Sigmoid",
            "tanh", "linear"] means that there the activation functions from the
            1st layer to the one before the output layer are "Relu", "Sigmoid",
            "tanh" and "linear" functions. This list's length is 1 less than
            that in nn_struc_para_list, and only a valid activation function
            for keras.models.Sequential() model can be the element of the list.
        l2_penalty:
            It is the L2_penalty parameter for keras.models.Sequential() model.
            It can be a single or a list containing positive floating number for
            each layer of the neural network model.
        bias:
            It is the bias parameters for a keras.models.Sequential() model.
            It can be a single or a list containing positive floating number for
            each layer of the neural network model.
        learning_rate:
            It is the learning rate of a keras.models.Sequential() model.
            Default in 0.01.
        loss:
            It is the strategy for the loss of a keras.models.Sequential()
            model. Default in MeanSquaredError.
        optimizer:
            It is the chosen strategy for the optimizer of a
            keras.models.Sequential() model. Default in adam.

        Return:
        An tuple containing the defined model and the input parameters for 
        defining the model.
        """
        # Prepare the l2_penalty_list and bias_list for the necessary layer of 
        # the neural network;
        if isinstance(l2_penalty, float):
            penalty_list = [l2_penalty for i in range(len(nn_struc_para_list)-1)]
            l2_penalty = penalty_list
        if isinstance(bias, bool):
            bias_list = [bias for i in range(len(nn_struc_para_list) - 1)]
            bias = bias_list

        # Generate the random seed for initializing model parameters
        r = int(str(time.time()).split('.')[1]) + np.random.randint(0, 100, 1)
        np.random.seed(seed=r)
        tf.random.set_seed(seed=r)

        # Define a single neural network model with keras.models.Sequential()
        # Then add the input layer of the model;
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(
            nn_struc_para_list[1], input_dim=nn_struc_para_list[0],
            activation=activation_func_list[0], use_bias=bias[0],
            kernel_regularizer=keras.regularizers.l2(l2_penalty[0])))

        # Tell the number_of_hidden_layers whether it is over 2
        if (len(nn_struc_para_list) - 2) > 1:
            for i in range(2, len(nn_struc_para_list) - 1):
                model.add(keras.layers.Dense(
                    nn_struc_para_list[i], use_bias=bias[i - 1],
                    kernel_regularizer=keras.regularizers.l2(l2_penalty[i - 1]),
                    activation=activation_func_list[i]))

        # Add the output layer of the model
        model.add(keras.layers.Dense(
            nn_struc_para_list[-1], activation=activation_func_list[-1],
            kernel_regularizer=keras.regularizers.l2(l2_penalty[-1]),
            use_bias=bias[-1]))

        # Specify the loss function, optimizer and learning rate
        model.compile(loss=loss, optimizer=optimizer)
        model.optimizer.learning_rate.assign(learning_rate)

        return model

    def define_a_model(self, x_d, y_d, model_type='define_a_keras_nn', **kargs):
        """A RandomForestRegressor model from sklearn for QBC.

        Args:
        kargs:
            parameters legal and valid for defining the assigned model.

        Return:
        A model and its defining parameters.
        """
        self.x_dimension = x_d
        self.y_dimension = y_d

        # Construct the model with the assigned type the its parameter
        if model_type == 'define_a_keras_nn' and kargs:
            model_type =  self.define_a_keras_nn
        
        if model_type != 'define_a_keras_nn': 
            model = model_type(**kargs)
        else:
            model = None

        return model, model_type, kargs
    
    def initialize_n_nn_models(self, l2_penalty, bias=True, n=10, fill_style='1',
                               x_dimension=1, y_dimension=1, learning_rate=0.01,
                               activation_func_list=['relu', 'linear'],
                               loss="MeanSquaredError", optimizer="adam"):
        """Initialize n Neural Network models as the QBC.
        
        Args:
        l2_penalty:
            It is the L2_penalty parameter for keras.models.Sequential() model.
            It can be a single or a list containing positive floating number for
            each layer of the neural network model.
        bias:
            It is the bias parameters for a keras.models.Sequential() model.
            It can be a single or a list containing positive floating number for
            each layer of the neural network model.
        n:
            The assigned number of models for the QBC, and the models in the QBC
            will contain hidden layers from 1 to n;
        fill_style:
            It controlls the number of neurons of each hidden layer when
            generate the NN models' architecture lists. If it is '1', then the 
            initialized number of neurons for each hidden layer will be 1; If 
            it is in 'diversity', then the initialized number of neurons for
            each hidden layer will be as the number of hidden layers the model
            containing; It is able to be assigned as a function, which will use
            the number of hidden layers as input parameters to have a number
            as the number of neurons for each hidden layer in the model.
        x_dimension:
            The number of variables for the input layer of the NN model.
        y_dimension:
            The number of variables for the output layer of the NN model. 
        learning_rate:
            It is the learning rate of a keras.models.Sequential() model.
            Default in 0.01.      
        activation_func_list:
            A list contains 2 strings to indicate the neural network activation
            function information. For example ["Relu", "linear"] means that 
            there the activation functions from the 1st layer to the one before
            the last layer are all in "Relu", and the output layer will have
            "linear" as its activation functions. Only a valid activation 
            function for keras.models.Sequential() model can be the element
            of this parameter.
        loss:
            It is the strategy for the loss of a keras.models.Sequential()
            model. Default in MeanSquaredError.
        optimizer:
            It is the chosen strategy for the optimizer of a
            keras.models.Sequential() model. Default in adam.

        Return:
        The updated self.model_list to show all the models in the QBC,
        and the self.define_parameters containing all the parameters for
        defining the NN models.
        """
        # Define intermediate list to containing NN models' architecture and 
        # activation function list
        all_model_structures, all_activation_functions = [], []
        # Loop to build the assigned number of models in the QBC
        for nr_hidden_layer in range(1, n+1):
            # Fill in the number of neurons into each hidden layer by the 
            # designed style for a model
            if fill_style == '1':
                a_model = [1] * (nr_hidden_layer + 2)
            elif fill_style == 'diversity':
                a_model = [nr_hidden_layer] * (nr_hidden_layer + 2)
            elif callable(fill_style):
                a_model = [fill_style(nr_hidden_layer)] * (nr_hidden_layer + 2)
            # Update the architecture list with the information for the input
            # layer and output layer, then append this model to the model list.
            a_model[0] = x_dimension
            a_model[-1] = y_dimension
            all_model_structures.append(a_model)
            # Create the activation function list for the corresponding NN model
            act_func_for_a_model = []
            for i in range(nr_hidden_layer + 1):
                if i != nr_hidden_layer:
                    act_func_for_a_model.append(activation_func_list[0])
                else:
                    act_func_for_a_model.append(activation_func_list[1])
            all_activation_functions.append(act_func_for_a_model)
        # Loop to use the defined self.define_a_single_model to create a model
        # by an architecture and an activation function list, and then use 
        # self.add_model_to_QBC to put the model into the QBC
        for i, a_model_arch in enumerate(all_model_structures):
            model, model_type, parameter = self.define_a_model(x_d=x_dimension, 
                y_d=y_dimension, nn_struc_para_list=a_model_arch, 
                activation_func_list=all_activation_functions[i], 
                l2_penalty=l2_penalty, learning_rate=learning_rate,
                bias=bias,loss=loss, optimizer=optimizer)
            self.add_model_to_QBC(model, model_type, parameter)

        return self.model_list, self.model_types, self.define_parameters

    def add_model_to_QBC(self, model, model_type, parameter=None, number=1):
        """Add a model or models into the QBC.
        
        Args:
        model:
            The model for the member of QBC.
        parameter:
            The parameters for defining the model.
        number:
            Default in 1. If it is a positive integer larger than 1, then the 
            assigned number of the model and parameter will be put into the QBC.

        Return:
        The updated attribute of this class to show all the models in the QBC,
        and all the parameters of defining all these models.
        """

        # Put the model into self.model_list and the corresponding
        # parameters into self.define_parameters with the assigned times.
        for i in range(number):
            self.model_list.append(model)
            self.model_types.append(model_type)
            self.define_parameters.append(parameter)

        return self.model_list, self.define_parameters

    def delete_model_from_QBC(self, number_of_the_model):
        """delete a model from the QBC.
        
        Args:
        number_of_the_model:
            An integer less than the length of self.model_list
            attribute and equal or larger than 0. This number indicate the 
            position of the model, which is going to be deleted, in QBC.

        Return:
        The updated self.model_list to show all the models in the QBC,
        and all the parameters for defining the remaining models.
        """
        total = len(self.model_list)
        if not (0 <= number_of_the_model < total):
            print('The assigned number is illegal as the position of a model, '
                  'it is negative or larger than the number of the models.')
        else:
            # Use pop to delete the model and its defining parameters from
            # the related list.
            self.model_list.pop(number_of_the_model)
            self.define_parameters.pop(number_of_the_model)

        return self.model_list, self.define_parameters

    def gaussian_est(self, y_np_2d_array, num_points=0, threshold=0, region_t=5):
        """Use Gaussian as the estimated distribution of a single y.
        
        Args:
        y_np_2d_array:
            The prediction array, contains all the y values from all models for 
            every x in x_range.

        Return:
        A np.array instance containing the expected improvement in every y.
        """
        mean = np.mean(y_np_2d_array, axis=1)
        mean = mean.reshape(-1, self.y_dimension)
        std = np.std(y_np_2d_array, axis=1)
        std = std.reshape(-1, self.y_dimension)
        if num_points and threshold: # will not work for y in multiple dimensions
            p10_y = np.zeros([len(y_np_2d_array), 1])
            low = self.y_min - region_t * np.abs(self.y_max - self.y_min)
            high = self.y_max + region_t * np.abs(self.y_max - self.y_min)
            y_range = np.linspace(low, high, num_points)
            for i in range(len(y_np_2d_array)):
                mean_y, std_y = mean[i], std[i]
                if (std_y == 0).all():
                    p10_y[i] = 0 #np.zeros([1, self.y_dimension])
                else:
                    dist = lambda x: 1/(2*np.pi*std_y**2)**0.5 *\
                                     np.exp(-(x-mean_y)**2 / (2*std_y**2))
                    den_y = dist(y_range)
                    norm_den_y = den_y / np.sum(den_y)
                    accum_y =  np.cumsum(norm_den_y)
                    target_y = np.argmax(accum_y >= threshold)
                    p10_y[i] = y_range[target_y]
            ei_2d_array = self.y_min - p10_y
            # print('y_range_low, high, y_min', low, high, self.y_min)
            # print('den_y', den_y[:5])
            # print('norm_den_y', norm_den_y[:5])
            # print('accum_y', accum_y[:5])
            # print('p10_y', p10_y[:5])
            # print('ei_2d_array', ei_2d_array[:5])
        else:
            u_std = std*(std != 0) + 1 * (std == 0)
            ei = (self.y_min - mean) * st.norm.cdf((self.y_min - mean) / u_std)\
                + u_std * st.norm.pdf((self.y_min - mean) / u_std)
            ei_2d_array = ei * (std != 0) 
        return ei_2d_array

    def iqr_est(self, y_np_2d_array):

        # Calculate the mean and variance of y for each x,
        # and update the attributes.
        # y_np_2d_array = y_np_2d_array.reshape(-1, self.y_dimension)
        pred_y_mean = np.mean(y_np_2d_array, axis=1).reshape(-1, self.y_dimension)
        # pred_y_std = np.std(y_np_2d_array, axis=1).reshape(-1, self.y_dimension)
        # print('pred_y_mean.shape', pred_y_mean.shape, pred_y_mean)
        first_q = np.array(np.percentile(y_np_2d_array, 1, axis=1)).reshape(-1, self.y_dimension)
        # print('first_q.shape', first_q.shape, first_q)
        third_q = np.array(np.percentile(y_np_2d_array, 99, axis=1)).reshape(-1, self.y_dimension)
        # print('third_q.shape', third_q.shape, third_q)

        ei_2d_array = self.y_min - (pred_y_mean - 2 * (third_q - first_q))
        # print('self.y_min', self.y_min)
        # print('self.y_min - (pred_y_mean - 2 * (third_q - first_q))', self.y_min,'\
        #  - (', pred_y_mean, ' - 2 * (', third_q,' - ', first_q, '))' )
        # ei_2d_array = (self.y_min - (pred_y_mean * np.exp(third_q - first_q)))
        ei_2d_array = ei_2d_array * (ei_2d_array > 0)
        # print('ei_2d_array', ei_2d_array)
        return ei_2d_array

    def kernel_density_est(self, y_np_2d_array, num_points=0, threshold=0, 
                           region_t=5, auto_h=False):
        """Use kernel density to estimate the single y's distribution.
        
        Args:
        y_np_2d_array:
            The prediction array, contains all the y values from all models for 
            every x in x_range.

        Return:
        A np.array instance containing the expected improvement in every y.
        """
        # Determine the bandwidths of all the kernel density estimators
        # Using Silverman´s rule of thumb, num_rows:number of data points, 
        # num_colus:number of predictions for a single point;
        num_rows, num_colus = y_np_2d_array.shape[0], y_np_2d_array.shape[1]
        iqr = st.iqr(y_np_2d_array, axis=1).reshape(-1, self.y_dimension)

        # Element wise minimization
        pred_y_std = np.std(y_np_2d_array, axis=1).reshape(-1, self.y_dimension)
        tmp1_np_2d_array = np.minimum(pred_y_std, iqr / 1.34)
        # Silverman´s rule of thumb
        h_np_2d_array = 0.9 * tmp1_np_2d_array * num_colus ** (-0.2)
        
        if num_points and threshold: 
            # will not work for y in multiple dimensions
            p10_y = np.zeros([len(y_np_2d_array), 1])
            low = self.y_min - region_t * np.abs(self.y_max - self.y_min)
            high = self.y_max + region_t * np.abs(self.y_max - self.y_min)
            y_range = np.linspace(low, high, num_points)
            for t in range(len(y_np_2d_array)):
                pred_y = y_np_2d_array[t]
                # print('len(pred_y)',len(pred_y))
                if not auto_h:
                    h = h_np_2d_array[t]
                else:
                    if np.abs(h_np_2d_array[t]) <= 1:
                        w = 1
                    else:
                        w = np.abs(h_np_2d_array[t])
                    bandwidths = [g for g in 10 ** np.linspace(-1*w, w, 100)]
                    # print('bandwidths', bandwidths.shape, type(bandwidths), bandwidths[:5])
                    # print('W', w, bandwidths.shape)
                    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                                        {'bandwidth': bandwidths},
                                        cv=LeaveOneOut())
                    grid.fit(pred_y[:, None])
                    h = list(grid.best_params_.values())[0]
                    # print('h', h)
                if h == 0:
                    p10_y[t] = 0
                else:
                    f = lambda a, b: 1/(h*(2*np.pi)**0.5)*np.exp(-0.5*((a-b)/h)**2)
                    dist = lambda x: 1 / num_colus * np.sum(np.array([f(x, y)\
                                     for y in pred_y]).reshape(-1, num_colus,\
                                      self.y_dimension), axis=1)
                    den_y = dist(y_range)
                    norm_den_y = den_y / np.sum(den_y)
                    accum_y =  np.cumsum(norm_den_y)
                    target_y = np.argmax(accum_y >= threshold)
                    p10_y[t] = y_range[target_y]               
            ei_2d_array = self.y_min - p10_y
        else:
            # Calculate the expected improvement
            ei_2d_array = np.zeros([num_rows, self.y_dimension])
            for i in range(num_rows):
                ei_i = (self.y_min - y_np_2d_array[i]) * \
                    st.norm.cdf((self.y_min - y_np_2d_array[i]) / \
                                h_np_2d_array[i]) + h_np_2d_array[i] * \
                    st.norm.pdf((self.y_min - y_np_2d_array[i]) / \
                                h_np_2d_array[i])
                ei_2d_array[i] = np.mean(ei_i.reshape(-1, self.y_dimension),
                                            axis=0)
        return ei_2d_array

    def fit_keras_nn(self, model, i, no_of_epochs=5000, batch_size=100, verbose=0,
                   max_allowed_pred_error=0.1, termination_mse_threshold=0.2,
                   loss_plot_size=(12, 6.7), show_plot=True, callback=None,
                   training_time_limit=None, update_models_structure=False,
                   total_time=0.75):
        """Fit the models in QBC.
        
        Args:
        x_train:
            It is initialized in None and will be a np.array instance containing
            all the training data points in x for the models in QBC when using
            fit method.      
        x_train:
            It is initialized in None and will be a np.array instance containing
            all the training data points in y for the models in QBC when using
            fit method.  
        no_of_epochs:
            An positive integer number to limit the training number of epochs
            for each model in the QBC.
        batch_size:
            An positive integer number to indicate the number of batches for
            each epoch of a model during training.
        verbose:
            By setting verbose into 0, 1 or 2 to track the training progress 
            for each epoch. verbose=0 will show nothing (silent); verbose=1 
            will show an animated progress bar; verbose=2 will just mention
            the number of epoch
        max_allowed_pred_error:
            An number a to indicate an interval from [-a, a], which to tell if
            the gap between predictions and the real values fall in this area.
            If 95% of the gaps fall in this area, then stop training the model. 
        termination_mse_threshold:
            One of the early stop criteria for models' training. When the mean
            squared errors, compared with y_train, of the predictions equal or
            smaller than the assigned threshold, the training will stop if this
            terminate. Otherwise, the training will continue until the designed
            time or number of epochs.
        loss_plot_size:
            It decides whether to plot or not and the size of the plots during
            process. Default in (12, 6.75), and if it is in False, then the
            plotting function will not work.
        show_plot:
            It indicates whether to show the plots or not during processing.
            When loss_plot_size in a tuple for the figure size and show_plot in
            False, the plotting function will work and the desired figure will
            be created and stored into the working folder while not showing
            it in the user interface.
        l2_penalty_update:
            This parameter is designed to update the L2 penalty parameter of a
            neural network model during training with setting it into a list
            matching the architecture of the model. Default in False.
        mse_threshold_update:
            This parameter is to update the mean squared error threshold for
            training the neural network models in the QBC by setting a list to
            match the architecture of the model. Default in False.
        callback:
            An object that can perform customized actions during training the
            model. The input is an instance of the defined class MyCallback.
        training_time_limit:
            The variable shows the policy of terminating the fitting process of
            the model. If it is a number instead of False, then the training
            will stop even the termination criteria is not met, like the maximum
            accepted prediction error or the number of epochs. The unit of this
            parameter is minute.
        update_models_structure:
            A boolen value to controll whether to add an extra neuron to a model
            in the QBC when it can not be properly trained under the given
            fitting constraints. If it is true, then add it.
        total_time:
            A positive decimal number for controlling tha add of neurons to the
            NN network, and if 3 times of this total_time is past, then stop
            adding neurons and fitting the model. Ignore and continue. United in
            hour.

        Return:
        The return is a list of string from training_end_flag_list. A flag is 
        initialized in 'normal' when start training a model, and it means the
        training has met the termination criteria within the number of epochs 
        and the training time constrain (if there is). If it is 'time', then 
        time constrain works; If it is 'epoch', then constrain from the number
        of epochs works. For each model in the QBC, its ending flag will be
        appended to the list as the final return.
        """
        # Choose the callback strategy when training the model, if it is
        # None then the customized one by MSE threshold and number of 
        # epochs will be applied.
        method = 'threshold'
        if max_allowed_pred_error: method = 'interval'
           
        # Define a flag to control the updating of the model and the list 
        # to record the training stop's constraint.
        delete_model_flag = False
        structure_update_flag = True    

        # Define intermediate variales for the total training time.
        time_to_tick, count, limit = time.time(), 1, np.inf
        if total_time: limit = 1200 * total_time

        while structure_update_flag:
            # Start to count the time for training since the MyCallback
            # needs it as the input parameter. If there is other 
            # customized callback methods from fit's input, then use  
            # the defined MyCallback class.
            start_time = time.time()
            if training_time_limit:
                training_duration = 60 * training_time_limit
                training_time_policy = [start_time, training_duration]
            else:
                training_time_policy = False
                training_duration = np.inf

            if not callback:
                new_callback = MyCallback(
                    termination_mse_threshold, self.x_train, self.y_train,
                    max_allowed_pred_error, no_of_epochs, method,
                    training_time_policy)
            else:
                new_callback = callback

            # Generate a random seed for initializing model's parameters
            r = int(str(time.time()).split('.')[1]) + 1001 * (i + 1)
            np.random.seed(seed=r)
            tf.random.set_seed(seed=r)
            print('Model', str(i + 1), end=': ')

            # Train the model
            history_es = model.fit(self.x_train, self.y_train, 
                                   epochs=no_of_epochs, batch_size=batch_size,
                                   verbose=verbose, callbacks=[new_callback])

            # Based on the training time, number of losses to tell how 
            # model fitting is ended. The flag is initialized in 'normal',
            # and it means the training has met the termination criteria
            # within the number of epochs and the training time constrain
            # (if there is).
            training_end_flag = 'normal'
            time_consumption = time.time() - start_time
            print('after {} seconds.'.format(round(time_consumption,3)))
            if time_consumption > training_duration:
                training_end_flag = 'time'
            elif no_of_epochs == len(history_es.history["loss"]):
                training_end_flag = 'epoch'

            # Plotting for the loss-trajectory during training the model
            if loss_plot_size:
                s = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
                name = ''.join([s, '-traj_loss-for_Model_', str(i + 1), '.png'])
                plotting(x=range(len(history_es.history["loss"])),
                         y=np.log(history_es.history["loss"]),
                         xlabel='epochs', ylabel='loss', legend=False,
                         plot_size=loss_plot_size, save_name=name,
                         title=False, dimension=2, show=show_plot)
            # Update the flag and store the parameters and model into local
            # disk according to the update_models_structure policy or the
            # model's training end flag. 
            if (not update_models_structure) or training_end_flag == 'normal':
                structure_update_flag = False
            else:
                # Implement the total time policy
                time_to_tack = time.time()
                time_for_a_model = time_to_tack - time_to_tick
                model_par = self.define_parameters[i]['nn_struc_para_list']
                if time_for_a_model > limit and count < 3:
                    nr_neurons = int(sum(model_par[0][1:-1])+20*count)
                    print('{} seconds past, add {} neurons directly.'.
                            format(time_for_a_model, 20*count))
                    count += 1
                    time_to_tick = time.time()
                elif time_for_a_model > limit and count == 3:
                    print('Model cannot be fitted within {} seconds. This model'
                          ' will be removed from the QBC!'.format(limit))
                    delete_model_flag = True
                    structure_update_flag = False
                else:
                    # If the training of the model does not end in 
                    # 'normal', then add an extra neuron to the model
                    nr_neurons = sum(model_par[0][1:-1]) + 1
                structure_list = model_structure_update(model_par[0], nr_neurons)
                
                model = self.define_a_keras_nn(structure_list, model_par[1],
                                                 model_par[2], model_par[3],
                                                 model_par[4], model_par[5], 
                                                 model_par[6])

                # Update the model and its parameters to the solver's attribute.
                self.model_list[i] = model
                self.define_parameters[i]['nn_struc_para_list'] = structure_list
        return delete_model_flag

    def add_fit_tree_model(self, x_train, y_train, constraints='mse', threshold=0,
                           model_type='tree'):
            
        total_samples = len(x_train)
        indices = [i for i in range(total_samples)]

        if 'tree' in model_type:
            n = int(np.ceil(total_samples * 2 / 3))
            sampled_indices = itertools.combinations(indices, n)
            for samples in sampled_indices:
                for i in range(n, 1, -1):
                    samples = list(samples)
                    x, y = x_train[samples], y_train[samples]
                    # print('x_train', len(x_train), x_train)
                    # print('samples, i', samples, i)
                    # print('n, indices', n, indices)
                    # print('x, y', x, y)  
                    target_model = DecisionTreeRegressor(min_samples_split=i)
                    target_model.fit(x, np.ravel(y))
                    pred = np.array(target_model.predict(x)).reshape(-1, self.y_dimension)
                    gap = pred - y
                    # print('y', y)
                    # print('pred_y', target_model.predict(x))
                    # print('gap', gap)
                    # if constraints == 'mse':
                    #     # print('np.mean(gap**2)', np.mean(gap**2))
                    #     if 0 < np.mean(gap**2) <= threshold:
                    #         self.tree_forest_parameters.append(
                    #             {'min_samples_split':i, 'training_index':samples})
                    #         self.tree_forest_models.append(target_model)
                    #         break
                    if constraints == 'mae':
                        # limit = threshold * np.ones_like(gap)
                        if (np.abs(gap)< threshold).all():
                            self.tree_forest_parameters.append({'min_samples_split':i,
                                                            'training_index':samples})
                            self.tree_forest_models.append(target_model)
                            break

            model = DecisionTreeRegressor(min_samples_split=2)
            model.fit(x_train, np.ravel(y_train))
            preds = np.array(model.predict(x_train)).reshape(-1, self.y_dimension)
            gaps = preds - y_train
            if (gaps == 0).all():
                self.tree_forest_parameters.append({'min_samples_split':2,
                                                    'training_index':indices})
                self.tree_forest_models.append(model)
                
        if 'no_bootstrap' in model_type:
            no_accepted_model = []
            for i in range(2, total_samples+1):
                model = DecisionTreeRegressor(min_samples_split=i)
                model.fit(x_train, np.ravel(y_train))
                preds = np.array(model.predict(x_train)).reshape(-1, self.y_dimension)
                gap = preds - y_train
                if constraints == 'mae':
                    # limit = threshold * np.ones_like(gap)
                    if (np.abs(gap)< threshold).all():
                        self.tree_forest_parameters.append({'min_samples_split':i,
                                                        'training_index':'all'})
                        self.tree_forest_models.append(model)
                        no_accepted_model.append(i)
            minimum_models = 4
            if len(no_accepted_model) < minimum_models:
                num_extra_model = minimum_models - len(no_accepted_model)
                m = max(no_accepted_model)
                for j in range(num_extra_model):
                    split_par = m + 1 + j 
                    model = DecisionTreeRegressor(min_samples_split=split_par)
                    model.fit(x_train, np.ravel(y_train))
                    self.tree_forest_parameters.append({'min_samples_split':split_par,
                                                    'training_index':'all'})
                    self.tree_forest_models.append(model)
            # print('no_accepted_model', no_accepted_model)
            # print('len(self.tree_forest_models):', len(self.tree_forest_models))
            pass

        if 'forest' in model_type:
            f = int(np.ceil(total_samples / 2))
            sampled_indices = itertools.combinations(indices, f)

            for c, samples in enumerate(sampled_indices):
                # for j in range(5):
                samples = list(samples)
                x, y = x_train[samples], y_train[samples]
                target_model1 = RandomForestRegressor()
                target_model1.fit(x, np.ravel(y))
                # pred = np.array(target_model1.predict(x)).reshape(-1, self.y_dimension)
                # gap1 =  pred - y
                # print('gap1', gap1)

                rest = [o for o in indices if o not in samples]
                x2, y2 = x_train[rest], y_train[rest]
                target_model2 = RandomForestRegressor()
                target_model2.fit(x2, np.ravel(y2))
                # gap2 =  target_model1.predict(x_train[rest]) - y_train[rest]

                self.tree_forest_parameters.append({'random_forest':samples})
                self.tree_forest_models.append(target_model1)
                self.tree_forest_parameters.append({'random_forest':rest})
                self.tree_forest_models.append(target_model2)

                if c == 9:
                    break

        #         if constraints == 'mse':
        #             if 0 < np.mean(gap1**2) <= threshold:
        #                 self.tree_forest_parameters.append({'random_forest':samples})
        #                 self.tree_forest_models.append(target_model1)
        #             # if 0 < np.mean(gap2**2) <= threshold:
        #             #     self.tree_forest_parameters.append({'random_forest':rest})
        #             #     self.tree_forest_models.append(target_model2)
        #         if constraints == 'mae':
        #             # print('threshold', threshold)
        #             # limit = threshold * np.ones_like(gap1)
        #             # print('threshold', threshold)

        #             if (np.abs(gap1) > 0).all() and (np.abs(gap1)< threshold).all():
        #                 self.tree_forest_parameters.append({'random_forest':samples})
        #                 self.tree_forest_models.append(target_model1)
        #             # if (0 < np.abs(gap2)).all() and (np.abs(gap2)< threshold).all():
        #             #     self.tree_forest_parameters.append({'random_forest':rest})
        #             #     self.tree_forest_models.append(target_model2)
        # # print('len(self.tree_forest_models):', len(self.tree_forest_models))
        # print('self.tree_forest_models', self.tree_forest_models)
        # print('self.tree_forest_parameters', self.tree_forest_parameters)
        pass

    def fit_models(self, x_train, y_train, save_model=False, update_model={},
                   booststraping={}, extra_tree_method='', tau=['mae', 0], **kargs):
        # Reshape the np.array to meet the input requirement of the object
        # function Update the attributes of the class
        x_train = x_train.reshape(-1, self.x_dimension)
        y_train = y_train.reshape(-1, self.y_dimension)
        
        # Find out the minimum point in y_train for expected improvement
        # calculation. If the dimension of y larger than 1, L2 distance will
        # be used.
        if self.y_dimension < 2:
            y_t = y_train
        else:
            y_t = np.linalg.norm(y_train, keepdims=True, axis=1)

            # np.sqrt(np.sum(np.power(y_train, 2), axis=1))
        self.y_max = y_train[np.argmax(y_t)].reshape(-1, self.y_dimension)
        self.y_min = y_train[np.argmin(y_t)].reshape(-1, self.y_dimension)

        self.fit_parameters = kargs
        self.fit_parameters['save_model'] = save_model
        self.fit_parameters['update_model'] = update_model
        self.fit_parameters['booststraping'] = booststraping
        self.fit_parameters['extra_tree_method'] = extra_tree_method
        self.fit_parameters['tau'] = tau
      
        if not (len(self.pred_by_QBC) + len(self.pred_by_PSO) == 0):
            for key in update_model:
                for i, model_para in enumerate(self.define_parameters):
                    which_model = 'Model_' + str(i)
                    if (which_model in key) or (key in str(self.model_types[i])):
                        for par in model_para:
                            if par in update_model[key].keys():
                                if not callable(update_model[key][par]):
                                    model_para[par] = update_model[key][par]
                                else:
                                    model_para[par] = update_model[key][par]\
                                                    (model_para[par])
                        # new_par = {**model_para, **update_model[key]}
                    new_model, type, pars = self.define_a_model(self.x_dimension,
                            self.y_dimension, self.model_types[i], **model_para)
                    self.model_list[i] = new_model

        delete_models = []
        for j, model in enumerate(self.model_list):
            # print('self.define_parameters:', self.define_parameters[i])
            if 'keras' in str(model):
                delete_flag = self.fit_keras_nn(
                    model, j, **self.fit_parameters['keras_nn'])
                if delete_flag:
                    delete_models.append(j)
                continue
            else:
                indices = [k for k in range(len(x_train))]
                if booststraping:
                    size = len(indices)
                    for func in booststraping:
                        if callable(func):
                            size = func(size)
                if ('all_models' in booststraping) or (j in booststraping) \
                   or (booststraping is True):
                    indices = np.random.choice(indices, size=size)

                fit_flag = False
                which_model = 'Model_' + str(j)
                for model_type in kargs.keys():
                    if model_type == which_model or model_type in str(model):
                        model.fit(x_train[indices], np.ravel(y_train[indices]),
                                  **self.fit_parameters[model_type])
                        fit_flag = True
                if not fit_flag:
                    # print('x_train[indices]', x_train[indices])
                    # print('y_train[indices]', y_train[indices])
                    model.fit(x_train[indices], np.ravel(y_train[indices]))
            if save_model:
                self.save_model(model_positions=j)
                self.save_parameters()
            # print('Model {} is fitted.'.format(i+1))

        for l in delete_models:
            self.delete_model_from_QBC(l)

        if 'tree' in extra_tree_method or 'forest' in extra_tree_method or\
            'no_bootstrap' in extra_tree_method:
            self.add_fit_tree_model(x_train, y_train, constraints=tau[0],
                                    threshold=tau[1], model_type=extra_tree_method)

        if len(self.pred_by_QBC) + len(self.pred_by_PSO) == 0:
            self.x_train, self.y_train = x_train, y_train
            self.model_list_PSO = self.model_list.copy()
            self.x_train_PSO, self.y_train_PSO = x_train.copy(), y_train.copy()

    def predict_a_step_by_QBC(self, x_range, y_range=False, pdf_type='Gaussian',
                              plot_size=(12, 6.7), show_plot=True, 
                              aqf_plot=False, threshold=0.01, epsilon=0.1):
        """Predict x_next from the query based committee (QBC).
        
        Args:
        x_range:
            It should be a np.array instance containing all the points desired
            to be investigated and evaluated for prediction x_next by QBC.
        y_true_in_x_range:
            The real y values for x in x_range. This is mainly used for plotting
            the original function in the figures for comparing the y predictions
            from models in QBC.
        pdf_type:
            When calculate the expected improvement of y for each x in x_range,
            a probability density function need to be estimated for y based on
            the multiple prediction values from models in QBC. Policies with a
            Gaussian distribution or processing by kernel density estimation can
            be chosen. Only 'Gaussian' or 'Kernel' can set to it, and it is in 
            'Gaussian' in default.
        plot_size:
            It decides whether to plot or not and the size of the plots during
            process. Default in (12, 6.7), and if it is in False, then the
            plotting function will not work.
        show_plot:
            It indicates whether to show the plots or not during processing.
            When plot_size in a tuple for the figure size and show_plot in
            False, the plotting function will work and the desired figure will
            be created and stored into the working folder while not showing
            it in the user interface.
        aqf_plot:
            It decides whether or not to plot the results from the Acquisition
            Function (aqf) Process. If it is True, then plotting function will
            work and the figures' sizes and the policy of showing them in the
            interface will follow the settings in plot_size and show_plot. If it
            is False, then the aqf data will not be plotted. 
        threshold:
            This threshold is for the policy of the solver escaping from stuck
            into a local minimum. When the predicted x_next is too close to a 
            training data point, x_next will be replaced by a random choice from
            other suitable points in x_range. The threshold is an decimal from 0
            to 1. It indicates the percentage of the maximum distance among all
            the points in x_train. This calculated distance will be used to tell
            whether x_next is too close to a point.
        epsilon:
            The epsilon indicates the solver's ability of continuing the search
            in the local area. When x_next is analysed within the threshold
            distance range of a training point, a probability of epsilon will be
            used to continue to use this x_next as the solution, and a 1-epsilon
            probability for change x_next by random choice.             

        Return:
        A np.array instance with the same shape of a single element in x_range.
        It is the prediction of x_next where have a higher opportunity to find
        out the optimal value of the object function.
        """
        # Reshape x_range to fit the object function, update the attributes,
        # and define variables for containing data in the middle of process.
        x_range = x_range.reshape(-1, self.x_dimension)
        self.x_range, self.y_range = x_range, y_range
        self.predict_parameters = [0] * 6
        for j, para in enumerate((pdf_type, plot_size, show_plot, aqf_plot,
                                  threshold, epsilon)):
            self.predict_parameters[j] = para

        all_pred_y_list, plot_legend = [], []

        # Predict y for each x in x_range by every model in QBC.
        for i, model in enumerate(self.model_list):
            predicted_y_in_x_range = model.predict(x_range)
            all_pred_y_list.append(np.squeeze(predicted_y_in_x_range))
            plot_legend.append(''.join(['model_', str(i + 1)]))

        k = len(self.model_list) + len(self.tree_forest_models)
        if len(self.tree_forest_models) > 0:
            for j, tree_model in enumerate(self.tree_forest_models):
                predicted_y_range = tree_model.predict(x_range)
                all_pred_y_list.append(np.squeeze(predicted_y_range))
                plot_legend.append(''.join(['extra_tree_', str(j + 1)]))
            # print('self.tree_forest_parameters:', self.tree_forest_parameters)
            self.tree_forest_models = []
            self.tree_forest_parameters = []

        # Transpose the array containing all the predictions according to
        # the dimensions in y;
        if self.y_dimension == 1:
            all_pred_y = np.transpose(np.asarray(all_pred_y_list))
        else:
            all_pred_y = np.transpose(np.asarray(all_pred_y_list), axes=(1,0,2))

        all_pred_y = all_pred_y.reshape(-1,k)      

        # print('all_pred_y contains np.nan:', [x_range[i] for i, j in enumerate(all_pred_y) if np.isnan(j).any()])
        # Choose the way of getting the estimated distribution for y of each x;
        if 'Gaussian' in pdf_type:
            aqf_est = self.gaussian_est
        elif 'Kernel' in pdf_type:
            aqf_est = self.kernel_density_est
        elif 'iqr' in pdf_type:
            aqf_est = self.iqr_est

        if not isinstance(pdf_type, list):
            pdf_type = [0]
        ei_np_2d_col_array = aqf_est(all_pred_y, *pdf_type[1:])
        # print('ei_np_2d_col_array.shape', ei_np_2d_col_array.shape)
        # Find out the index with the maximum EI in EI_np_2d_col_array
        # based on the dimension of y. L2 distance is used for dimension over 1;
        # Mask ei to -np.inf in the training points' positions.
        if self.y_dimension < 2:
            temp_ei = ei_np_2d_col_array
        else:
            temp_ei = np.sqrt(np.sum(np.power(ei_np_2d_col_array, 2), axis=1))
        
        # print('temp_ei.shape', temp_ei.shape)
        # ei_array = nparray_process(self.x_train, x_range, temp_ei, process='inf')
        ei_array = np.copy(temp_ei)
        for x0 in self.x_train:
            for a, x1 in enumerate(x_range):
                if (x0 == x1).all():
                    ei_array[a] = np.zeros_like(x1)

        ei_array = rolling_mean(self.x_dimension, ei_array, n=5)

        # tem_ei = np.linalg.norm(ei_np_2d_col_array, keepdims=True, axis=1)
        # ei_array = nparray_process(self.x_train, x_range, tem_ei, process='inf')

        # ei_array[np.isnan(ei_array)] = 0
        # print('ei_array[200:250]', ei_array[200:250])
        # print('x_range[200:250]', x_range[200:250])
        # plt.plot(x_range[170:190],ei_array[170:190])
        # plt.show()

        ei_max = np.max(ei_array)
        ei_indices = [i for i, j in enumerate(ei_array) if (j==ei_max).all()]
        if len(ei_indices) % 2 == 1:
            pred_index = int(np.median(ei_indices))
        else:
            pred_index = ei_indices[int(len(ei_indices)/2)]

        # print('pred_index median', pred_index)
        # Get the x_next as the predicted optimal by the index in maximum EI.
        # pred_index = np.argmax(ei_array)
        # print('pred_index argmax directly', pred_index)
        pred_x_next = x_range[pred_index].reshape(1, self.x_dimension)

        # Employ the probability to avoid trapping into the local minimum;
        probability = np.random.uniform(0, 1, 1).item()
        if threshold and (probability <= epsilon):
            gap_matrix = self.x_train - pred_x_next
            percentage = np.linalg.norm(gap_matrix, keepdims=True, axis=1)\
                                              / max_distance(self.x_train)

            if np.min(percentage) <= threshold:
                all_candidates = nparray_process(self.x_train, x_range,
                                                 threshold, process='threshold')
                if len(all_candidates) == 0:
                    threshold = False
                    self.epsilon_filter_out = []
                    print('!! The threshold is too large and filters out all '
                          'the data in x_range. Switch epsilon-threshold off.')
                if threshold:
                    if isinstance(y_range, list):
                        noised_y_true = y_range[1]
                    else:
                        noised_y_true = y_range
                    indices = indices_by_kmeans_process(x_range, self.x_train, 
                               [noised_y_true, self.y_train], fsize=plot_size,
                               show_plot=show_plot)
                    ei_in_clusters = ei_array[indices]
                    new_x_next = x_range[indices[np.argmax(ei_in_clusters)]]
                    print('!!!! The x_next predicted by the model is {}. Since '
                          'it is close to {} in the training dataset, the new '
                          'x_next becomes {} by epsilon-mechanism.'.format(
                        np.squeeze(pred_x_next), np.squeeze(self.x_train\
                        [np.argmin(percentage)]), np.squeeze(new_x_next)))
                    pred_x_next = new_x_next
                    # self.epsilon_filter_out.append(pred_x_next.tolist())
        # Store the prediction into the attribute
        self.pred_by_QBC.append(np.squeeze(pred_x_next).tolist())

        # Gaining the dimension information to tell whether plot works or not
        dimension = self.x_dimension + self.y_dimension
        # For x in 2D and y in 1d, plotting all the surfaces makes the figure 
        # messy and plotting the mean of the predictions instead.  
        # For x and y both in 1D, plotting all the predictions in the figure.
        mean = np.mean(all_pred_y, axis=1)
        pred_std_QBC = np.std(all_pred_y, axis=1)
        if dimension == 3 and self.y_dimension == 1:
            y_groups_for_plot, plot_legend = [mean], ['mean_of_all_preds']
        else:
            y_groups_for_plot = all_pred_y_list
            # y_groups_for_plot, plot_legend = [], []
            first_q = np.percentile(all_pred_y, 25, axis=1)
            third_q = np.percentile(all_pred_y, 75, axis=1)
            data = [first_q, third_q, mean]
            labels = ['25%_quantile', '75%_quantile', 'mean_of_all_preds']
            for c in range(3):
                y_groups_for_plot.append(data[c])
                plot_legend.append(labels[c])

            if len (all_pred_y_list) > 70:
                y_groups_for_plot, plot_legend = [], []
                first_q = np.percentile(all_pred_y, 25, axis=1)
                third_q = np.percentile(all_pred_y, 75, axis=1)
                data = [first_q, third_q, mean]
                labels = ['1%_quantile', '99%_quantile', 'mean_of_all_preds']
                for c in range(3):
                    y_groups_for_plot.append(data[c])
                    plot_legend.append(labels[c])
            # else:            
            #     y_groups_for_plot.append(mean)
            #     plot_legend.append('mean_of_all_preds')


        # plotting
        if plot_size:
            if dimension ==  2:
                x_range_plot = [x_range, pred_std_QBC]
            else:
                x_range_plot = x_range
            title = 'All training, predicted, and true dataset'
            stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
            save_name = ''.join([stamp, '_all_predicted_and_true_y_values', '.png'])
            plotting(x=self.x_train, y=self.y_train, all_x=x_range_plot,
                     all_true_y=y_range, all_ptd_y=y_groups_for_plot,
                     save_name=save_name, predicted_point=pred_x_next,
                     zlabel='y', legend=plot_legend, plot_size=plot_size,
                     show=show_plot, title=title, dimension=dimension)
            # For the case of y1, y2=f(x), the EI-x plot should be a 2D plot.
            if self.y_dimension == 2:
                dimension = 2
            if aqf_plot:
                title = 'EI by ' + str(aqf_est).split(' ')[2][21:]
                if isinstance(y_range, list):
                    noised_y_true = y_range[1]
                else:
                    noised_y_true = y_range
                ti_stamp= datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
                plotting(x=x_range, y=ei_array, title=title,
                         predicted_point=[pred_x_next, noised_y_true],
                         plot_size=plot_size, show=show_plot,
                         dimension=dimension,
                         legend=['EI ' + str(aqf_est).split(' ')[2][21:],
                                 'Pred_x_next'],
                         save_name=''.join([ti_stamp, '_EI_in_all_x', '.png']))

        print('The predicted x_next={} with EI of y in {}.'.format(
               np.squeeze(pred_x_next), np.max(ei_np_2d_col_array)))

        return self.pred_by_QBC

    def predict_n_steps_by_QBC(self, true_y_next=False, n_step=1,
                               orig_func=False, x_range=False, pdf_type='Gaussian',
                               plot_size=(12, 6.7), show_plot=True,
                               aqf_plot=False, threshold=0.01, epsilon=0.1, 
                               accept_x_interval=[np.nan, np.nan]):

        """Predict x_next in n steps if the original function or data is given.
        
        Args:
        true_y_next:
            Default in False. If given, it will be the y value of pred_x_next
            from the predict method. And this pair of data will be as an extra 
            point put into the training dataset.
        n_step:
            Integer indicate the number of prediction steps when the orig_func is
            set with an function. When true_y_next is set with values for y, then
            only 1 step can be proceeded.
        orig_func:
            The original function of generating x_train and y_train, and used to 
            get the y value of x_next to put into the training dataset.
        noise_level:
            Import an noise level for the y value calculated from original
            function.
        x_range:
            It should be a np.array instance containing all the points desired
            to be investigated and evaluated for prediction x_next by QBC.
        pdf_type:
            When calculate the expected improvement of y for each x in x_range,
            a probability density function need to be estimated for y based on
            the multiple prediction values from models in QBC. Policies with a
            Gaussian distribution or processing by kernel density estimation can
            be chosen. Only 'Gaussian' or 'Kernel' can set to it, and it is in 
            'Gaussian' in default.
        plot_size:
            It decides whether to plot or not and the size of the plots during
            process. Default in (12, 6.7), and if it is in False, then the
            plotting function will not work.
        show_plot:
            It indicates whether to show the plots or not during processing.
            When plot_size in a tuple for the figure size and show_plot in
            False, the plotting function will work and the desired figure will
            be created and stored into the working folder while not showing
            it in the user interface.
        aqf_plot:
            It decides whether or not to plot the results from the Acquisition
            Function (aqf) Process. If it is True, then plotting function will
            work and the figures' sizes and the policy of showing them in the
            interface will follow the settings in plot_size and show_plot. If it
            is False, then the aqf data will not be plotted. 
        threshold:
            This threshold is for the policy of the solver escaping from stuck
            into a local minimum. When the predicted x_next is too close to a 
            training data point, x_next will be replaced by a random choice from
            other suitable points in x_range. The threshold is an decimal from 0
            to 1. It indicates the percentage of the maximum distance among all
            the points in x_train. This calculated distance will be used to tell
            whether x_next is too close to a point.
        epsilon:
            The epsilon indicates the solver's ability of continuing the search
            in the local area. When x_next is analysed within the threshold
            distance range of a training point, a probability of epsilon will be
            used to continue to use this x_next as the solution, and a 1-epsilon
            probability for change x_next by random choice. 
    
        Return:
        All the predicted x_next from the given predicted steps.
        """
        # Check whether the solver has predicted by self.predict_by_QBC or not
        # If not, then make the fist prediction
        if (not isinstance(x_range,bool)) and (not self.pred_by_QBC):
            # Find out all the y values over x_range for self.predict_by_QBC
            if callable(orig_func):
                y_range = orig_func(x_range).reshape(-1, self.y_dimension)
            else:
                y_range = orig_func#.reshape(-1, self.y_dimension)
            print('----------------------------------------------------------')
            print('The 1st prediction by QBC.', end=' ')
            self.predict_a_step_by_QBC(x_range, y_range, pdf_type, plot_size,
                                       show_plot, aqf_plot, threshold, epsilon)
            if n_step:
                n_step -= 1

            x_test = np.array(self.pred_by_QBC[-1]).reshape(1,-1)
            if (x_test >= accept_x_interval[0]).all() and \
               (x_test <= accept_x_interval[1]).all():
                print('The accepted minimum x_next is found at {}, which is in',
                      ' the range of [{}, {}], thus stop running further steps.'.
                      format(x_test, accept_x_interval[0], accept_x_interval[1]))
                n_step = 0

        if n_step > 0:
            # Get the y value for x_next according to the input parameter
            if (not isinstance(orig_func, bool)) and isinstance(true_y_next, bool):
                x_next = np.array(self.pred_by_QBC[-1]).reshape(1,-1)
                if callable(orig_func):
                    y_next = orig_func(x_next)
                else:
                    if isinstance(orig_func, list):
                        used_y = orig_func[0]
                    else:
                        used_y = orig_func
                    y_next = np.array([used_y[i] for i, j in enumerate(x_range)
                                    if (j==x_next).all()]).reshape(1,-1)

            elif (orig_func is False) and (not isinstance(true_y_next, bool)):
                y_next = true_y_next
                # Giving true_y instead of orig_func leads only 1 prediction
                n_step = 0

            # Put x_next and its y value into the training dataset.
            self.x_train = np.append(self.x_train, x_next)
            self.y_train = np.append(self.y_train, y_next)
            print('Extended x_train:', np.squeeze(self.x_train))
            print('Extended y_train:', np.squeeze(self.y_train))

            # Training the models with the updated training dataset;
            self.fit_models(self.x_train, self.y_train, **self.fit_parameters)
            nr_prediction = len(self.pred_by_QBC) + 1
            print('----------------------------------------------------------')
            print('The {}th prediction by QBC.'.format(nr_prediction), end=' ')
            # Predict x_next and store it in self.pred_by_QBC by the method;
            # Retrieve the parameters of creating and fitting models in QBC;
            ppar = self.predict_parameters
            self.predict_a_step_by_QBC(self.x_range, self.y_range, ppar[0], 
                                       ppar[1], ppar[2], ppar[3], ppar[4], ppar[5])

            x_test = np.array(self.pred_by_QBC[-1]).reshape(1,-1)
            if (x_test >= accept_x_interval[0]).all() and \
               (x_test <= accept_x_interval[1]).all():
                print('The accepted minimum x_next is found at {}, which is in',
                      ' the range of [{}, {}], thus stop running further steps.'.
                      format(x_test, accept_x_interval[0], accept_x_interval[1]))
                n_step = 0

            # To check how many prediction steps left. If larger than 1, repeat the
            # process with steps reducing 1.
            if n_step > 1:
                self.predict_n_steps_by_QBC(true_y_next=true_y_next, n_step=n_step-1,
                        orig_func=orig_func, x_range=x_range, pdf_type=pdf_type,
                        plot_size=plot_size, show_plot=show_plot, aqf_plot=aqf_plot,
                        threshold=threshold, epsilon=epsilon)
                    
        return self.pred_by_QBC

    def predict_by_PSO(self, search_space, aqf='Gaussian', plot_size=[8, 6],
                         orig_func=False, n_steps=1, show_plot=True, 
                         nr_po_x_range=100, nr_particles=100, v_max=2.5, c1=0.3,
                         c2=2.1, w=1.0, nr_steps=1000):
        """Predict x_next by PSO and the query based committee.
        
        Args:
        search_space:
            A special designed np.array with shape in nx3 to indicate the search
            space for the solution. n is the dimension information for x. The
            first column of the array indicates the data type. '1' stands for
            continuous data, '2' stands for categorized data and '3' stands for
            encoded data. For continuous data, the second and third column are
            the minimum and maximum limit, any number between them is valid;
            For categorized data, the second and third column are the lowest and
            highest limit, only integer number between them is valid; For
            encoded data, the second and third columns are the beginning and
            ending positions of the encoding group, each encoded group should be
            in continuous rows, and only a '1' and others in '0' is valid; An
            example for search_space is as following:
            np.array([[1, 1.1, 5.1],  # continuous data between 1.1 to 5.1
                    [3, 1, 3],        # encoded data group 1 from row 1
                    [3, 1, 3],        # encoded data group 1 with row 2
                    [3, 1, 3],        # encoded data group 1 ends in row 3
                    [2, 0, 5],        # categorized data in 0, 1, 2, 3, 4, 5.
                    [1, -10, 20],     # continuous data between -10 to 20
                    [3, 6, 7],        # encoded data group 2 from row 6
                    [3, 6, 7],        # encoded data group 2 ends in row 7
                    [2, 2, 4],        # categorized data in 2, 3, 4.
                    ]])
            In this case, a valid data point for x is as following:
            np.array([2.2, 0, 1, 0, 3, 8.5, 1, 0, 4])
        aqf:
            When calculate the expected improvement of y for each x,
            a probability density function need to be estimated for y based on
            the multiple prediction values from models in QBC. Policies with a
            Gaussian distribution or processing by kernel density estimation can
            be chosen. Only 'Gaussian' or 'Kernel' can set to it, and 'Gaussian'
            is in default.
        plot_size:
            It decides whether to plot or not and the size of the plots during
            process. Default in (12, 6.75), and if it is in False, then the
            plotting function will not work.
        show_plot:
            It indicates whether to show the plots or not during processing.
            When plot_size in a tuple for the figure size and show_plot in
            False, the plotting function will work and the desired figure will
            be created and stored into the working folder while not showing
            it in the user interface.
        pred_steps:
            Integer indicate the number of prediction steps when the orig_func 
            is set with the original function or all true y values. Default in
            1 for orig_func is not given.
        orig_func:
            The original function of generating x_train and y_train, or an
            array containing all true y values over x_range. It is used to 
            get the y value of x_next to put into the training dataset.
        noise_level:
            Import an noise level for the y value calculated from original
            function.
        nr_particles, v_max, c1, c2, w, nr_steps: Parameters for PSO method.

        Return:
        The self.pred_by_PSO attribute contianing np.array instance from
        predictions by PSO method.
        """
        # fitted_QBC_models = self.model_list
        if len(self.pred_by_PSO) == 0:
            self.model_list, self.model_list_PSO = self.model_list_PSO, self.model_list

        # Update the search space attribute to the solver
        self.search_space = search_space
        # Define the objective function for PSO. In this case, the function
        # should be the one to return the EI of a x input.

        # Choose the estimation method to calculate the EI
        if 'Gaussian' in aqf:
            func = self.gaussian_est
        elif 'Kernel' in aqf:
            func = self.kernel_density_est
        elif 'iqr' in aqf:
            func = self.iqr_est
        if not isinstance(aqf, list):
            aqf_par = [0]
        else:
            aqf_par = aqf

        def aqf_for_a_single_point(x):
            all_preds_for_x = []
            # Use QBC to predict the y values for the given x.
            for model in self.model_list:
                all_preds_for_x.append(model.predict(x))
            input = np.array([all_preds_for_x]).reshape(1, -1)
            # print('input', input, np.mean(input), np.var(input))
            aqf_evaluation = func(input, *aqf_par[1:])
            # In case of y in multiple dimensions. Since EI is alway positive,
            # thus use np.sum directly instead of L2 distance.
            # output = np.sum(np.array(aqf_evaluation))
            # print('output', type(output), output)
            # print('aqf_evaluation', aqf_evaluation)
            return np.array([aqf_evaluation])

        # Define PSO object with the given/defined parameters
        my_pso = PSO(aqf_for_a_single_point, nr_particles, search_space,
                     v_max, c1, c2, w, False)
        # Use autorun to find the optimal point by PSO
        x_pso, f_pso = my_pso.auto_run(nr_steps=nr_steps)
        print('The optimal x found by PSO is {} with EI in {}.'
              .format(x_pso, f_pso))
        # Store the results to the attribute
        self.pred_by_PSO.append((x_pso.tolist(), f_pso.tolist()))

        # Check the conditions or prepare the data for plotting
        if np.sum(search_space[:,0] > 1) > 0:
            print('Categorized or encoded data exists, cannot plot.')
        elif len(search_space) < 3:
            raw_x_range = []
            for i in range(len(search_space)):
                low, high = search_space[i][1], search_space[i][2]
                raw_x_range.append(np.linspace(low, high, nr_po_x_range))
            if len(raw_x_range) == 1:
                x_range = raw_x_range[0]
            else:
                x10, x20 = np.meshgrid(raw_x_range[0], raw_x_range[1])
                x1, x2 = x10.ravel(), x20.ravel()
                x_range = np.zeros([len(x1), 2])
                for j in range(len(x1)):
                    x_range[j] = np.array([x1[j], x2[j]])
        # Plotting
        if (not isinstance(orig_func, bool)) and plot_size:
            dimension = self.x_dimension + self.y_dimension
            if callable(orig_func):
                y_range = orig_func(x_range)
            elif isinstance(orig_func, list) or isinstance(orig_func, np.ndarray):
                y_range = orig_func
            title = 'All true, training and predicted points'
            stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
            save_name = ''.join([stamp, '-predicted_by_PSO', '.png'])
            plotting(x=self.x_train_PSO, y=self.y_train_PSO, all_x=x_range,
                      zlabel='y', all_true_y=y_range, save_name=save_name,
                     predicted_point=x_pso, plot_size=plot_size, legend=[],
                     show=show_plot, title=title, dimension=dimension)

        # If the original function is not given, perform only 1 prediction.
        if isinstance(orig_func, bool):
            n_steps = 0

        # To check how many prediction steps left. If larger than 1, repeat the
        # process with steps reducing 1.
        if n_steps > 1:
            # Calculate y_next for x_next to add them into the training dataset
            x_next = np.array(self.pred_by_PSO[-1][0]).reshape(1, -1)
            if callable(orig_func):
                y_next = orig_func(x_next)
            elif not (callable(orig_func) and isinstance(self.x_range, list)):
                if isinstance(orig_func, list):
                    all_y = orig_func[1]
                if isinstance(orig_func, np.ndarray):
                    all_y = orig_func
                y_next = all_y[np.argmin(np.sqrt((self.x_range - x_next)**2))]
                y_next = y_next.reshape(1,-1)
                # y_next = np.array([all_y[i] for i, j in enumerate(self.x_range)
                #                    if (j==x_next).all()]).reshape(1,-1)
            else:
                print('y_next cannot be gained, thus cannot continue!')
            # Put x_next and its y value into the training dataset.
            self.x_train_PSO = np.append(self.x_train_PSO, x_next)
            self.y_train_PSO = np.append(self.y_train_PSO, y_next)
            # print('self.x_train_PSO', self.x_train_PSO)
            # print('self.y_train_PSO', self.y_train_PSO)

            # Training the models with the updated training dataset;
            self.fit_models(self.x_train_PSO, self.y_train_PSO, **self.fit_parameters)
            # Use PSO to predict one more step
            self.predict_by_PSO(search_space, aqf, plot_size, orig_func, 
                                n_steps-1, show_plot, nr_po_x_range,
                                nr_particles, v_max, c1, c2, w, nr_steps)
        # else:
        #     # self.model_list_PSO = self.model_list
        #     # self.model_list = fitted_QBC_models
        #     self.model_list_PSO, self.model_list = self.model_list, self.model_list_PSO

        return self.pred_by_PSO

    def predict_by_random_guess(self, x_train, y_train, x_range, n_step=1,
                                y_range=False, plot_size=(12, 6.75),
                                show_plot=True):
        """A random guess solver to predict x_next.
        
        Args:
        x_train:
            It is a np.array instance containing all the training data points
            in x.
        x_train:
            It is a np.array instance containing all the training data points
            in y. x_train and y_train are used to plotting purpose.
        x_range:
            It is a np.array instance containing all the points desired
            to be investigated and evaluated as x_next by this solver.
        n_step:
            Integer indicate the number of prediction steps by this solver;
        y_true_in_x_range:
            The real y values for x in x_range. This is used for plotting
            the original function into the figures for illustration.
        plot_size:
            It decides whether to plot or not and the size of the plots during
            process. Default in (12, 6.75), and if it is in False, then the
            plotting function will not work.
        show_plot:
            It indicates whether to show the plots or not during processing.
            When plot_size in a tuple for the figure size and show_plot in
            False, the plotting function will work and the desired figure will
            be created and saved into the working folder while not showing
            it in the user interface.
        noise_level:
            Import a noise level for the y value calculated from original
            function.

        Return:
        A list instance containing all the predicted x_next in the assigned
        n_step.
        """
        # Get the dimension information from the input data if they are not set.
        if not (self.x_dimension and self.y_dimension):
            self.x_dimension = x_train[0].size
            self.y_dimension = y_train[0].size

        # Reshape the np.array to meet the input requirement of the object
        # function Update the attributes of the class
        x_train = x_train.reshape(-1, self.x_dimension)
        y_train = y_train.reshape(-1, self.y_dimension)
        self.x_train_rand, self.y_train_rand = x_train, y_train

        # Remove the training data from x_range and y_true_in_x_range. If 
        # y_true_copy is not generated, it cannot be assured that the y value
        # for x_next are paired.
        x_range_copy = nparray_process(x_train, x_range, x_range, process='del')
        if isinstance(y_range, list):
            y_true_copy = nparray_process(x_train, x_range, y_range[1], process='del')
        else:
            y_true_copy = nparray_process(x_train, x_range, y_range, process='del')
        # Generate a random int as the index for x_next and its noised y value.
        random_int = np.random.randint(len(x_range_copy), size=1)
        x_next = x_range_copy[random_int]
        y_next = y_true_copy[random_int]

        # Print and append the data into the self.pred_by_random attribute.
        self.pred_by_random.append((x_next.tolist(), y_next.tolist()))
        print('The predicted x_next by random guess is {}, and the corresponding'
              ' y of x_next is {}.'.format(np.squeeze(x_next), np.squeeze(y_next)))

        # Get the dimension information for plotting.
        dimension = self.x_dimension + self.y_dimension
        if plot_size:
            title = 'Random prediction with all true and training data points'
            stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
            save_name = ''.join([stamp, '-pred_x_next_by_random_guess', '.png'])
            plotting(x=x_train, y=y_train, all_x=x_range, all_true_y=y_range,
                     show=show_plot, all_ptd_y=False, zlabel='y',
                     dimension=dimension, predicted_point=x_next, legend=[],
                     plot_size=plot_size, title=title, save_name=save_name)

        # Tell if it needs to continue to predict x_next by using this solver.
        if n_step > 1:
            # Update the training dataset with x_next and its y value.
            self.x_train_rand = np.append(self.x_train_rand, x_next)
            self.y_train_rand = np.append(self.y_train_rand, y_next)
            self.predict_by_random_guess(self.x_train_rand, self.y_train_rand,
                                         x_range, n_step-1, y_range,
                                         plot_size, show_plot)
        if n_step == 1:
            self.save_parameters()
        return self.pred_by_random

    def predict_by_GP(self, x_train, y_train, x_range, n_step=1, aqf_plot=True,
                      plot_size=(12, 6.75), show_plot=True,
                      y_true_in_x_range=False, constant_value=1.0,
                      orig_func=False, length_scale=1.0, alpha=1e-10,
                      constant_value_bounds=(1e-05, 100000.0),
                      length_scale_bounds=(1e-05, 100000.0),
                      optimizer='fmin_l_bfgs_b', n_restarts_optimizer=30,
                      normalize_y=True):
        """A Gaussian Process solver from Sklearn to predict x_next.

        The referenced link of the arguments for defining the Gaussian Process
        Model are as following:
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html
        https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html

        Args:
        x_train:
            It is a np.array instance containing all the training points in x.
        x_train:
            It is a np.array instance containing all the training data points
            in y. x_trian and y_train are used to training the model.
        x_range:
            It should be a np.array instance containing all the points desired
            to be investigated and evaluated as x_next by this solver.
        n_step:
            Integer indicate the number of prediction steps by this solver;
        aqf_plot:
            It decides whether or not to plot the results from the Acquisition
            Function (aqf) Process. If it is True, then plotting function will
            work and the figures' size and the policy of showing them in the
            interface will follow the settings in plot_size and show_plot. If it
            is in False, then the aqf data will not be plotted.  
        plot_size:
            It decides whether to plot or not and the size of the plots during
            process. Default in (12, 6.75), and if it is in False, then the
            plotting function will not work.
        show_plot:
            It indicates whether to show the plots or not during processing.
            When plot_size in a tuple for the figure size and show_plot in
            False, the plotting function will work and the desired figure will
            be created and stored into the working folder while not showing
            it in the user interface.          
        y_true_in_x_range:
            The real y values for x in x_range. This is mainly used for plotting
            the original function into the figures for illustration.
        orig_func:
            The original function of generating x_train and y_train, and used to 
            get the y value of x_next to put into the training dataset.
        constant_value:
            A constant value which defines the covariance for the class of
            sklearn.gaussian_process.kernels.ConstantKernel.
        length_scale:
            The length scale of the kernel. If it is a float, an isotropic
            kernel wil be used. If it is an array, an anisotropic kernel will
            be used where each dimension of l defines the length-scale of the
            respective feature dimension.
        alpha:
            Value added to the diagonal of the kernel matrix during fitting.
        constant_value_bounds:
            The lower and upper bound on constant_value. If set to 'fixed', 
            constant_value cannot be changed during hyperparameter tuning.
        length_scale_bounds:
            The lower and upper bound on length_scale. If set to 'fixed',
             length_scale cannot be changed during hyperparameter tuning.
        optimizer:
            Default in 'fmin_l_bfgs_b'. 
        n_restarts_optimizer:
            The number of restarts of the optimizer for finding the kernel's
            parameters which maximize the log-marginal likelihood.
        normalize_y:
            Whether to normalize the target values y by removing the
            mean and scaling to unit-variance.
        noise_level:
            Import a noise level for the y value calculated from original
            function.           

        Return:
        A list instance containing all the predicted x_next in the assigned
        n_step.
        """
        # Get the dimension information from the input data if they are not set.
        if not (self.x_dimension and self.y_dimension):
            self.x_dimension = x_train[0].size
            self.y_dimension = y_train[0].size
        # Reshape the data to meet the input requirement of the orig_func.
        x_train = x_train.reshape(-1, self.x_dimension)
        y_train = y_train.reshape(-1, self.y_dimension)
        x_range = x_range.reshape(-1, self.x_dimension)
        self.x_train_GP, self.y_train_GP = x_train, y_train

        # Construct the Gaussian Process Model from Sklearn and train it.
        kernel = gp.kernels.ConstantKernel(constant_value,
                                           constant_value_bounds) \
                 * gp.kernels.RBF(length_scale, length_scale_bounds)
        model = gp.GaussianProcessRegressor(kernel=kernel, optimizer=optimizer,
                                            n_restarts_optimizer=n_restarts_optimizer,
                                            alpha=alpha,
                                            normalize_y=normalize_y)
        model.fit(x_train, y_train)
        # print(model.kernel_.hyperparameters)
        # Using the Gaussian Process the y values for every x in x_range.
        # pred_std is the standard deviation of predicted y value.
        y_pred, pred_std = model.predict(x_range, return_std=True)

        # Calculate expected improvement and mask the training position in -np.inf
        pred_std = pred_std.reshape(-1, self.y_dimension)

        # Find out the minimum point in y_train for expected improvement calculation.
        # If the dimension of y larger than 1, L2 distance will be used.
        if self.y_dimension < 2:
            y_min = y_train[np.argmin(y_train)].reshape(-1, self.y_dimension)
        else:
            y_train_l2 = np.sqrt(np.sum(np.power(y_train, 2), axis=1))
            y_min = y_train[np.argmin(y_train_l2)].reshape(-1, self.y_dimension)

        # Calculate the Expected improvement
        ei = (y_min - y_pred) * st.norm.cdf((y_min - y_pred) / pred_std) \
             + pred_std * st.norm.pdf((y_min - y_pred) / pred_std)

        # Find out the index with the maximum EI in EI_np_2d_col_array
        # based on the dimension of y. L2 distance is used for dimension over 1;
        if self.y_dimension < 2:
            # Mask ei to -np.inf in the training points' positions.
            ei = nparray_process(x_train, x_range, ei, process='inf')
        else:
            # For y dimension over 1, calculate L2 distance at first, then Mask ei
            # to -np.inf in the training points' positions.
            temp_ei = np.sqrt(np.sum(np.power(ei, 2), axis=1))
            ei = nparray_process(x_train, x_range, temp_ei, process='inf')
        pred_index = np.argmax(ei)

        # Predict x_next with the maximum EI and print the results.
        x_next = x_range[pred_index].reshape(-1, self.x_dimension)
        self.pred_by_GP.append((x_next.tolist(), np.max(ei).tolist()))
        print('The predicted next x by Gaussian Process is {} with the maximum'
              'Expected_Improvement in {}.'.format(x_next, np.max(ei)))

        # Check the dimension information for plotting.
        dimension = self.x_dimension + self.y_dimension
        if plot_size:
            if dimension ==  2:
                x_range_plot = [x_range, pred_std]
            else:
                x_range_plot = x_range

            title = 'All true training and GP prediction data points'
            plot_legend = ['GP Prediction']
            stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
            save_name = ''.join([stamp, '_predict_next_by_GP.png'])
            plotting(x=x_train, y=y_train, all_x=x_range_plot, all_ptd_y=[y_pred],
                     title=title, zlabel='y', all_true_y=y_true_in_x_range,
                     legend=plot_legend, plot_size=plot_size,
                     save_name=save_name, predicted_point=x_next,
                     dimension=dimension, show=show_plot)
            # For the case of y1, y2=f(x), the EI-x plot should be a 2D plot.
            if self.y_dimension == 2:
                dimension = 2
            if aqf_plot:
                stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
                if isinstance(y_true_in_x_range, list):
                    y_range = y_true_in_x_range[1]
                else:
                    y_range = y_true_in_x_range
                plotting(x=x_range, y=ei, plot_size=plot_size,
                         dimension=dimension,
                         predicted_point=[x_next, y_range],
                         zlabel='y', legend=['EI', 'pred_x_next'],
                         title='EI of Gaussian Process Solver',
                         save_name=''.join([stamp, '_EI_in_all_x.png']),
                         show=show_plot)

        # Tell if it needs to continue to predict x_next by using this solver.
        if n_step > 1:
            # Update the training dataset with put x_next and its y values.
            if callable(orig_func):
                y_next = orig_func(x_next)
            else:
                y_next = np.array([orig_func[i] for i, j in enumerate(x_range)
                                   if (j==x_next).all()]).reshape(1,-1)

            if isinstance(alpha, np.ndarray) and len(self.pred_by_GP):
                tmp_alpha =  np.zeros(len(alpha)+1)
                tmp_alpha[:-1] = alpha
                tmp_alpha[-1] = np.random.uniform(0,1,1)
                alpha = tmp_alpha
                # alpha = np.concatenate(alpha.reshape(1,-1), np.random.uniform(0,1,1).reshape(1,-1))
                
            self.x_train_GP = np.append(self.x_train_GP, x_next)
            self.y_train_GP = np.append(self.y_train_GP, y_next)

            self.predict_by_GP(self.x_train_GP, self.y_train_GP, x_range,
                               n_step-1, aqf_plot, plot_size, show_plot,
                               y_true_in_x_range, constant_value, orig_func,
                               length_scale, alpha, constant_value_bounds,
                               length_scale_bounds, optimizer,
                               n_restarts_optimizer, normalize_y)
        if n_step == 1:
            self.save_parameters()
        return self.pred_by_GP

    def show_nn_models_parameters_in_QBC(self):
        """The method is to get all the parameters of the models in QBC."""
        for model in self.model_list:
            # For TensorFlow Keras model in QBC
            if 'keras' in str(model):
                print(model.summary)
                print(model.get_weights(), '\n')

    def save_parameters(self):
        """Save some parameters of the solver into a json file..
        """
        # Create a dictionary instance to put prepare the paramenters for saving 
        para_dic = {'define_parameters': self.define_parameters,
                    'predict_parameters': self.predict_parameters,
                    'fit_parameters': self.fit_parameters,
                    'x_dimension': self.x_dimension,
                    'y_dimension': self.y_dimension,
                    # 'x_train': self.x_train.tolist(),
                    # 'y_train': self.y_train.tolist(),
                    # 'y_min': self.y_min.tolist(),
                    'pred_by_QBC': self.pred_by_QBC,
                    'pred_by_random': self.pred_by_random,
                    'pred_by_GP': self.pred_by_GP,
                    'pred_by_PSO': self.pred_by_PSO, }
        if len(self.x_train):
            para_dic['x_train']=self.x_train.tolist()
        if len(self.y_train):
            para_dic['y_train']=self.y_train.tolist()
        if self.y_min is not None:
            para_dic['y_min']=self.y_min.tolist()
        if len(self.y_train_PSO):
            para_dic['y_train_PSO']=self.y_train_PSO.tolist()
        if len(self.y_train_PSO):
            para_dic['y_train_PSO']=self.y_train_PSO.tolist()
        # Use time stamp as the key to store the dictionary above since once 
        # a model is propoerly fitted, this method will be called.
        time_stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
        save_content = {time_stamp: para_dic}

        # Write the data into the json file at the current working folder.
        with open('parameters_of_the_solver.json', 'a') as f:
            f.write(json.dumps(str(save_content)))
            f.write('\n')
        print('Parameters are saved at {}'.format(time_stamp))

    def load_parameters(self, json_file=None, data=False):
        """Load parameters to the solver.
        
        Args:
        json_file:
            The file that is created by the save_parameters method.
        data:
            The time stamp (the key) in the json file for loading a assigned 
            group of data. If it is in False, then the data in the last line
            of the json file will be loaded.
        """
        if not json_file:
            json_file = 'parameters_of_the_solver.json'

        with open(os.path.join(os.getcwd(), json_file), 'r') as f:
            if not data:
                last_line = f.readlines()[-2]
                para_dic = json.loads(eval(last_line))
            else:
                for line in f.readlines():
                    if data in line:
                        para_dic = json.loads(eval(line))

            for time_stamp in para_dic:
                para = para_dic[time_stamp]

            self.define_parameters = para['define_parameters']
            self.predict_parameters = para['predict_parameters']
            self.fit_parameters = para['fit_parameters']
            self.x_dimension = para['x_dimension']
            self.y_dimension = para['y_dimension']
            x, y = self.x_dimension, self.y_dimension
            self.x_train = np.array(para['x_train']).reshape(-1, x)[:-1]
            self.y_train = np.array(para['y_train']).reshape(-1, y)[:-1]
            self.y_min = np.array(para['y_min']).reshape(1, -1)
            self.pred_by_QBC = para['pred_by_QBC']
            self.pred_by_random = para['pred_by_random']
            self.pred_by_GP = para['pred_by_GP']
            self.pred_by_PSO = para['pred_by_PSO']
            if 'x_train_PSO' in para.keys():
                self.x_train_PSO = np.array(para['x_train_PSO']).reshape(-1, x)[:-1]
            if 'y_train_PSO' in para.keys():
                self.y_train_PSO = np.array(para['y_train_PSO']).reshape(-1, x)[:-1]

    def save_model(self, model_positions=False, model_names=False, path=False):
        """This method is to save the models in QBC into local disk.
        
        Args:
        model_positions:
            A list of integer smaller than the totall number of models and
            equal or larger than 0. This number indicate the position of the
            model, which is going to be saved. Default in False to save all
            the model in QBC. If it is not in defalut, then model_names
            should also be a list with the same length.
        model_names:
            A list of for the models' name when save them into the disk.
            Default in False will name the models in 'saving_model_' with its 
            order number in the committee. If it is not in defalut, then
            model_positions should also be a list with the same length.
        path:
            Default in False, and it will store the model into the working
            folder. Otherwise, it will store the model in the assigned path.
        """
        if not path:
            path = os.getcwd()

        # For saving TensorFlow model
        if isinstance(model_positions, bool):
            model_positions = [m for m in range(len(self.model_list))]
        elif isinstance(model_positions, int):
            model_positions = [model_positions]

        for i in range(len(model_positions)):
            j = model_positions[i]
            model = self.model_list[j]
            if isinstance(model_names, list):
                name = model_names[i]
            else:
                stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
                if 'keras' in str(model):
                    model_par = self.define_parameters[j]['nn_struc_para_list']
                    nr_hlayers = len(model_par) - 2
                    nr_neurons = sum(model_par[1:-1])
                    name = 'Model_{}_as_Keras_nn_with_{}_hidden_layers_{}_neurons_{}.h5'\
                            .format(str(j+1).zfill(2), nr_hlayers, nr_neurons, stamp)
                    model.save(os.path.join(path, name))
                else:
                    model_type = str(self.model_types[j]).split("'")[1]
                    name = 'Model_{}_as_{}_{}.sav'.format(str(j+1).zfill(2),
                            model_type, stamp)
                    pickle.dump(model, open(name, 'wb'))
            print('{} has been stored in {}' .format(name, path))

    def load_model(self, model_names=False, path=False, key_word=False, n=True):
        """This method is to load the models into QBC from local disk.
        
        Load the lastest stored n models among many models into the QBC.

        Args:
        model_names:
            A list of for the models' name when load them from disk.
            Default in False will load the models named in 'saving_model_'
            with its order number in the path. If it is not in defalut, 
            then the model(s) should be under the assigned path or working
            folder.
        path:
            Default in False, and will search and load the model from the 
            current working folder, otherwise it will search and load the 
            model from the assigned path.
        key_word:
            A string that is contained in the name of the model. Used to filter
            out the folders that are not keras model.
        n:
            Assigned the number of models to be loaded. 
        """
        if not path:
            path = os.getcwd()
        if model_names:
            all_models = model_names
        else:
            if not key_word:
                key_word = 'Model'
            all_models = [i for i in os.listdir(path)
                          if ((key_word in i) and (i[-3:] in ['.h5', '.sav']))]
        # Sorting the models by their creating time.
        sorted_list = sorted(all_models, key=lambda x: os.path.\
                             getmtime(os.path.join(path, x)), reverse=True)
        if isinstance(n, int):
            sorted_list = sorted_list[0:n]
        sorted_list.sort()

        # For loading TensorFlow model process
        for name in sorted_list:
            model = os.path.join(path, name)
            if name.endwith('.h5'):
                self.model_list.append(tf.keras.models.load_model(model))
            else:
                 self.model_list.append(pickle.load(open(model, 'rb')))
            print('{} is loaded'.format(name))
        print('Done! {} models have been loaded.'.format(len(sorted_list)))

    def result_analysis(self, x_range, noised_all_y, obj_f, x_train, y_train,
                        figsize=(12, 6.75), show_plot=True, noise_distance=0):
        """This method is the statistic of the predicted x_next.
        
        Analyze the minimum y and the accpeted minimum y; Load the predictions
        from QBC, Gaussian, Random Guess and PSO solver;  Count how many
        predictions of x_next leading to the accepted minimum. Find out the
        first time that the corresponded solver finds out the accepted_minimum.

        Args:
        x_range:
            It is a np.array instance containing all the points desired
            to be investigated and evaluated as x_next by this solver.
        noised_all_y:
            The y values for all x in x_range, which used to pick out the y_next
            to extend the training set for multiple prediction steps.
        obj_f:
            The original function or noise free y values for a in x_range. If it
            is a function, then it will be used to calculate all the noise free 
            y values.
        x_train:
            It is a np.array instance containing all the initial training data
            in x.
        x_train:
            It is a np.array instance containing all the initial training data
            in y. x_train and y_train are used to plotting purpose.
        figsize:
            Contorl the figures' size when plotting.
        """
        # Find out the dimension information for a single x/y point
        x_d = self.x_dimension
        y_d = self.y_dimension
        output_analysis = []
        # Find out all the noise free y values for calculate the minimums.
        if callable(obj_f):
            all_true_y = obj_f(x_range)
        else:
            all_true_y = obj_f
        x_range = x_range.reshape(-1, x_d)
        all_true_y = all_true_y.reshape(-1, y_d)

        # For y in dimensions over 2, L2 norm is used to locate the minimum
        if y_d == 1:
            y = all_true_y
        elif y_d >= 2:
            y = np.linalg.norm(all_true_y, keepdims=True, axis=1)
        # Find out the minimum and median of y, then get the accepted minimum.
        y_median = np.median(y)
        y_min = np.min(y)
        accept_y_min = y_min + noise_distance #0.02*abs(y_median - y_min)  #noise_level * max_distance_in_y
        # Find out the corresponding x and y for the accepted minimums.
        mask = y <= accept_y_min
        chosen_x =  np.array([x_range[i] for i, j in enumerate(mask) if j])\
                    .reshape(-1, x_d)
        chosen_y =  np.array([y[i] for i, j in enumerate(mask) if j])\
                    .reshape(-1, y_d)

        # Print the minimum related information.
        info1 = 'The true y_min is '+str(y_min)+', but the accepted min ' + \
                'values (y_min + noise_distance) is ' + \
                str(accept_y_min)+ '.\nThere are ' + str(len(chosen_x)) + \
                ' accepted minimum points over x_range.'
        output_analysis.append(info1)
        print(info1)

        # Create a list then append the predictions from different solvers into
        # the list. Used for plotting and further analysis.
        all_data = []
        # pred_by_QBC = []
        # for i in self.pred_by_QBC:
        #     if isinstance(i[0], list): pred_by_QBC.append(i[0][0])
        #     elif isinstance(i, float): pred_by_QBC.append(i)
        #     else: pred_by_QBC.append(i[0])
        # all_data.append(np.array([pred_by_QBC]).reshape(-1,x_d))
        all_data.append(np.array([i for i in self.pred_by_QBC]).reshape(-1,x_d))
        all_data.append(np.array([i[0] for i in self.pred_by_GP]).reshape(-1,x_d))
        all_data.append(np.array([i[0] for i in self.pred_by_random]).reshape(-1,x_d))
        all_data.append(np.array([i[0] for i in self.pred_by_PSO]).reshape(-1,x_d))
        save_name = ['QBC_Solver', 'Gaussian_Process_Solver', 
                     'Random_Guess_Solver','PSO_Solver']

        # Create a list object to contain the statistics of the oredictions of
        # x_next from differetn solvers.
        all_analysis = []
        for n, pred in enumerate(all_data):
            analysis = np.zeros(len(pred))
            # Check how many predictions fall into the accepted minimum
            for m, x_next in enumerate(pred):
                tmp = [1 for point in chosen_x if (x_next==point).all()]
                analysis[m] = sum(tmp)
            all_analysis.append(analysis)

        # Prepare data to plot for x in 2D.
        if x_d == 2:
            nr_point = int(np.sqrt(len(x_range)))
            x10, x20 = np.meshgrid(np.unique(x_range[:,0]), np.unique(x_range[:,0]))
            plt_y = noised_all_y.reshape(nr_point, -1) 

        # Loop in different solvers to plot and print the statistic information
        for n, pred in enumerate(all_data):
            # Check whether there is any prediction of x_next by a solver
            if len(pred):
                # Find out how many predictions, the valid prediction and the
                # first prediction of x_next for a y minimum.
                total_time = len(all_analysis[n])
                in_mini = [p for p, v in enumerate(all_analysis[n]) if v==1]
                ok_time = len(in_mini)
                if in_mini:
                    first_time = int(in_mini[0] + 1)
                else: 
                    first_time = None
                info2 = save_name[n] + ' predict ' + str(total_time) + \
                        ' steps, and there are ' + str(ok_time) + \
                        ' predictions of x_next within the accepted minimum' + \
                        ' range.\nStep_' + str(first_time) + ' is the first' + \
                        ' time of the solver having x_next within the' + \
                        ' accepted minimum range.'
                output_analysis.append(info2)
                print(info2)

                # print('{} predict {} steps, and there are {} predictions of '
                #     'x_next within the accepted minimum range.\nStep_{} '
                #     'is the first time of the solver having x_next within'
                #     ' the accepted minimum range.'. 
                #     format(save_name[n], total_time, ok_time, first_time))

                # Plotting for x in different dimensions
                if x_d <= 2:
                    plt.figure(figsize=figsize)
                    # Plot for 1D case
                    if x_d == 1:
                        plt.plot(x_range, noised_all_y, label='all_data')
                        plt.scatter(x_train, y_train, label='training_points',color='red')
                        plt.scatter(chosen_x, chosen_y, label='all_accepted_min',color='black')
                        y_for_pred = [noised_all_y[i] for j in pred 
                                      for i, k in enumerate(x_range) if k==j]
                        if save_name[n] == 'PSO_Solver':
                            y_for_pred = [obj_f(x) for x in pred]
                        pred = np.array(pred).reshape(-1,1)
                        y_for_pred = np.array(y_for_pred).reshape(-1, y_d)
                        # print(save_name[n], len(pred), len(y_for_pred))
                        plt.scatter(pred, y_for_pred, label=save_name[n], color='blue')
                        for k, point in enumerate(pred):
                            plt.annotate(str(k+1), xy=(point, y_for_pred[k]),
                                               xytext=(point, y_for_pred[k]))
                        plt.grid()
                    # Plot for 2D case
                    elif x_d == 2:
                        surf = plt.contourf(x10, x20, plt_y, cmap=cm.coolwarm,
                                            antialiased=False)
                        plt.colorbar(surf, shrink=0.5, aspect=5)
                        plt.scatter(chosen_x[:,0], chosen_x[:,1], marker='x',
                                    color='black', label='all_accepted_min')
                        plt.scatter(x_train[:,0], x_train[:,1], color='blue',
                                    label='training_points')
                        plt.scatter(pred[:,0], pred[:,1], color='red',
                                    label=save_name[n])
                        for i, point in enumerate(pred):
                            plt.annotate(str(i + 1), xy=(point[0], point[1]),
                                        xytext=(point[0], point[1]))
                    plt.legend()
                    stamp = datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')[:-3]
                    plt.savefig('{}_predict_results_by_{}.png'.
                                format(stamp, save_name[n]))
                    if show_plot:
                        plt.show()
                    else:
                        plt.close()
        # with open('results_analysis.txt', 'a') as f:
        #     for line in output_analysis:
        #         f.write(line)
        #         f.write('\n')
        # print('Result analysis is saved at results_analysis.txt')
        return output_analysis

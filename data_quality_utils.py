import numpy as np
from tqdm import tqdm


cube_distances = {
    'inner': 1,
    'outer': 2
}

hexahedron_distances = {
    'inner_cube': 0.5,
    'outer_wide': 1.5,
    'outer_tall': 1.0
}

threshold = 0
horizontal_rate = 0.5
horizontal_ratio = horizontal_rate * hexahedron_distances['outer_wide'] / hexahedron_distances['inner_cube']
vertical_ratio = 0.5

def get_conductors_num_in_cube(in_file, conductor_index, cube_distance):
    """
    Find the number of conductor points in the cube with conductor_index as center
    
    +---------+
    |         |
    |    o    |
    |         |    
    +---------+
         |----| <- cube distance
    
    Parameters
    ----------
    in_file : laspy.file.File
        LAS file handle
        
    conductor_index : int
        index of the conductor point to be examined
        
    cube_distance : float
        cube distance
        
    Returns
    -------
    int
        number of conductor points in the cube
    """
    
    return get_conductors_num_in_hexahedron(in_file, conductor_index, cube_distance, cube_distance)

    
def get_conductors_num_in_hexahedron(in_file, conductor_index, horizotal_distance, vertical_distance):
# def get_conductors_num_in_cube(in_file, conductor_index, cube_distance):
    """
    Find the number of conductor points in the hexahedron with conductor_index as center
    
    [ Side view ]
    
    +---------+----
    |         |  |  <-- vertical_distance
    |    o    |----
    |         |    
    +---------+
         |----| <- horizotal_distance 
    
    Parameters
    ----------
    in_file : laspy.file.File
        LAS file handle
        
    conductor_index : int
        index of the conductor point to be examined
        
    horizotal_distance : float
        horizotal distance of hexahedron
        
    vertical_distance : float
        vertical distance of hexahedron
        
    Returns
    -------
    int
        number of conductor points in the hexahedron
    """
    conductor_x = in_file.x[conductor_index]
    conductor_y = in_file.y[conductor_index]
    conductor_z = in_file.z[conductor_index]

    inner_hexahedron_x_min = conductor_x - horizotal_distance
    inner_hexahedron_x_max = conductor_x + horizotal_distance
    inner_hexahedron_y_min = conductor_y - horizotal_distance
    inner_hexahedron_y_max = conductor_y + horizotal_distance
    inner_hexahedron_z_min = conductor_z - vertical_distance
    inner_hexahedron_z_max = conductor_z + vertical_distance

    conductors_in_hexahedron_x = np.logical_and((inner_hexahedron_x_min < in_file.x), (in_file.x < inner_hexahedron_x_max))
    conductors_in_hexahedron_y = np.logical_and((inner_hexahedron_y_min < in_file.y), (in_file.y < inner_hexahedron_y_max))
    conductors_in_hexahedron_z = np.logical_and((inner_hexahedron_z_min < in_file.z), (in_file.z < inner_hexahedron_z_max))

    conductors_in_hexahedron = np.logical_and((in_file.Classification == 0), conductors_in_hexahedron_x)
    conductors_in_hexahedron = np.logical_and(conductors_in_hexahedron, conductors_in_hexahedron_y)
    conductors_in_hexahedron = np.logical_and(conductors_in_hexahedron, conductors_in_hexahedron_z)

    conductors_in_hexahedron_indexes = np.where(conductors_in_hexahedron)

    num_conductors_in_hexahedron = np.count_nonzero(conductors_in_hexahedron)
    
    return num_conductors_in_hexahedron


def verify_conductor_points(in_file):
    
    return verify_conductor_points_cube(in_file)

def verify_conductor_points_cube(in_file):
    """
    Verify if each conductor point is a true conductor using 2 cube algorithm
    
    Parameters
    ----------
    in_file : laspy.file.File
        LAS file handle
        
    Returns
    -------
    tuple
        tuple with one element, array of conductor indexes 
            (array([ 25526,  25692,  25788, ..., 544375, 544463, 544507]),)
    
    list
        list of True or False, where True for true positive and False for false positive
        
    list
        list of indexes of false positives
    """
    conductor_indexes = np.where(in_file.Classification == 0)
    
    is_true_conductors = []
    false_conductors_index_list = []
    
    for conductor_index in tqdm(conductor_indexes[0], desc='Conductors'):
        inner_num_conductors = get_conductors_num_in_cube(in_file, conductor_index, cube_distances['inner'])
        outer_num_conductors = get_conductors_num_in_cube(in_file, conductor_index, cube_distances['outer'])

        if outer_num_conductors - inner_num_conductors > threshold:
            is_true_conductors.append(True)
        else:
            is_true_conductors.append(False)
            false_conductors_index_list.append(conductor_index)
            
    return conductor_indexes, is_true_conductors, false_conductors_index_list


def verify_conductor_points_hexahedron(in_file):
    """
    Verify if each conductor point is a true conductor using 1 cube and 2 hexahedron algorithm
    
    Parameters
    ----------
    in_file : laspy.file.File
        LAS file handle
        
    Returns
    -------
    tuple
        tuple with one element, array of conductor indexes 
            (array([ 25526,  25692,  25788, ..., 544375, 544463, 544507]),)
    
    list
        list of True or False, where True for true positive and False for false positive
        
    list
        list of indexes of false positives
    """
    conductor_indexes = np.where(in_file.Classification == 0)
    
    is_true_conductors = []
    false_conductors_index_list = []
    
    for conductor_index in tqdm(conductor_indexes[0], desc='Conductors'):
        
        inner_num_conductors = get_conductors_num_in_cube(in_file, conductor_index, hexahedron_distances['inner_cube'])
        
        outer_horizontal_num_conductors = get_conductors_num_in_hexahedron(in_file, conductor_index, hexahedron_distances['outer_wide'], hexahedron_distances['inner_cube'])
        
        outer_vertical__num_conductors = get_conductors_num_in_hexahedron(in_file, conductor_index, hexahedron_distances['inner_cube'], hexahedron_distances['outer_tall'])

        if (outer_horizontal_num_conductors > inner_num_conductors * horizontal_ratio) and (outer_vertical__num_conductors - inner_num_conductors) <= inner_num_conductors * vertical_ratio :
            is_true_conductors.append(True)
        else:
            is_true_conductors.append(False)
            false_conductors_index_list.append(conductor_index)
    
    return conductor_indexes, is_true_conductors, false_conductors_index_list

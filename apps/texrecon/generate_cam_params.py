import os
import argparse
import configparser

import numpy as np


def generate_cam_files(output_dir, intrinsics_dict, extrinsics_list):
    for idx, extrinsics in enumerate(extrinsics_list):
        R = extrinsics[:3, :3] # Default is row major
        t = extrinsics[:3, -1]
        R = R.ravel() # MVE is row major (No need to transform)
        with open(os.path.join(output_dir, 'image_{:03d}.cam'.format(idx)),
            'w') as fptr:
            fptr.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'
                .format(*(t.tolist()), *(R.tolist())))
            fptr.write('{0} {1} {2} {3} {4} {5}\n'.format(
                intrinsics_dict['focal_length'],
                *intrinsics_dict['radial_distortion'],
                intrinsics_dict['pixel_aspect'],
                *intrinsics_dict['principal_point']))


def generate_mve_metas(base_dir, intrinsics_dict, extrinsics_list):
    dir_list = sorted(os.listdir(base_dir))

    # for idx, extrinsics in enumerate(extrinsics_list):
    for idx, extrinsics in enumerate(extrinsics_list):
        R = extrinsics[:3, :3] # Default is row major
        t = extrinsics[:3, -1]
        R = R.ravel() # MVE is row major (No need to transform)

        file_path = os.path.join(base_dir, dir_list[idx], 'meta.ini')
        config = configparser.ConfigParser()
        config.read(file_path)
        try:
            config_camera = config['camera']
        except:
            config.add_section('camera')
            config_camera = config['camera']
        config_camera['focal_length'] = str(intrinsics_dict['focal_length'])
        config_camera['pixel_aspect'] = str(intrinsics_dict['pixel_aspect'])
        config_camera['principal_point'] = '{} {}'.format(*intrinsics_dict['principal_point'])
        config_camera['radial_distortion'] = '{} {}'.format(*intrinsics_dict['radial_distortion'])
        config_camera['rotation'] = '{0} {1} {2} {3} {4} {5} {6} {7} {8}'.format(*(R.tolist()))
        config_camera['translation'] = '{0} {1} {2}'.format(*(t.tolist()))
        with open(file_path, 'w') as config_file:
            config.write(config_file)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='mve', choices=['mve', 'cam'],
        help='Camera parameter types')
    parser.add_argument('--intrinsics_file', type=str, default='intrinsics.txt', 
        help='Intrinsics file')
    parser.add_argument('--pose_dir', type=str, default=None, 
        help='Pose directory')
    parser.add_argument('--extrinsics_dir', type=str, default=None, 
        help='Extrinsics directory')
    parser.add_argument('--output_dir', type=str, default='images',
        help='Output directory')    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    ''' Argument validity check '''
    if args.pose_dir == None and args.extrinsics_dir == None:
        print('Neither pose nor extrinsics directory is specified.')
        exit(1)

    ''' Generate output folder '''
    os.makedirs(args.output_dir, exist_ok=True)

    ''' Intrinsics from file (Currently only one instrinsics)'''
    intrinsics = np.loadtxt(args.intrinsics_file, dtype=np.float64)
    sensor_width = 36 # Sensor width of standard 35mm camera
    intrinsics_dict = {
        'focal_length': intrinsics[0, 0] / sensor_width,
        'pixel_aspect': 1,
        'principal_point': (intrinsics[0, 2], intrinsics[1, 2]),
        'radial_distortion': (0, 0)
    }

    ''' Load extrinsics directly from files or converted from poses '''
    extrinsics_list = []
    if args.pose_dir != None: # Poses
        for idx, file_name in enumerate(sorted(os.listdir(args.pose_dir))):
            file_path = os.path.join(args.pose_dir, file_name)
            pose = np.loadtxt(file_path, dtype=np.float64)
            R = pose[:3, :3].T
            camera_center = pose[:3, -1]
            t = np.matmul(R, -camera_center)
            # extrinsics = np.linalg.inv(pose)
            extrinsics = np.concatenate((R, np.expand_dims(t, axis=-1)), axis=-1)
            extrinsics_list.append(extrinsics)
    else: # Extrinsics
        for idx, file_name in enumerate(sorted(os.listdir(args.pose_dir))):
            file_path = os.path.join(args.pose_dir, file_name)
            extrinsics = np.loadtxt(file_path, dtype=np.float64)
            extrinsics_list.append(extrinsics)

    ''' Write files with extrinsics & intrinsics'''
    if args.type == 'cam':
        generate_cam_files(args.output_dir, intrinsics_dict, extrinsics_list)
    elif args.type == 'mve':
        generate_mve_metas(args.output_dir, intrinsics_dict, extrinsics_list)
        
import os
import argparse

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()

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
    texrecon_focal_length = intrinsics[0, 0] / sensor_width
    distortions = (0, 0)
    pixel_aspect = 1
    ppx = intrinsics[0, 2]
    ppy = intrinsics[1, 2]
    intrinsics_line = (texrecon_focal_length, *distortions, pixel_aspect, ppx, ppy)

    ''' Generate file based on pose files or extrinsics files '''
    if args.pose_dir != None: # Poses
        for idx, file_name in enumerate(sorted(os.listdir(args.pose_dir))):
            file_path = os.path.join(args.pose_dir, file_name)
            pose = np.loadtxt(file_path, dtype=np.float64)
            extrinsics = np.linalg.inv(pose)
            # extrinsics = pose
            R = extrinsics[:3, :3] # Default is row major
            t = extrinsics[:3, -1]
            # t = pose[:3, -1]
            # t = np.matmul(-R, t)
            R = R.flatten()
            # R = extrinsics[:3, :3].flatten(order='F') # Default is column major
            with open(os.path.join(args.output_dir, 'image_{:03d}.cam'.format(idx)),
                'w') as fptr:
                fptr.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'
                    .format(*(t.tolist()), *(R.tolist())))
                fptr.write('{0} {1} {2} {3} {4} {5}\n'.format(*intrinsics_line))
    else: # Extrinsics
        for idx, file_name in enumerate(sorted(os.listdir(args.pose_dir))):
            file_path = os.path.join(args.pose_dir, file_name)
            extrinsics = np.loadtxt(file_path, dtype=np.float64)
            t = extrinsics[:3, -1]
            R = extrinsics[:3, :3].flatten()
            with open(os.path.join(args.output_dir, 'image_{:03d}.cam'.format(idx)),
                'w') as fptr:
                fptr.write('{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'
                    .format(*(t.tolist()), *(R.tolist())))
                fptr.write('{0} {1} {2} {3} {4} {5}\n'.format(*intrinsics_line))
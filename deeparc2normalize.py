from read_write_model import read_model, write_model, Image, Point3D
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation
import torch

def parse_filename(pattern,file_name):
    [[arc_id, ring_id]] = re.findall(pattern,file_name)
    arc_id = int(arc_id)
    ring_id = int(ring_id)
    return arc_id, ring_id

def camera_position(extirnsic):
    return np.matmul(extirnsic['rotation'].T,extirnsic['translation'])

def write_ply(data, path = 'output.ply'):
    with open(path,'w') as f:
        header = "ply\n" \
                + "format ascii 1.0\n" \
                + "element vertex {}\n" \
                + "property float x\n" \
                + "property float y\n" \
                + "property float z\n" \
                + "property uchar red\n" \
                + "property uchar green\n" \
                + "property uchar blue\n" \
                + "end_header\n"
        f.write(header.format(data.shape[0]))
        for i in range(data.shape[0]):
            color = '255 255 255'
            if i < 42:
                color = '0 255 0'
            line = '{} {} {} {}\n'.format(data[i,0],data[i,1],data[i,2],color)
            f.write(line)

def main(args):
    extrinsics = {}
    cameras, images, points3D = read_model(args.input,'.bin')
    num_arc = 0
    num_ring = 0
    for image_id in images:
        arc, ring = parse_filename(args.pattern,images[image_id][4])
        if arc not in extrinsics:
            extrinsics[arc] = {}
        qvec = images[image_id][1]
        rotation = Rotation.from_quat([qvec[1],qvec[2],qvec[3],qvec[0]])
        extrinsics[arc][ring] = {
            'rotation': rotation.as_matrix(),
            'translation': images[image_id][2]
        }
        if arc+1 > num_arc:
            num_arc = arc+1
        if ring+1 > num_ring:
            num_ring = ring+1
    base_ring = np.zeros((num_ring,3))
    for i in range(num_ring):
        base_ring[i] = camera_position(extrinsics[0][i])
    mean_shift = np.mean(base_ring,axis=0)
    cam_points = np.zeros((num_arc * num_ring,3))
    rotation_matrix = np.zeros((num_ring , 3, 3))
    translation_vector = np.zeros((num_ring, 3))
    #update point3d
    points3D_old = points3D
    points3D = {}
    for point_id in points3D_old:
        points3D[point_id] = Point3D(
            id=points3D_old[point_id][0],
            xyz=points3D_old[point_id][1] - mean_shift,
            rgb=points3D_old[point_id][2],
            error=points3D_old[point_id][3],
            image_ids=points3D_old[point_id][4],
            point2D_idxs=points3D_old[point_id][5]
        )
    # update extrinsic
    for i in range(num_arc):
        for j in range(num_ring):
            extrinsics[i][j]['translation'] -= np.matmul(extrinsics[i][j]['rotation'],mean_shift)
            if i == 0:
                rotation_matrix[i*num_ring+j,:,:] = extrinsics[i][j]['rotation']
                translation_vector[i*num_ring+j,:] = extrinsics[i][j]['translation']
            cam_points[i*num_ring+j] = camera_position(extrinsics[i][j])
    
    
    #############################################
    # TORCH OPIMIZATION!
    #############################################
    rotation_matrix = torch.tensor(rotation_matrix, requires_grad=False) # SHOUDNT UPDATE
    translation_vector = torch.tensor(translation_vector, requires_grad=False) # SHOUDNT UPDATE
    translation_matrix = translation_vector.reshape((-1,3,1))
    rotation_adjuster = torch.tensor([1.0, 0.0,1.0, 0.0,1.0, 0.0],requires_grad=True)
    optimizer = torch.optim.Adam([rotation_adjuster], lr=0.001)
    epoch_count = 0
    previous_loss = 0
    
    while True:
        optimizer.zero_grad()
        base_x = torch.sqrt(rotation_adjuster[0]**2 + rotation_adjuster[1]**2)
        A_x = rotation_adjuster[0] / base_x
        B_x = rotation_adjuster[1] / base_x
        base_y = torch.sqrt(rotation_adjuster[2]**2 + rotation_adjuster[3]**2)
        A_y = rotation_adjuster[2] / base_y
        B_y = rotation_adjuster[3] / base_y
        base_z = torch.sqrt(rotation_adjuster[4]**2 + rotation_adjuster[5]**2)
        A_z = rotation_adjuster[4] / base_z
        B_z = rotation_adjuster[5] / base_z
        Q_x = torch.eye(3)
        Q_x[1,1] = A_x
        Q_x[1,2] = -B_x
        Q_x[2,1] = B_x
        Q_x[2,2] = A_x
        Q_y = torch.eye(3)
        Q_y[0,0] = A_y
        Q_y[0,2] = B_y
        Q_y[2,0] = -B_y
        Q_y[2,2] = A_y
        Q_z = torch.eye(3)
        Q_z[0,0] = A_z
        Q_z[0,1] = -B_z
        Q_z[1,0] = B_z
        Q_z[1,1] = A_z
        rotation_corrector =  Q_z @ Q_y @ Q_x
        camera_rotation = rotation_matrix.clone()
        R_add = rotation_corrector.reshape(1,3,3).repeat(camera_rotation.shape[0],1,1).double()
        camera_rotation = torch.bmm(camera_rotation,R_add)
        cam_points = torch.bmm(camera_rotation.transpose(1,2) , translation_matrix)
        total_loss = torch.sum((cam_points[:,2]) ** 2)
        total_loss.backward(retain_graph=True)
        optimizer.step()
        epoch_count += 1
        if epoch_count % 10 == 0:
            loss_value = total_loss.item()
            print("EPOCH %4d - loss %30.6f" % (epoch_count,loss_value))
            relative_loss = np.abs((loss_value - previous_loss)/loss_value)
            if loss_value < 1e-6 or relative_loss < 1e-6:
                break
            previous_loss = loss_value
    rotation_corrector = rotation_corrector.detach().numpy()
    cam_points = np.zeros((num_arc * num_ring,3))
    for i in range(num_arc):
        for j in range(num_ring):
            extrinsics[i][j]['rotation'] = np.matmul(extrinsics[i][j]['rotation'],rotation_corrector)
            cam_points[i*num_ring+j] = camera_position(extrinsics[i][j])

    ################
    # Write output to colmap bin format
    ##############
    images_old = images
    images = {}
    for image_id in images_old:
        arc, ring = parse_filename(args.pattern,images_old[image_id][4])
        rotation = Rotation.from_matrix(extrinsics[arc][ring]['rotation'])
        q = rotation.as_quat()
        qvec = np.array([q[3],q[0],q[1],q[2]])
        images[image_id] = Image(
            id=image_id,
            qvec=qvec,
            tvec=extrinsics[arc][ring]['translation'],
            camera_id=images_old[image_id][3],
            name=images_old[image_id][4],
            xys=images_old[image_id][5],
            point3D_ids=images_old[image_id][6]
        )
        
    write_model(cameras, images, points3D, args.output, '.bin')
    print("FINISHED!")


def entry_point():
    parser = argparse.ArgumentParser(
        description='deeeparc2normalize.py - convert position of colmap from any position to object stay at -1 to 1')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        #required=True,
        default='C:\\Datasets\\deeparc\\teabottle_green\\model\\distrort_model\\',
        help='colmap model directory / colmap database file (.db)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output/',
        #required=True,
        help='deeparc file output')
    """
    parser.add_argument(
        '-arc',
        '--arc_size',
        type=int,
        #required=True,
        default=10,
        help='arc_size')
    parser.add_argument(
        '-ring',
        '--ring_size',
        type=int,
        #required=True,
        default=41,
        help='ring_size')
    """
    parser.add_argument(
        '-p',
        '--pattern',
        type=str,
        #required=True,
        default='^cam(?:[0-9]+)\\/cam([0-9]+)_([0-9]+)\.(?:png|jpg)$',
        help='file name pattern in regex style (default: \'^cam(?:[0-9]+)\\/cam([0-9]+)_([0-9]+)\.(?:png|jpg)$\' )')
    main(parser.parse_args())

if __name__ == "__main__":
    entry_point()
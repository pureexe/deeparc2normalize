from read_write_model import read_model, write_model, Image, Point3D
import argparse, os, re, torch
import numpy as np
from scipy.spatial.transform import Rotation
from timeit import default_timer as timer

def detect_model(model_path,filetype = '.bin'):
    """ 
        detect is this colmap model directory by detect 3 files
        which is cameras.bin images.bin and points3D.bin
    """
    paths = [
        os.path.join(model_path,'cameras{}'.format(filetype)),
        os.path.join(model_path,'images{}'.format(filetype)),
        os.path.join(model_path,'points3D{}'.format(filetype)),
    ]
    for path in paths:
        if not os.path.exists(path):
            return False
    return True

def parse_filename(pattern,file_name):
    [[arc_id, ring_id]] = re.findall(pattern,file_name)
    arc_id = int(arc_id)
    ring_id = int(ring_id)
    return arc_id, ring_id

def camera_position(extirnsic):
    return np.matmul(extirnsic['rotation'].T,extirnsic['translation'])

def find_rotation_corrector(rotation_matrix,translation_vector):
    """
    find rotation change using PyTorch
    please see to under stand what is Q_x Q_y Q_z
    @see https://en.wikipedia.org/wiki/Rotation_matrix#In_three_dimensions
    @params rotation_matrix, translation_vector
    @return rotation matrix that use for multiply
    """
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
            relative_loss = np.abs((loss_value - previous_loss)/loss_value)
            if loss_value < 1e-6 or relative_loss < 1e-6:
                break
            previous_loss = loss_value
    return rotation_corrector.detach().numpy()

def main(args):
    start_timer = timer()
    extrinsics = {}
    model_extension = ''
    if detect_model(args.input,'.bin'):
        output_extension = '.bin'
    elif detect_model(args.input,'.txt'):
        output_extension = '.txt'
    else: 
        raise RuntimeError('Cannot find colmap sparse model. please check input path')
    cameras, images, points3D = read_model(args.input,output_extension)
    num_arc = 0
    num_ring = 0  
    # read colmap images and get rotation and translation
    for image_id in images:
        arc, ring = parse_filename(args.pattern,images[image_id][4])
        if arc not in extrinsics:
            extrinsics[arc] = {}
        qvec = images[image_id][1]
        rotation = Rotation.from_quat([qvec[1],qvec[2],qvec[3],qvec[0]])
        extrinsics[arc][ring] = {
            'rotation': rotation.as_matrix(),
            'translation': images[image_id][2].copy()
        }
        #find number of arc and number of ring
        if arc+1 > num_arc:
            num_arc = arc+1
        if ring+1 > num_ring:
            num_ring = ring+1

    # find camera position in most bottom ring
    base_ring = np.zeros((num_ring,3))
    for i in range(num_ring):
        base_ring[i] = camera_position(extrinsics[0][i])
    mean_shift = np.mean(base_ring,axis=0)

    # update extrinsic (translation only)
    for i in range(num_arc):
        for j in range(num_ring):
            extrinsics[i][j]['translation'] -= np.matmul(extrinsics[i][j]['rotation'],mean_shift)
    
    #we use only  most bottom camera to optimize
    rotation_matrix = np.zeros((num_ring , 3, 3))
    translation_vector = np.zeros((num_ring, 3))
    for i in range(num_ring):
        rotation_matrix[i,:,:] = extrinsics[0][i]['rotation']
        translation_vector[i,:] = extrinsics[0][i]['translation']

    # optimize to find rotaiton corrector 
    rotation_corrector = find_rotation_corrector(rotation_matrix,translation_vector)

    # update extrinsic (rotation only)
    for i in range(num_arc):
        for j in range(num_ring):
            extrinsics[i][j]['rotation'] = np.matmul(extrinsics[i][j]['rotation'],rotation_corrector)

    # update colmap's images
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
    
    #update colmap's  point3d
    points3D_old = points3D
    points3D = {}
    for point_id in points3D_old:
        points3D[point_id] = Point3D(
            id=points3D_old[point_id][0],
            xyz=np.matmul(rotation_corrector.T, points3D_old[point_id][1] + mean_shift),
            rgb=points3D_old[point_id][2],
            error=points3D_old[point_id][3],
            image_ids=points3D_old[point_id][4],
            point2D_idxs=points3D_old[point_id][5]
        )

    #write to binary output
    if not os.path.exists(args.output):
        os.mkdir(args.output)
    write_model(cameras, images, points3D, args.output, output_extension)
    total_time = timer() - start_timer
    print('Finished in {:.2f} seconds'.format(total_time))
    print('output are write to {}'.format(os.path.abspath(args.output)))

def entry_point():
    parser = argparse.ArgumentParser(
        description='deeeparc2normalize.py - convert position of colmap from any position to object stay at -1 to 1')
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        required=True,
        help='colmap model directory',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=str,
        default='output/',
        help='deeparc file output (default: \'output/\')')
    parser.add_argument(
        '-p',
        '--pattern',
        type=str,
        #required=True,
        default='^cam(?:[0-9]+)\\/cam([0-9]+)_([0-9]+)\.(?:png|jpg)$',
        help='file name pattern in regex style (default: \'^cam(?:[0-9]+)\\/cam([0-9]+)_([0-9]+)\.(?:png|jpg)$\')')
    main(parser.parse_args())

if __name__ == "__main__":
    entry_point()
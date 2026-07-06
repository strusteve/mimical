from ..utils import MLP
from ..utils import SquareIntersectionPredictor
from ..utils import make_nn
import torch
import torchvision
import os

install_dir = os.path.dirname(os.path.realpath(__file__))
netdir = (install_dir + "/utils/neural_networks").replace("/models", "")


class Rotator(object):

    def __init__(self, device="cpu"):

        model = MLP()
        if not os.path.isfile(netdir + "/square_intersection_nn.pth"):
            model = make_nn(model)
        else:
            model.load_state_dict(torch.load(netdir + "/square_intersection_nn.pth", map_location=device))
        self.square_predictor = SquareIntersectionPredictor(model, device)

    def loop(self, images, angles):
        """
        Basic image-cube rotation function, not vectorised but is generally faster
        for single image fits and/or without acceleration. Loops over PyTorch
        torchvision module (similar to scipy.ndimage.rotate).
        """

        rotated_images = images.clone().detach()
        for i in range(len(angles)):
            if angles[i] == 0.:
                continue
            else:
                interpmode = torchvision.transforms.InterpolationMode.BILINEAR
                rotated_images[i] = torchvision.transforms.functional.rotate(
                    images[i].unsqueeze(0), float(angles[i]),
                    interpolation=interpmode)

        return rotated_images

    def mario(self, images, angles, base_x=None, base_y=None, utilnum=None):
        """
        Fully vectorised image-cube rotation function inspired by the rotation of
        sprites in old 8-bit games. Uses three rounded shear matrices to imitate
        rotation. Conserves pixel values but is unstable for coarse and/or unsmooth
        PSFs.
        """

        if (base_x is None) & (base_y is None):
            base_x, base_y = torch.meshgrid(torch.arange(images[0].shape[1],
                                                        device=images.device),
                                            torch.arange(images[0].shape[0],
                                                        device=images.device),
                                            indexing='xy')
        elif (base_x is not None) & (base_y is None):
            raise Exception("Both 'base_x' and 'base_y' must "
                            "be ignored or provided")
        elif (base_x is None) & (base_y is not None):
            raise Exception("Both 'base_x' and 'base_y' must "
                            "be ignored or provided")

        if utilnum is None:
            utilnum = torch.arange(len(angles), device=images.device)

        x = base_x-((images[0].shape[1]-1)/2)
        y = base_y-((images[0].shape[0]-1)/2)

        coords = torch.stack([x.flatten(), y.flatten()])
        coords_all = torch.stack([coords]*len(angles))

        # Find which coords these came from in input image
        anglemask = ((angles < -90) | (angles > 90))[0]
        images[anglemask] = images.flip(dims=(1, 2))[anglemask]
        angles[anglemask] -= torch.sign(angles[anglemask]) * 180
        theta = torch.deg2rad(angles)
        alpha = -torch.tan(theta/2)
        beta = torch.sin(theta)

        # Perform rotation by three shear operations, rounding the shear value
        coords_all[:, 0] = coords_all[:, 0] + torch.round((alpha *
                                                        coords_all[:, 1].T)).T
        coords_all[:, 1] = coords_all[:, 1] + torch.round((beta *
                                                        coords_all[:, 0].T)).T
        coords_all[:, 0] = coords_all[:, 0] + torch.round((alpha *
                                                        coords_all[:, 1].T)).T

        # Centre and integer coords
        coords_all[:, 0] += (images[0].shape[1]-1)/2
        coords_all[:, 1] += (images[0].shape[0]-1)/2
        coords_all = coords_all.to(torch.int)

        # Reshape pixel coord array for masking
        dim0 = ((coords_all[:, 0]*0).T + utilnum).T.flatten()
        dim1 = coords_all[:, 1].flatten()
        dim2 = coords_all[:, 0].flatten()
        final_rot = torch.stack((dim0, dim1, dim2))

        # Get the original values of each pixel
        padded_images = torch.nn.functional.pad(images, (0, 100, 0, 100))
        values = padded_images[*final_rot]

        # Mask out boundaries and reshape
        mask = ((final_rot[1] < 0) |
                (final_rot[1] > images[0].shape[0]-1) |
                (final_rot[2] < 0) |
                (final_rot[2] > images[0].shape[1]-1))
        values[mask] = 0
        fluxes = values.reshape(*images.shape)

        return fluxes

    def nearest_neighbour(self, images, angles, base_x=None, base_y=None,
                                utilnum=None):
        """
        Fully vectorised image-cube rotation function that rotates via a matrix
        then performs grid sampling using the nearst pixel value. Generally faster
        than 'loop-rotation' for multi-band accelerated fits, but not as accurate
        and can have duplicate pixel values.
        """

        if (base_x is None) & (base_y is None):
            base_x, base_y = torch.meshgrid(torch.arange(images[0].shape[1],
                                                        device=images.device),
                                            torch.arange(images[0].shape[0],
                                                        device=images.device),
                                            indexing='xy')
        elif (base_x is not None) & (base_y is None):
            raise Exception("Both 'base_x' and 'base_y' must "
                            "be ignored or provided")
        elif (base_x is None) & (base_y is not None):
            raise Exception("Both 'base_x' and 'base_y' must "
                            "be ignored or provided")

        if utilnum is None:
            utilnum = torch.arange(len(angles), device=images.device)

        # Rotate output coords to origin input coords
        x = base_x-((images[0].shape[1]-1)/2)
        y = base_y-((images[0].shape[0]-1)/2)
        coords = torch.stack([x.flatten(), y.flatten()])
        coords_all = torch.stack([coords]*len(angles))
        theta = torch.deg2rad(angles)
        rotM = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)]),
                            torch.stack([torch.sin(theta), torch.cos(theta)])])
        rotM = torch.flip(rotM,
                        dims=(1, 0)).swapaxes(0, 2).reshape(len(theta), 2, 2)
        origin_coords = (rotM @ coords_all)
        origin_coords[:, 0] += (images[0].shape[1]-1)/2
        origin_coords[:, 1] += (images[0].shape[0]-1)/2
        origin_coords = origin_coords.reshape(origin_coords.shape[0],
                                            origin_coords.shape[1],
                                            images.shape[1],
                                            images.shape[2])

        # Snap rotated output coords to nearest input pixel
        coords_all = torch.round(origin_coords).to(torch.int)

        # Reshape pixel coord array for masking
        dim0 = ((coords_all[:, 0]*0).permute(2, 1, 0) +
                utilnum).permute(2, 1, 0).flatten()
        dim1 = coords_all[:, 1].flatten()
        dim2 = coords_all[:, 0].flatten()
        final_rot = torch.stack((dim0, dim1, dim2))

        # Get the original values of each pixel
        padded_images = torch.nn.functional.pad(images, (0, 100, 0, 100))
        values = padded_images[*final_rot]

        # Mask out boundaries and reshape
        mask = ((final_rot[1] < 0) |
                (final_rot[1] > images[0].shape[0]-1) |
                (final_rot[2] < 0) |
                (final_rot[2] > images[0].shape[1]-1))
        values[mask] = 0
        fluxes = values.reshape(*images.shape)

        return fluxes

    def intersection(self, images, angles, base_x=None, base_y=None, utilnum=None):
        """
        Fully vectorised image-cube rotation function that rotates via a matrix
        then performs grid sampling using sub-functions to find the summing weights
        (overlap areas) of the 9 origin pixels describing the rotated pixels local
        region. Ideally this is done using the intersection polygon of rotated
        squares, however until I figure out how to do this in a fully vectorised
        manner over an image cube, the current method uses the intersection area of
        identical circles of unity area, which performs simlarly to interpolation.
        Generally faster than 'loop-rotation' for multi-band accelerated fits.
        """

        if (base_x is None) & (base_y is None):
            base_x, base_y = torch.meshgrid(torch.arange(images[0].shape[1],
                                                        device=images.device),
                                            torch.arange(images[0].shape[0],
                                                        device=images.device),
                                            indexing='xy')
        elif (base_x is not None) & (base_y is None):
            raise Exception("Both 'base_x' and 'base_y' must "
                            "be ignored or provided")
        elif (base_x is None) & (base_y is not None):
            raise Exception("Both 'base_x' and 'base_y' must "
                            "be ignored or provided")

        if utilnum is None:
            utilnum = torch.arange(len(angles), device=images.device)

        # Rotate output coords to origin input coords
        x = base_x-((images[0].shape[1]-1)/2)
        y = base_y-((images[0].shape[0]-1)/2)
        coords = torch.stack([x.flatten(), y.flatten()])
        coords_all = torch.stack([coords]*len(angles))
        theta = torch.deg2rad(angles)
        rotM = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)]),
                            torch.stack([torch.sin(theta), torch.cos(theta)])])
        rotM = torch.flip(rotM,
                        dims=(1, 0)).swapaxes(0, 2).reshape(len(theta), 2, 2)
        origin_coords = (rotM @ coords_all)
        origin_coords[:, 0] += (images[0].shape[1]-1)/2
        origin_coords[:, 1] += (images[0].shape[0]-1)/2
        origin_coords = origin_coords.reshape(origin_coords.shape[0],
                                            origin_coords.shape[1],
                                            images.shape[1],
                                            images.shape[2])

        # Snap rotated output coords to nearest input pixel
        origin_coords_rounded = torch.round(origin_coords).to(torch.int)

        # Expand about nearest input pixel
        expanded_origin = torch.zeros((*origin_coords_rounded.shape, 3, 3),
                                    device=images.device)
        expanded_origin = (expanded_origin.permute(5, 4, 3, 2, 1, 0) +
                        origin_coords_rounded.permute(3, 2, 1, 0))
        expanded_origin = expanded_origin.permute(5, 4, 3, 2, 1, 0)
        expanded_origin[:, 0, :, :, :, :] += torch.tensor([[-1, 0, 1],
                                                        [-1, 0, 1],
                                                        [-1, 0, 1]],
                                                        device=images.device)
        expanded_origin[:, 1, :, :, :, :] += torch.tensor([[1, 1, 1],
                                                        [0, 0, 0],
                                                        [-1, -1, -1]],
                                                        device=images.device)

        # Reorder expanded neighbour pixel grid for indexing
        dim0 = ((expanded_origin[:, 0]*0).permute(4, 3, 2, 1, 0) +
                utilnum).permute(4, 3, 2, 1, 0).flatten()
        dim1 = expanded_origin[:, 1].flatten()
        dim2 = expanded_origin[:, 0].flatten()
        expanded_origin_dimcoords = torch.stack((dim0, dim1, dim2)).to(torch.int)

        # Find the flux values of each origin pixel in expanded grid
        pad = int(1.5 * (0.5**0.5-0.5) * max(*images.shape[1:]))
        padded_images = torch.nn.functional.pad(images, (0, pad, 0, pad))
        values = padded_images[*expanded_origin_dimcoords]

        # Set any outside original image boundaries to be zero
        mask = ((expanded_origin_dimcoords[1] < 0) |
                (expanded_origin_dimcoords[1] > images[0].shape[0]-1) |
                (expanded_origin_dimcoords[2] < 0) |
                (expanded_origin_dimcoords[2] > images[0].shape[1]-1))
        values[mask] = 0

        def weight_neighbours_circleintersection(origin_coords, expanded_origin):
            """ Weight pixels in the local region by circular overlap. """
            # Circle of area 1 centred on each square
            radius = 1 / (3.14159265358979323846**0.5)
            separation_xy = (expanded_origin.permute(5, 4, 3, 2, 1, 0) -
                            origin_coords.permute(3, 2, 1, 0))
            separation_xy = separation_xy.permute(5, 4, 3, 2, 1, 0)
            separation = torch.sum(separation_xy**2, dim=1)
            intersecting_area = (2 * (radius**2) *
                                torch.arccos(separation / (2*radius))) - \
                                (0.5 * separation *
                                torch.sqrt((4*(radius**2)) - (separation**2)))
            norm = torch.nansum(intersecting_area, dim=(3, 4)).permute(2, 1, 0)
            intersecting_area = (intersecting_area.permute(4, 3, 2, 1, 0) /
                                norm).permute(4, 3, 2, 1, 0)
            intersecting_area[separation >= (2*radius)] = 0
            return intersecting_area
        
        def weight_neighbours_squareintersection(origin_coords, expanded_origin):
            """ Weight pixels in the local region by square overlap. 
                As there is no simple analytical formula for this, a
                neural network is used to determine the overlap area. """
            separation_xy = (expanded_origin.permute(5, 4, 3, 2, 1, 0) -
                            origin_coords.permute(3, 2, 1, 0))
            separation_xy = separation_xy.permute(5, 4, 3, 2, 1, 0)
            dxs = separation_xy[:,0,:,:,:,:]
            dys = separation_xy[:,1,:,:,:,:]
            thetas = ((dxs.clone().detach() * 0).permute(4,3,2,1,0) + theta).permute(4,3,2,1,0)
            intersecting_area = self.square_predictor.predict(dxs.flatten(), dys.flatten(), thetas.flatten())
            intersecting_area = intersecting_area.reshape(*dxs.shape)
            return intersecting_area
        
        # Get summing weights of all neighbour input pixels
        weights = weight_neighbours_squareintersection(origin_coords, expanded_origin)

        # Sum over neighbour fluxes to find the total flux
        values = values.reshape(*weights.shape)
        fluxcontribs = values * weights
        fluxes = torch.sum(fluxcontribs, dim=(-2, -1))

        print(fluxes.device)

        return fluxes

from ..utils import MLP
from ..utils import SquareIntersectionPredictor
from ..utils import make_nn
import torch
import os
import torch.nn.functional as F

install_dir = os.path.dirname(os.path.realpath(__file__))
netdir = (install_dir + "/utils/neural_networks").replace("/models", "")


class Rotator(object):
    """ General image rotation class. """

    def __init__(self, device="cpu"):

        model = MLP()
        if not os.path.isfile(netdir + "/square_intersection_nn.pth"):
            model = make_nn(model)
        else:
            model.load_state_dict(torch.load(netdir +
                                             "/square_intersection_nn.pth",
                                             map_location=device))
        self.square_predictor = SquareIntersectionPredictor(model, device)

        offset_y, offset_x = torch.meshgrid(
        torch.tensor([-1, 0, 1], device=device),
        torch.tensor([-1, 0, 1], device=device),
        indexing="ij")
        self.offsets = torch.stack((offset_x, offset_y), dim=0)

    def interpolation(self, images, angles):
        """
        Fully vectorised image-cube rotation function based on the
        classic interpolation method.
        """

        N, H, W = images.shape

        images4 = images.unsqueeze(1)  # (N,1,H,W)

        theta = torch.zeros(
            (N, 2, 3),
            device=images.device,
            dtype=images.dtype)

        # Build affine-rotation matrix 
        theta[:, 0, 0] = torch.cos(torch.deg2rad(angles))
        theta[:, 0, 1] = -torch.sin(torch.deg2rad(angles))
        theta[:, 1, 0] = torch.sin(torch.deg2rad(angles))
        theta[:, 1, 1] = torch.cos(torch.deg2rad(angles))

        # Compute pytorch flowfield grid
        grid = F.affine_grid(
            theta,
            images4.shape,
            align_corners=False
        )

        # Sample pytorch flowfield grid
        rotated = F.grid_sample(
            images4,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False
        )

        return rotated[:, 0]

    def mario(self, images, angles, base_x=None, base_y=None, utilnum=None):
        """
        Fully vectorised image-cube rotation function inspired by the rotation
        of sprites in old 8-bit games. Uses three rounded shear matrices to
        imitate rotation. Conserves pixel values but is unstable for coarse
        and/or unsmooth PSFs.
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

        base_x = base_x-((images[0].shape[1]-1)/2)
        base_y = base_y-((images[0].shape[0]-1)/2)

        coords = torch.stack([base_x.flatten(), base_y.flatten()])
        coords = torch.stack([coords]*len(angles))

        # Find which coords these came from in input image
        anglemask = ((angles < -90) | (angles > 90))[0]
        images[anglemask] = images.flip(dims=(1, 2))[anglemask]
        angles[anglemask] -= torch.sign(angles[anglemask]) * 180
        theta = torch.deg2rad(angles)
        alpha = -torch.tan(theta/2)
        beta = torch.sin(theta)

        # Perform rotation by three shear operations, rounding the shear value
        coords[:, 0] = coords[:, 0] + torch.round((alpha *
                                                   coords[:, 1].T)
                                                  ).T
        coords[:, 1] = coords[:, 1] + torch.round((beta *
                                                   coords[:, 0].T)
                                                  ).T
        coords[:, 0] = coords[:, 0] + torch.round((alpha *
                                                   coords[:, 1].T)
                                                  ).T

        # Centre and integer coords
        coords[:, 0] += (images[0].shape[1]-1)/2
        coords[:, 1] += (images[0].shape[0]-1)/2
        coords = coords.to(torch.int)

        # Reshape pixel coord array for masking
        dim0 = ((coords[:, 0]*0).T + utilnum).T.flatten()
        dim1 = coords[:, 1].flatten()
        dim2 = coords[:, 0].flatten()
        final_rot = torch.stack((dim0, dim1, dim2))

        # Get the original values of each pixel
        padded_images = torch.nn.functional.pad(images, (0, 100, 0, 100))
        fluxes = padded_images[*final_rot]

        # Mask out boundaries and reshape
        mask = ((final_rot[1] < 0) |
                (final_rot[1] > images[0].shape[0]-1) |
                (final_rot[2] < 0) |
                (final_rot[2] > images[0].shape[1]-1))
        fluxes[mask] = 0
        fluxes = fluxes.reshape(*images.shape)

        return fluxes

    def nearest_neighbour(self, images, angles, base_x=None,
                          base_y=None, utilnum=None):
        """
        Fully vectorised image-cube rotation function that rotates via a matrix
        then performs grid sampling using the nearst pixel value. Generally
        faster than 'loop-rotation' for multi-band accelerated fits, but not
        as accurate and can have duplicate pixel values.
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
        base_x = base_x-((images[0].shape[1]-1)/2)
        base_y = base_y-((images[0].shape[0]-1)/2)
        coords = torch.stack([base_x.flatten(), base_y.flatten()])
        coords = coords.unsqueeze(0).expand(len(angles), -1, -1)
        theta = torch.deg2rad(angles)
        rotM = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)]),
                            torch.stack([torch.sin(theta), torch.cos(theta)])])
        rotM = torch.flip(rotM,
                          dims=(1, 0)).swapaxes(0, 2).reshape(len(theta), 2, 2)
        coords = (rotM @ coords)
        coords[:, 0] += (images[0].shape[1]-1)/2
        coords[:, 1] += (images[0].shape[0]-1)/2
        coords = coords.reshape(coords.shape[0],
                                coords.shape[1],
                                images.shape[1],
                                images.shape[2])

        # Snap rotated output coords to nearest input pixel
        coords = torch.round(coords).to(torch.int)

        # Reshape pixel coord array for masking
        dim0 = utilnum[:, None, None].expand_as(coords[:, 0]).flatten()
        dim1 = coords[:, 1].flatten()
        dim2 = coords[:, 0].flatten()
        final_rot = torch.stack((dim0, dim1, dim2))

        # Get the original values of each pixel
        padded_images = torch.nn.functional.pad(images, (0, 100, 0, 100))
        fluxes = padded_images[*final_rot]

        # Mask out boundaries and reshape
        mask = ((final_rot[1] < 0) |
                (final_rot[1] > images[0].shape[0]-1) |
                (final_rot[2] < 0) |
                (final_rot[2] > images[0].shape[1]-1))
        fluxes[mask] = 0
        fluxes = fluxes.reshape(*images.shape)

        return fluxes

    def intersection(self, images, angles, base_x=None,
                     base_y=None, utilnum=None, type='square'):
        """
        Fully vectorised image-cube rotation function that rotates via a matrix
        then performs grid sampling using sub-functions to find the summing
        weights (overlap areas) of the 9 origin pixels describing the rotated
        pixels local region. Ideally this is done using the intersection
        polygon of rotated squares, however until I figure out how to do this
        in a fully vectorised manner over an image cube, the current method
        uses the intersection area of identical circles of unity area, which
        performs simlarly to interpolation. Generally faster than
        'loop-rotation' for multi-band accelerated fits.
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
        base_x = base_x-((images[0].shape[1]-1)/2)
        base_y = base_y-((images[0].shape[0]-1)/2)
        coords = torch.stack([base_x.flatten(), base_y.flatten()])
        oords = coords.unsqueeze(0).expand(len(angles), -1, -1)
        theta = torch.deg2rad(angles)
        rotM = torch.stack([torch.stack([torch.cos(theta), -torch.sin(theta)]),
                            torch.stack([torch.sin(theta), torch.cos(theta)])])
        rotM = torch.flip(rotM,
                          dims=(1, 0)).swapaxes(0, 2).reshape(len(theta), 2, 2)
        coords = (rotM @ coords)
        coords[:, 0] += (images[0].shape[1]-1)/2
        coords[:, 1] += (images[0].shape[0]-1)/2
        coords = coords.reshape(coords.shape[0],
                                coords.shape[1],
                                images.shape[1],
                                images.shape[2])

        # Snap rotated output coords to nearest input pixel
        origin_coords_rounded = torch.round(coords).to(torch.int)

        # Expand about nearest input pixel
        expanded_origin = (origin_coords_rounded[..., None, None] +
                           self.offsets[None, :, None, None, :, :])

        # Reorder expanded neighbour pixel grid for indexing
        dim0 = utilnum[:, None, None, None, None].expand_as(expanded_origin[:, 0]).flatten()
        dim1 = expanded_origin[:, 1].flatten()
        dim2 = expanded_origin[:, 0].flatten()
        expanded_origin_dimcoords = torch.stack((dim0,
                                                 dim1,
                                                 dim2)).to(torch.int)

        # Find the flux values of each origin pixel in expanded grid
        pad = int(1.5 * (0.5**0.5-0.5) * max(*images.shape[1:]))
        padded_images = torch.nn.functional.pad(images, (0, pad, 0, pad))
        fluxes = padded_images[*expanded_origin_dimcoords]

        # Set any outside original image boundaries to be zero
        mask = ((expanded_origin_dimcoords[1] < 0) |
                (expanded_origin_dimcoords[1] > images[0].shape[0]-1) |
                (expanded_origin_dimcoords[2] < 0) |
                (expanded_origin_dimcoords[2] > images[0].shape[1]-1))
        fluxes[mask] = 0

        def weight_neighbours_circleintersection(origin_coords,
                                                 expanded_origin):
            """ Weight pixels in the local region by circular overlap. """
            # Circle of area 1 centred on each square
            radius = 1 / (3.14159265358979323846**0.5)
            separation_xy = (expanded_origin.float()
                             - origin_coords[..., None, None])
            separation = torch.sum(separation_xy**2, dim=1)

            intersecting_area = (2 * (radius**2) *
                                 torch.arccos(separation / (2*radius))) - \
                                (0.5 * separation *
                                 torch.sqrt((4*(radius**2)) - (separation**2)))
            
            norm = torch.nansum(intersecting_area, dim=(-2, -1), keepdim=True)
            intersecting_area = intersecting_area / norm
            intersecting_area[separation >= (2*radius)] = 0
            return intersecting_area

        def weight_neighbours_squareintersection(origin_coords,
                                                 expanded_origin):
            """ Weight pixels in the local region by square overlap.
                As there is no simple analytical formula for this, a
                neural network is used to determine the overlap area. """
            separation_xy = (expanded_origin.float()
                             - origin_coords[..., None, None])

            dxs = separation_xy[:, 0]
            dys = separation_xy[:, 1]

            thetas = theta[:,None,None,None,None]

            intersecting_area = self.square_predictor.predict(dxs.reshape(-1),
                                                              dys.reshape(-1),
                                                              thetas.expand_as(dxs).reshape(-1))
            return intersecting_area.reshape(dxs.shape)

        if type == 'circle':
            # Get summing weights of all neighbour input pixels
            weights = weight_neighbours_circleintersection(coords,
                                                           expanded_origin)
        elif type == 'square':
            # Get summing weights of all neighbour input pixels
            weights = weight_neighbours_squareintersection(coords,
                                                           expanded_origin)
        else:
            raise Exception("Type must be either 'square' or 'circle'.")

        # Sum over neighbour fluxes to find the total flux
        fluxes = fluxes.reshape(*weights.shape)
        fluxes = fluxes * weights
        fluxes = torch.sum(fluxes, dim=(-2, -1))

        return fluxes

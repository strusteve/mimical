import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
import os

from ...models import ImageModel
from ...models import Sersic

device = 'mps'

install_dir = os.path.dirname(os.path.realpath(__file__))

def make_oversampling_table():
    # Grid in 'r_eff' and 'n' parameter space
    n_array = torch.arange(0.1,10.1,0.1)
    np.savetxt(install_dir + '/n_values.txt', n_array, fmt='%.1f')
    r_eff_array = np.append(0.1, np.arange(1,21,1))
    np.savetxt(install_dir + '/r_eff_values.txt', r_eff_array, fmt='%.1f')

    # Table to append oversampling factors to
    tabledat = np.zeros((len(r_eff_array), len(n_array), 3))

    # Loop over 'r_eff' and 'n' parameter space
    print('One-time generation of oversampling table...')
    for i in tqdm(range(len(r_eff_array))):
        for j in range(len(n_array)):
            
            # Make perfect reference image
            reference_model = ImageModel(torch.arange(101, device=device), torch.arange(101, device=device), [Sersic()], None, 0., oversample=[10000, 100, 50], oversample_radii=[1, max(2, r_eff_array[i]), max(3, 3*r_eff_array[i])])
            reference_model.update_parameters(torch.tensor([1000, r_eff_array[i], n_array[j], 50, 50, 0, 0]).to(torch.float32).to(device=device).unsqueeze(0), 0)
            reference_image = reference_model.render().cpu()
    

            # Initiate starting factor and radii values
            osam = [1,1,1]
            orad = [1, max(2, r_eff_array[i]), max(3, 3*r_eff_array[i])]

            # Loop over radii 
            for k in range(3):
                
                # While the maximum residual is greater than one 1000th the maximum reference image value.
                while True:
                    
                    # Make current model

                    model = ImageModel(torch.arange(101, device=device), torch.arange(101, device=device), [Sersic()], None, 0., oversample=osam, oversample_radii=orad)
                    model.update_parameters(torch.tensor([1000, r_eff_array[i], n_array[j], 50, 50, 0, 0]).to(torch.float32).to(device=device).unsqueeze(0), 0)
                    image = model.render().cpu()

                    # Calculate residuals with reference model
                    residual = torch.abs(reference_image - image) / torch.max(reference_image)

                    # Create mask for pixels within current radii
                    base_xgrid, base_ygrid = torch.meshgrid(torch.arange(101), torch.arange(101), indexing='xy')
                    centred_base_xgrid = base_xgrid - 50
                    centred_base_ygrid = base_ygrid - 50
                    # If first radii, include centre
                    if k == 0:
                        curr_mask = (centred_base_xgrid**2 + centred_base_ygrid**2 <= orad[k]**2)
                    # Else, mask in annuli
                    else:
                        curr_mask = (centred_base_xgrid**2 + centred_base_ygrid**2 <= orad[k]**2) & (centred_base_xgrid**2 + centred_base_ygrid**2 > orad[k-1]**2)

                    # If no pixels in current radii, skip
                    if torch.sum(curr_mask)==0:
                        break
                    
                    # If criterion reached or maxed out, break
                    if (torch.max(residual[0][curr_mask]) < 0.01) | (osam[k]==1000):
                        break
                    # If not, continue
                    else:
                        osam[k]+=1
                        continue
            
            
            # Save table
            tabledat[i,j] = osam

            np.savetxt(install_dir + '/table1_values.txt', tabledat[:,:,0], fmt='%.0f')
            np.savetxt(install_dir + '/table2_values.txt', tabledat[:,:,1], fmt='%.0f')
            np.savetxt(install_dir + '/table3_values.txt', tabledat[:,:,2], fmt='%.0f')




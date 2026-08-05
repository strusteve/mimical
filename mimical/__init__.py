import os
if "MIMICAL_BANNER_SHOWN" not in os.environ:
    os.environ["MIMICAL_BANNER_SHOWN"] = "1"
    print(r""" __    __     __     __    __     __     ______     ______     __        """ + "\n"
          r"""/\ "-./  \   /\ \   /\ "-./  \   /\ \   /\  ___\   /\  __ \   /\ \       """ + "\n"
          r"""\ \ \-./\ \  \ \ \  \ \ \-./\ \  \ \ \  \ \ \____  \ \  __ \  \ \ \____  """ + "\n"
          r""" \ \_\ \ \_\  \ \_\  \ \_\ \ \_\  \ \_\  \ \_____\  \ \_\ \_\  \ \_____\ """ + "\n"
          r"""  \/_/  \/_/   \/_/   \/_/  \/_/   \/_/   \/_____/   \/_/\/_/   \/_____/ """ + "\n"
             "------------------------------------------------------------------------" + "\n"
             "Modelling the Intensity of Multiply-Imaged CelestiAl Light"+ "\n"
             "https://github.com/strusteve/mimical\n")

from . import fitting
from . import plotting
from . import models
from . import priors

from .fitting import fit
from .fitting import fitCatalogue

from .models import ImageModel
from .models import Sersic, Point

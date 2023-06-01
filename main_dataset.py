

import utils.dataset
import utils.convert_hdf5

if __name__ == '__main__':
    utils.dataset.main()  # Create data patches
    utils.convert_hdf5.main()  # Create hdf5 files
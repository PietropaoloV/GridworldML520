import subprocess
import shutil
import sys
import os.path as path
import numpy as np


def main():
    # parse command line arguments
    project = sys.argv[1]  # set to project1, project2, or project3
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    num_iter = sys.argv[4]

    # java command to run
    java_path = shutil.which('java')
    class_path = path.abspath('../old_projects/class')
    cmd = [java_path, '-cp', class_path, f'{project}.Simulator', str(x), str(y), num_iter]

    # run simulation
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    grids = p.stdout

    # encode data into numpy arrays
    grid_array = np.fromstring(grids, sep=' ', dtype=np.byte).reshape((-1, y, x))

    # write output
    np.save(f'{project}-grids', grid_array)


if __name__ == '__main__':
    main()

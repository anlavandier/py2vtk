try:
    import mpi4py

    from .api import *
except ImportError as e:
    msg = (
        "Py2VTK's MPI submodule requirements are not installed.\n\n"
        "Please pip install as follows:\n\n"
        '   python -m pip install "py2vtk[mpi]" --upgrade'
    )
    raise ImportError(str(e) + "\n\n" + msg) from e

import pytest
from weave.codes.base import NoiseModel
from weave.util import pcm

@pytest.fixture
def zero_noise():
    # A NoiseModel with all noise levels set to zero.
    return NoiseModel(data=0.0, z_check=0.0, x_check=0.0, circuit=0.0, crossing=0.0)

@pytest.fixture
def repetition_clist():
    # Generate a clist for a simple repetition code (n=4)
    H = pcm.repetition(4)  # H is (3,4)
    clist = pcm.to_clist(H)
    return clist

@pytest.fixture
def hamming_clist():
    # Generate a clist for a Hamming code (n=7, for example)
    H = pcm.hamming(7)
    clist = pcm.to_clist(H)
    return clist

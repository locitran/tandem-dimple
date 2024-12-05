__version__ = '1.0'
__author__ = 'Loci Tran'
__email__ = 'quangloctrandinh1998vn@gmail.com'
__github__ = ''

from pathlib import Path

one2three = {
    'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS', 'E': 'GLU', 'Q': 'GLN',
    'G': 'GLY', 'H': 'HIS', 'I': 'ILE', 'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE',
    'P': 'PRO', 'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
}
three2one = {v: k for k, v in one2three.items()}
standard_aa = list(one2three.values())
ROOT_DIR = Path(__file__).resolve().parents[2]
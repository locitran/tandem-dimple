from propka import run
import pandas as pd
import logging
import traceback
logger = logging.getLogger(__name__)

def getPropKa(pdbPath, pHCondition=7.0, write_pka=False):
    "Get pKa data from PDB file"
    try:
        mol = run.single(pdbPath, write_pka=write_pka)
    except Exception:
        logger.exception(f"> TANDEM:PropKa: {traceback.format_exc()}")
        return None

    groups = mol.conformations['AVR'].groups
    pka_charge = []
    for group in groups:
        if (group.charge > 0) == (group.pka_value > pHCondition):
            pka_charge.append(group.charge)
        else:
            pka_charge.append(0)

    propkaData = [(group.residue_type, group.atom.res_num, group.atom.icode.strip(), group.atom.chain_id, group.pka_value, pka_charge[i]) 
            for i, group in enumerate(groups)]
    propkaData = pd.DataFrame(propkaData, columns=['resName','resID', 'iCode', 'chainID','pKa','pka_charge'])
    propkaData = propkaData.groupby(['resName', 'chainID', 'resID', 'iCode'])['pka_charge'].sum().reset_index(level=None)
    return propkaData

# pdbPath = '1G0D.pdb'
# propkaData = getPropKa(pdbPath, pHCondition=7.0, write_pka=False)
# print(propkaData)

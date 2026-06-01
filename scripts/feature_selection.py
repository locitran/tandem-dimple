import os
import sys
addpath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, addpath) # /tandem/
os.chdir(addpath)

from src.train.train import reproduce_foundation_model
from src.utils.settings import TANDEM_GJB2, TANDEM_RYR1, TANDEM_R20000, CLUSTER
from src.features import TANDEM_FEATS, dynamics_feat, structure_feat, sequence_feat, rhapsody_feat, evolution_feat, chemical_feat

if __name__ == "__main__":

    reproduce_foundation_model(
        name='feature-selection/seq-str-dyn',
        featset=list(sequence_feat.keys())+list(dynamics_feat.keys())+list(structure_feat.keys()),
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )

    reproduce_foundation_model(
        name='feature-selection/seq',
        featset=list(sequence_feat.keys()),
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )

    reproduce_foundation_model(
        name='feature-selection/evo',
        featset=list(evolution_feat.keys()),
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )

    reproduce_foundation_model(
        name='feature-selection/chem',
        featset=list(chemical_feat.keys()),
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )

    reproduce_foundation_model(
        name='feature-selection/str',
        featset=list(structure_feat.keys()),
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )


    reproduce_foundation_model(
        name='feature-selection/dyn',
        featset=list(sequence_feat.keys()),
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )

    reproduce_foundation_model(
        name='feature-selection/rhapsody-2020-8features',
        featset=list(rhapsody_feat.keys()),
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )

    reproduce_foundation_model(
        name='feature-selection/tandem-33features',
        featset=TANDEM_FEATS['v1.1'],
        featds=TANDEM_R20000,
        gjb2ds=TANDEM_GJB2,
        ryr1ds=TANDEM_RYR1,
        clstr=CLUSTER,
    )
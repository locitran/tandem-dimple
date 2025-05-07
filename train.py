from src.train.optimization import test_numberOflayers_TANDEM, test_numberOflayers_RHAPSODY, test_ranking_method
from src.train.optimization import test_batch_size, test_different_numberOfneurons, visualization_optimization
# from src.train.transfer_learning import train_model
from src.train.direct_train import train_model
from src.utils.settings import TANDEM_GJB2, TANDEM_RYR1
if __name__ == "__main__":
    # train_model(
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="RYR1",
    #     seed=0,
    # )

    train_model(
        TANDEM_testSet=TANDEM_GJB2,
        name="GJB2",
        seed=73,
    )
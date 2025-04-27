from src.train.optimization import test_numberOflayers_TANDEM, test_numberOflayers_RHAPSODY, test_ranking_method
from src.train.optimization import test_batch_size, test_different_numberOfneurons, visualization_optimization
from src.train.transfer_learning import train_model
from src.utils.settings import TANDEM_GJB2, TANDEM_RYR1
if __name__ == "__main__":
    # test_ranking_method(seed=17)
    # test_numberOflayers_TANDEM(seed=17)
    # test_numberOflayers_RHAPSODY(seed=17)
    # test_batch_size(seed=17)
    # test_different_numberOfneurons(seed=17)

    train_model(
        base_models="/mnt/nas_1/YangLab/loci/tandem/logs/final/different_number_of_layers/20250423-1234-tandem/n_hidden-5",
        TANDEM_testSet=TANDEM_GJB2,
        name="GJB2",
        seed=17,
    )
    # train_model(
    #     base_models="/mnt/nas_1/YangLab/loci/tandem/logs/final/different_number_of_layers/20250418-1127-tandem/n_hidden-5",
    #     TANDEM_testSet=TANDEM_RYR1,
    #     name="RYR1",
    #     seed=100,
    # )

    # visualization_optimization(
    #     "/mnt/nas_1/YangLab/loci/tandem/logs/Optimization_Tandem_NumberOfLayers/20250417-1610",
    #     "/mnt/nas_1/YangLab/loci/tandem/logs/Optimization_Rhapsody_NumberOfLayers/20250417-1611",
    #     hidden_layers=[5, 6, 8, 10, 12],
    # )
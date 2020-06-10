from cascade import Cascador


config = {
    'name': "face",

    # Different dataset using different reading method
    'dataset': "I",
    'version': "1.0",
    'stageNum': 4,

    'regressorPara':
        {
            'name': 'lbf_reg',
            'para':
                {
                    'maxTreeNums': [100],
                    'treeDepths': [4],
                    'feaNums': [1000, 750, 500, 375, 250],
                    'radiuses': [0.4, 0.3, 0.2, 0.15, 0.12],
                    # Following para is used to quantize the feature
                    'binNums': [511],
                    'feaRanges': [[-255, 255]],
                }
        },

    'dataPara':
        {
            'path': "./data/I/",

            # augNum < 1 means don't do augmenting
            'augNum': 0
        }
}
cascade = Cascador()
cascade.config(config)
save_path = './'
cascade.train(save_path)
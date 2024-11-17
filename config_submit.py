config = {'datapath':r'C:\Users\20192032\ownCloud\CT images\patient X',
          'preprocess_result_path':r'C:\Users\20192032\ownCloud\CT images\prep_results',
          'outputfile':'prediction.csv',
          'output_feature': True,
          
          'detector_model':'net_detector',
         'detector_param':'./model/detector.ckpt',
         'classifier_model':'net_classifier',
         'classifier_param':'./model/classifier.ckpt',
         'n_gpu':0,
         'n_worker_preprocessing':1,
         'use_exsiting_preprocessing':False,
         'skip_preprocessing':False,
         'skip_detect':False}

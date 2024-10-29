import random
import numpy as np
import os
import logging
import xarray as xr

class ValidityDomain:
    cluster: int = 10
    test_nb: int = 20
    envir_bin : dict = {
        'simu_hs': 1,
        'simu_tp': 1,
        'simu_dp':30,
        'simu_mag10': 2,
        'simu_theta10': 30
        }

    def find_test_set_in_model_validity_domain(self, df):
        """This function is used to find a valid test set.
        The validity domain of the model is defined by the environmental bin.
        A test sample is within validity domain of the model if it exist a training sample "close" to the test sample.
        "Close" means that all the environmental variables of the test sample are within the user defined environmental bin.
        """
        
        count = 0
        found=False
        max_guesses_allowed = 200
        
        log =  logging.getLogger('log')
        log.info('#####')
        log.info("start guessing valid training / test set with the following environmental bin :")
        log.info(str(self.envir_bin))
        log.info(f'looking for {self.test_nb} test samples in training set, divided in {self.cluster} clusters ')
        
        training_list = np.arange(0,len(df.time))
        div = int(len(training_list)/self.cluster)
        test_number_per_cluster = int(self.test_nb/self.cluster)
  
        while not found and count < max_guesses_allowed:

            # generate random test list
            test_index = []
            for i in range(self.cluster) :
                if i != self.cluster-1:
                    test_index += random.sample(range(i*div, (i+1)*div), test_number_per_cluster)
                else:
                    if not (self.test_nb/self.cluster) % 1 == 0:
                        test_number_per_cluster+=1
                    test_index += random.sample(range(i*div, len(training_list)), test_number_per_cluster)
                
            test_index.sort()
            df_test = df.isel(time=test_index)

            # remove test samples from training set
            df_training = df.drop(test_index, errors='ignore')

            nb_training_sample_in_bin = []

            # check if test set is valid, i.e there is at least one valid training sample within envir_bin
            for test_time in df_test.time.values :
                df_valid = df_training
                list_var=''
                for envir_var in self.envir_bin.keys() :
                    df_valid = self.get_valid_training_samples_for_one_test_sample_on_one_variable(df_valid, df_test, test_time, envir_var, self.envir_bin)
                    list_var = list_var + ' & ' + envir_var
                nb_training_sample_in_bin.append(len(df_valid.time))
                

            nb_training_sample_in_bin_dict = {
                "attrs":{
                    "description" : "int representing the number of valid training sample for the current test set"},
                "dims" : "time",
                "data" : nb_training_sample_in_bin,
                "name" : "nb_training"
            } 
            DataArray_nb_training  =  xr.DataArray.from_dict(nb_training_sample_in_bin_dict)
            Dataset_nb_training = xr.Dataset(
                { "nb_training" : DataArray_nb_training},
                coords={"time" : df_test.time.values}
            )
            df_test = xr.merge([df_test, Dataset_nb_training], combine_attrs="drop_conflicts")
            
            # If all test_set has at least one training sample in validity domain, return the test and training set
            if float(df_test.nb_training.min().values) >= 1 :
                found = True
                print(' Success : test_set found. returning valid dataset with len {} vs. training set '.format(str(len(df_test.time))))
                return df_training, df_test
            elif count%10 == 0 :
                print(f'all guess until {count} failed. Keep trying')
            count +=1
        print('Did not found one valid training sample for each test set')
        print(f'nb of test : {max_guesses_allowed}')
        return False, False


    def get_valid_training_samples_for_one_test_sample_on_one_variable(self, df_training, df_test, test_time_index, variable, envir_bin) :
        # catch key error on df_test : 
        
        # Prepare filters :
        try : 
            lte = df_test[variable].sel(time=test_time_index).values + envir_bin[variable]
            gte = df_test[variable].sel(time=test_time_index).values - envir_bin[variable]
        except KeyError :
            
            log =  logging.getLogger(os.environ['logger_name'])
            log.info('KeyError : variable {} not found in df_test'.format(variable))
            return False
            
        lte = float(lte)
        gte = float(gte)
        
        # return values between lte and gte
        df_valid = df_training.isel(time = (df_training[variable]>gte) & (df_training[variable]<lte))
        
        return df_valid
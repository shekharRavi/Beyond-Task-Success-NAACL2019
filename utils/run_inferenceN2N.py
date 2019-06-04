import os

#This to run inference n number of times
#TODO: We could in a code itself
for ep in range(65,80):
    bin_path = 'bin/SL/model_ensemble_mod6Dec_E_%s'%(ep)
    os.system('python3 -m train.GamePlay.inference -dataparallel -load_bin_path '+bin_path)

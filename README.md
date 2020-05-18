[![license](https://img.shields.io/badge/license-GPL%203.0-green.svg)](https://github.com/u1234x1234/AutoSpeech2020/blob/master/LICENSE)

# AutoSpeech2020
1st place solution

## Usage

```bash
git clone https://github.com/u1234x1234/AutoSpeech2020.git
```

class `Model` from the file `model.py` satisfies the [interface](https://www.automl.ai/competitions/2#learn_the_details-evaluation), so you could just run the following line in order to reproduce the results:
```bash
python run_local_test.py -dataset_dir=path_to_dataset -code_dir=path_to_model_file
```
Please refer to the official detailed [description](https://www.automl.ai/competitions/2#learn_the_details-evaluation) of the evaluation protocol.


## How it works

The basic ideas:
* Get a decent result as fast as possible with the simplest models, then train more elaborate ones.
* There's no single model that performs the best on the all datasets, so try different ones.

Used models:
1. Logistic Regression on features extracted with pretrained model [trained on the speaker recognition task](https://github.com/clovaai/voxceleb_trainer/)
2. Logistic Regression on features extracted with pretrained model [trained on the music genre classification task](https://github.com/jordipons/musicnn)
3. Logistic Regression on the combination of features from 1. 2.
4. [AutoSpeech 2019 1st place solution by Hazza Cheng](https://github.com/HazzaCheng/AutoSpeech)
5. [AutoSpeech 2019 3rd place Solution by Kon](https://github.com/Y-oHr-N/autospeech19)
6. Fine-tuning of pretrained network from 1.

Then average the results from multiple models with geometric mean.


## Acknowledgements
 
1. [AutoSpeech 2019 1st place solution by Hazza Cheng](https://github.com/HazzaCheng/AutoSpeech), [GPL Licence](https://github.com/HazzaCheng/AutoSpeech/blob/master/LICENSE), [Code modifications](./AutoSpeech2019_1.diff)
2. [AutoSpeech 2019 3rd place Solution by Kon](https://github.com/Y-oHr-N/autospeech19), [MIT Licence](https://github.com/Y-oHr-N/autospeech19/blob/master/LICENSE)
3. [Speaker recognition pretrained model by ClovaAI](https://github.com/clovaai/voxceleb_trainer), [MIT Licence](https://github.com/clovaai/voxceleb_trainer/blob/master/LICENSE.md)
4. [Musicnn by Jordi Pons](https://github.com/jordipons/musicnn), [ISC Licence](https://github.com/jordipons/musicnn/blob/master/LICENSE.md)

## Notice

Each subdirectory from `3rdparty` contains subcomponents with separate copyright notices and license terms. 
Please refer to Licence provided in specific subdirectory.

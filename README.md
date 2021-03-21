# flow-based-CoopNets

## Usage

### Environment Setup
  1. Make sure you have [Anaconda or Miniconda](https://conda.io/docs/download.html)
  installed.
  2. Clone repo with `git clone https://github.com/chrischute/glow.git glow`.
  3. Go into the cloned repo: `cd glow`.
  4. Create the environment: `conda env create -f environment.yml`.
  5. Activate the environment: `source activate glow`.

### Train
  1. Make sure you've created and activated the conda environment as described above.
  2. Run `python train.py -h` to see options.
  3. Run `python train.py [FLAGS]` to train. *E.g.,* run
  `python train.py` for the default configuration, or run
  `python train.py --gpu_ids=0,1` to run on
  2 GPUs instead of the default of 1 GPU. This will also double the batch size.
  4. At the end of each epoch, samples from the model will be saved to
  `samples/epoch_N.png`, where `N` is the epoch number.

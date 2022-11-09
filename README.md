# Your task

This project was initially used for my thesis and abandomed midway due to low priority. Hence, I am looking forward for your suggestions
to finish up this implementation. The basic setup is done. You need to validate and test all components and reproduce the results of
the original paper. As a tip, I would definatly start with the MixtureOfExperts model and compare it to the original implementation.
(My suggestion would be to drop the batch normalization first and integrate it back after all other sources of error are eliminated.)
Please read the Appendix in the original work, where most implentation details are hidden. Thanks for your help! I am looking forward for
your PRs :)).

Note: This project is still WIP. Benchmarks and further details will be added soon.
# PyTorch implementation Dynamics-Aware Discovery of Skills (DADS).
The ``dads-torch`` repository contains all modules needed for the implementation of [DADS](https://arxiv.org/abs/1907.01657).
In addition, all routines for off-policy training and zero-shot execution in MujoCo environments are supported.
The RL algorithm SAC used here is a modification of the implementation from [pranz24](https://github.com/pranz24/pytorch-soft-actor-critic).

### Installation
Just clone the repository and run ``pip`` with the needed requirements. Note that a valid MuJoCo 2.1.0 installation is required.

    git clone https://github.com/freiberg-roman/dads-torch.git
    pip install -e ".[dev]"

To test your installation, run

    python -m pytest tests/

### Run experiments
Change the required test point directory in the ``off_policy.yaml`` file and run

    python -m dads.main.main mode=dads_train

then change the directory in the ``zero_shot.yaml`` file to the required directory containing the trained networks and
execute

    python -m dads.main.main mode=dads_eval

to evaluate your networks in a concrete environment. If you are familiar with Hydra, you can also manipulate all entries from the command line :)

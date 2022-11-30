from active_learning.strategies import random, uncertainty, bayesian_uncertainty


def build_query(args):
    if args.al_strategy.name == "random":
        query = random.RandomSampling()
    elif args.al_strategy.name == "uncertainty":
        query = uncertainty.UncertaintySampling(uncertainty_type=args.al_strategy.uncertainty_type)
    else:
        raise NotImplementedError(f"{args.al_strategy.name} is not implemented!")
    return query
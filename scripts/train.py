from argparse import ArgumentParser


def get_arg():
    arg_parser = ArgumentParser()
    # program parameter
    arg_parser.add_argument("--network", type=str, default=None, required=True, help="the network to be used")
    arg_parser.add_argument("--name", type=str, default="default_name", required=False, help="the run instance's name")
    arg_parser.add_argument("--logdir", type=str, default=None, help="tensorboard log directory")

    # network parameter
    arg_parser.add_argument("--epoch", type=int, default=50, help="epochs to be used")
    arg_parser.add_argument("--lr", type=int, default=1e-4, help="learning rate")
    arg_parser.add_argument("--momentum", type=int, default=0.4, help="optimizer momentum")
    arg_parser.add_argument("--l2", type=int, default=1e-4, help="l2 penalty")
    
    opt = arg_parser.parse_args()
    return vars(opt)


def main():
    pass


if __name__ == "__main__":
    config = get_arg()
    # model
    if config["network"] == "unet":
        pass
    elif config["network"] == 

    # tensorboard

    main()
    
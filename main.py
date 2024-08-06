import argparse
import sys
from src.single_pendulum.data_gen.generator import generate_single_pendulum_data
from src.double_pendulum.data_gen.generator import generate_double_pendulum_data
from src.single_pendulum.train.training import train_single_pendulum
from src.double_pendulum.train.training import train_double_pendulum

def main():
    parser = argparse.ArgumentParser(description="Pendulum Simulation and Training CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Generator parsers
    gen_parser = subparsers.add_parser("generate", help="Generate pendulum data")
    gen_parser.add_argument("pendulum_type", choices=["single", "double"], help="Type of pendulum to generate data for")
    gen_parser.add_argument("--output_path", type=str, help="Path to save the generated data")
    gen_parser.add_argument("--integration_mode", type=str, default="stormer_verlet", help="Integration method for single pendulum")

    # Training parsers
    train_parser = subparsers.add_parser("train", help="Train pendulum model")
    train_parser.add_argument("pendulum_type", choices=["single", "double"], help="Type of pendulum to train on")

    args = parser.parse_args()

    if args.command == "generate":
        if args.pendulum_type == "single":
            generate_single_pendulum_data(args.output_path, args.integration_mode)
        elif args.pendulum_type == "double":
            generate_double_pendulum_data(args.output_path, args.integration_mode)
    elif args.command == "train":
        if args.pendulum_type == "single":
            train_single_pendulum()
        elif args.pendulum_type == "double":
            train_double_pendulum()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
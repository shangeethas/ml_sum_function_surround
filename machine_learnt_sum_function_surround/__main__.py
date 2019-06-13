import argparse
from surround import Assembler
from stages import Main, ValidateData
from batch_runner import BatchRunner


def main():
    parser = argparse.ArgumentParser(prog='Machine_learnt_sum_function_surround', description="Surround mode(s) available to run this module")
    parser.add_argument('--mode', help="Mode to run (train, batch)", default="batch")
    execute_assembler(parser.parse_args().mode)


def execute_assembler(mode):
    assembler = Assembler("Machine Learnt Sum Function", ValidateData(), Main())
    if mode == "train":
        BatchRunner(assembler).run(True)
    else:
        BatchRunner(assembler).run()


if __name__ == "__main__":
    main()

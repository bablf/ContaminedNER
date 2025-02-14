import argparse
from pathlib import Path

from args import train_argparser, eval_argparser
from config_reader import process_configs
from diffusionner import input_reader
from diffusionner.diffusionner_trainer import DiffusionNERTrainer
import warnings

warnings.filterwarnings("ignore")


def __train(run_args):
    trainer = DiffusionNERTrainer(run_args)
    trainer.train(train_path=run_args.train_path, valid_path=run_args.valid_path,
                  types_path=run_args.types_path, input_reader_cls=input_reader.JsonInputReader)
    run_args.model_path = run_args.save_path + "/final_model"
    for test_dataset in run_args.test_paths:
        test_dataset_path = Path(test_dataset)
        dataset_label = test_dataset_path.stem.split("_", 1)[1]
        trainer.eval(
            dataset_path=test_dataset, types_path=run_args.types_path,
            input_reader_cls=input_reader.JsonInputReader, dataset_label=dataset_label
        )


def _train():
    arg_parser = train_argparser()
    process_configs(target=__train, arg_parser=arg_parser)


def __eval(run_args):
    trainer = DiffusionNERTrainer(run_args)
    trainer.eval(dataset_path=run_args.dataset_path, types_path=run_args.types_path,
                 input_reader_cls=input_reader.JsonInputReader)


def _eval():
    arg_parser = eval_argparser()
    process_configs(target=__eval, arg_parser=arg_parser)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument('mode', type=str, help="Mode: 'train' or 'eval'")
    args, _ = arg_parser.parse_known_args()

    if args.mode == 'train':
        _train()
    elif args.mode == 'eval':
        _eval()
    else:
        raise Exception("Mode not in ['train', 'eval'], e.g. 'python diffusionner.py train ...'")

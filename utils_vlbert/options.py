import abc
import argparse

class arguments(abc.ABC):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()
        self.args = self.parser.parse_args()

    @abc.abstractmethod
    def add_arguments(self):
        pass
    
    @property
    def get_args(self):
        return self.args
    
    @property
    def print_args(self):
        for arg in vars(self.args):
            print(f"{arg}: {getattr(self.args, arg)}")
            
class TrainingArguments(arguments):
    def add_arguments(self):
        self.parser.add_argument("--json_file_path", type=str, required=True, help="Path to the JSON file")
        self.parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name")
        self.parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        self.parser.add_argument("--max_length", type=int, default=128, help="Max length of the input")
        self.parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
        self.parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
        self.parser.add_argument("--model_output_path", type=str, required=True, help="Path to save the model")

class TestingArguments(arguments):
    def add_arguments(self):
        self.parser.add_argument("--json_file_path", type=str, required=True, help="Path to the JSON file")
        self.parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model name")
        self.parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
        self.parser.add_argument("--max_length", type=int, default=128, help="Max length of the input")
        self.parser.add_argument("--generate_report", action="store_true", help="Generate report")
        self.parser.add_argument("--save_report_path", type=str, default="./", help="Path to save the report")
        self.parser.add_argument("--save_report_name", type=str, default="report", help="Name of the report")
        self.parser.add_argument("--criterion_name", type=str, nargs='+', choices=['accuracy', 'f1', 'precision', 'recall', 'weighted_accuracy', 'specificity', 'sensitivity', 'AUC'], default=['weighted_accuracy', 'specificity', 'sensitivity', 'AUC'], help="Criterion name")
import argparse

parser = argparse.ArgumentParser(description='Time-Series Anomaly Detection')
parser.add_argument('--dataset', 
					metavar='-d', 
					type=str, 
					required=False,
					default='SWaT',
                    help='dataset from ["SWaT", "NAB", "UCR", "MBA", "SMAP", "SMD", "MSL", "WADI"]')
parser.add_argument('--model',
					metavar='-m',
					type=str,
					required=False,
					default='MCFMAAE',
                    help="model name")
parser.add_argument('--test', 
					action='store_true',
					help="test the model")
parser.add_argument('--retrain', 
					action='store_true', 
					help="retrain the model")
parser.add_argument('--less', 
					action='store_true',
					help="train using less data")
parser.add_argument('--epochs',
					type=int,
					default=10,
					help="train round"
					)

args = parser.parse_args()


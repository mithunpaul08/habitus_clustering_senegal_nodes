import argparse
argparser=argparse.ArgumentParser("to parse causal documents")
argparser.add_argument("input_file_name", help="name of the input file where causal and effect variables are kept")
args=argparser.parse_args()
print((args.input_file_name))


# import argparse
# parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('integers', metavar='N', type=int, nargs='+',
#                     help='an integer for the accumulator')
# parser.add_argument('--sum', dest='accumulate', action='store_const',
#                     const=sum, default=max,
#                     help='sum the integers (default: find the max)')
#
# args = parser.parse_args()
# print((args.integers)